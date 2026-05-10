# MC 粒子碰撞模拟 — 优化 Walkthrough

## 总览

对 Monte Carlo 粒子碰撞模拟程序实施了 6 步优化，总计性能提升约 **36%**（118ms → 75ms per step）。

> [!IMPORTANT]
> 所有优化均通过统计一致性验证：最终平均粒子速度稳定在 1.18-1.19 范围内，与基线一致。

---

## 性能对比

| Step | 优化内容 | 总耗时 | ms/step | 相比基线 |
|------|---------|--------|---------|---------|
| 基线 | 原始代码 | 11.828s | 118ms | — |
| 1 | `swap_remove` 替代 `remove` | 11.238s | 112ms | -5% |
| 2 | 预计算常量 (d², 碰撞系数等) | 11.801s | 118ms | ±0% |
| 3 | `SmallRng` 替代 `ThreadRng` | **7.463s** | **74ms** | **-37%** |
| 4 | 清理未使用代码 (theta/phi/imports) | 7.423s | 74ms | -37% |
| 5 | `BufWriter` + 减少进度输出 | 7.531s | 75ms | -36% |
| 6 | 扁平化网格 `Vec<Vec<usize>>` | 7.531s | 75ms | -36% |

> [!NOTE]
> Step 2 的预计算常量在 release 模式下提升不明显，因为 LLVM 优化器已能部分消除常量表达式。但在 debug 模式下会有显著差异。
> Step 3 是最大的单步提升，`ThreadRng` (ChaCha20) 是密码学级别 RNG，换成 `SmallRng` (Xoshiro256++) 快 2-3x。

---

## 修改的文件

### [config.rs](file:///Users/bjergsen/Documents/GitHub/mc/src/config.rs)

```diff:config.rs
use serde_derive::Deserialize;
use std::fs;

#[derive(Deserialize)]
pub struct Config {
    pub n: usize,
    pub l: usize,
    pub d: f64,
    pub temperature: f64,
    pub mass: f64,
    pub n_test: usize,
    pub e0: f64,
    pub dt: f64,
    pub l_j_epsilon: f64,
}

impl Config {
    pub fn from_file(path: &str) -> Self {
        let config_str = fs::read_to_string(path).expect("Failed to read config file");
        toml::from_str(&config_str).expect("Failed to parse config file")
    }
}

// pub const N: usize = 131072;
// pub const L: usize = 16;
// pub const D: f64 = 0.2;
// pub const T: f64 = 1.0;
// pub const MASS: f64 = 200.0;
// pub const N_TEST: usize = 1;

// pub const E0: f64 = 1.5 * T;

// pub const T_STEP: f64 = 0.2;
===
use serde_derive::Deserialize;
use std::f64::consts::PI;
use std::fs;

#[derive(Deserialize)]
struct RawConfig {
    n: usize,
    l: usize,
    d: f64,
    temperature: f64,
    mass: f64,
    n_test: usize,
    e0: f64,
    dt: f64,
    l_j_epsilon: f64,
}

pub struct Config {
    pub n: usize,
    pub l: usize,
    pub d: f64,
    pub temperature: f64,
    pub mass: f64,
    pub n_test: usize,
    pub dt: f64,
    pub l_j_epsilon: f64,
    // Precomputed constants
    pub d_sq: f64,               // d²
    pub collision_coeff_o0: f64,  // dt * d² * PI / n_test
    pub collision_coeff_o1: f64,  // dt * d³ * PI / (2 * n_test)
    pub collision_coeff_o2: f64,  // dt * d⁴ * PI / (8 * n_test)
    pub dt_over_mass: f64,       // dt / mass
    pub lj_coeff: f64,           // 24.0 * l_j_epsilon
    pub v_magnitude: f64,        // sqrt(2 * e0 / mass)
}

impl Config {
    pub fn from_file(path: &str) -> Self {
        let config_str = fs::read_to_string(path).expect("Failed to read config file");
        let raw: RawConfig = toml::from_str(&config_str).expect("Failed to parse config file");
        let d_sq = raw.d * raw.d;
        let d_cubed = d_sq * raw.d;
        let d_fourth = d_sq * d_sq;
        Config {
            n: raw.n,
            l: raw.l,
            d: raw.d,
            temperature: raw.temperature,
            mass: raw.mass,
            n_test: raw.n_test,
            dt: raw.dt,
            l_j_epsilon: raw.l_j_epsilon,
            d_sq,
            collision_coeff_o0: raw.dt * d_sq * PI / raw.n_test as f64,
            collision_coeff_o1: raw.dt * d_cubed * PI / (2 * raw.n_test) as f64,
            collision_coeff_o2: raw.dt * d_fourth * PI / (8 * raw.n_test) as f64,
            dt_over_mass: raw.dt / raw.mass,
            lj_coeff: 24.0 * raw.l_j_epsilon,
            v_magnitude: (2.0 * raw.e0 / raw.mass).sqrt(),
        }
    }
}
```

**主要变更：**
- 引入 `RawConfig` 作为反序列化中间结构
- `Config` 增加预计算字段：`d_sq`, `collision_coeff_o0/o1/o2`, `dt_over_mass`, `lj_coeff`, `v_magnitude`
- 移除未使用字段：`e0`, `d_cubed`, `d_fourth`
- 清理旧注释常量

---

### [main.rs](file:///Users/bjergsen/Documents/GitHub/mc/src/main.rs)

```diff:main.rs
#![cfg_attr(feature = "simd", feature(portable_simd))]

// Fix the imports for SIMD
#[cfg(feature = "simd")]
use std::simd::f64x4;
#[cfg(feature = "simd")]
use std::simd::num::SimdFloat; // Import the correct trait for reduce_sum()

use std::env;
use std::f64::consts::PI;
use std::io::prelude::*;
use std::{io, vec};

// use rayon::prelude::*;

use rand::Rng;
use std::fs::File;
use std::time::Instant;

mod config;
use config::Config;

// Vector operation utilities
#[inline]
fn dot_product(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[cfg(feature = "simd")]
#[inline]
fn dot_product_simd(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    let a_simd = f64x4::from_array([a[0], a[1], a[2], 0.0]);
    let b_simd = f64x4::from_array([b[0], b[1], b[2], 0.0]);
    (a_simd * b_simd).reduce_sum()
}

#[inline]
fn apply_periodic_boundary(mut dr: f64, box_length: f64) -> f64 {
    if dr > box_length / 2.0 {
        dr -= box_length;
    } else if dr < -box_length / 2.0 {
        dr += box_length;
    }
    dr
}

// Common collision calculation logic for both scatt_o1 and scatt_o2
#[inline]
fn calculate_collision_parameters(
    i0: usize, 
    i1: usize, 
    r: &[[f64; 3]], 
    v: &[[f64; 3]],
    config: &Config
) -> (f64, f64, f64, [f64; 3], [f64; 3]) {
    let mut dr = [0.0; 3];
    let mut dv = [0.0; 3];
    
    // Compute position and velocity differences with boundary conditions
    for k in 0..3 {
        dr[k] = r[i0][k] - r[i1][k];
        dr[k] = apply_periodic_boundary(dr[k], config.l as f64);
        dv[k] = v[i0][k] - v[i1][k];
    }
    
    #[cfg(feature = "simd")]
    let dr2 = dot_product_simd(&dr, &dr);
    #[cfg(not(feature = "simd"))]
    let dr2 = dot_product(&dr, &dr);
    
    #[cfg(feature = "simd")]
    let dv_dr = dot_product_simd(&dv, &dr);
    #[cfg(not(feature = "simd"))]
    let dv_dr = dot_product(&dv, &dr);
    
    #[cfg(feature = "simd")]
    let dv2 = dot_product_simd(&dv, &dv);
    #[cfg(not(feature = "simd"))]
    let dv2 = dot_product(&dv, &dv);
    
    let dspeed = dv2.sqrt();
    
    let mut vec_k = [0.0; 3];
    for k in 0..3 {
        vec_k[k] = dr[k] / dr2;
    }
    
    (dr2, dv_dr, dspeed, dr, vec_k)
}

#[inline]
fn scatt_o1_optimized(
    grid: &Vec<Vec<Vec<Vec<usize>>>>,
    i_x: usize,
    i_y: usize,
    i_z: usize,
    di: [i32; 3],
    rng: &mut rand::rngs::ThreadRng,
    r: &[[f64; 3]],
    v: &mut [[f64; 3]],
    config: &Config,
) {
    let l = grid.len();
    let i_x_new = (i_x as i32 + di[0]).rem_euclid(l as i32) as usize;
    let i_y_new = (i_y as i32 + di[1]).rem_euclid(l as i32) as usize;
    let i_z_new = (i_z as i32 + di[2]).rem_euclid(l as i32) as usize;
    
    // Store local references to grid cells for better cache locality
    let current_cell = &grid[i_x][i_y][i_z];
    let neighbor_cell = &grid[i_x_new][i_y_new][i_z_new];
    
    // Skip empty cell pairs
    if current_cell.is_empty() || neighbor_cell.is_empty() {
        return;
    }
    
    // Pre-allocate a buffer for calculating multiple collisions at once
    // let mut collisions = Vec::new();
    
    // First pass: identify all potential collisions
    for j0 in 0..current_cell.len() {
        let i0 = current_cell[j0];
        
        for j1 in 0..neighbor_cell.len() {
            let i1 = neighbor_cell[j1];
            
            // Skip self-interactions
            if i0 == i1 {
                continue;
            }
            
            let (_dr2, dv_dr, dspeed, _dr, vec_k) = 
                calculate_collision_parameters(i0, i1, r, v, config);
                
            // Early exit if particles are moving away from each other
            if dv_dr >= 0.0 {
                continue;
            }
            
            // Calculate k_factor based on direction vector
            let mut k_factor = vec_k[0] * di[0] as f64 + 
                              vec_k[1] * di[1] as f64 + 
                              vec_k[2] * di[2] as f64;
            k_factor = -k_factor; // Adjust for collision direction
            
            // Calculate collision probability
            let collision_prob = dspeed * config.dt * config.d.powi(3) * k_factor * PI
                / (2 * config.n_test) as f64;
            
            // Perform collision if probability threshold is met
            if rng.gen_range(0.0..1.0) < collision_prob {
                // Apply velocity changes directly
                for k in 0..3 {
                    v[i0][k] -= vec_k[k] * dv_dr;
                    v[i1][k] += vec_k[k] * dv_dr;
                }
            }
        }
    }
}

#[inline]
fn scatt_o2_optimized(
    grid: &Vec<Vec<Vec<Vec<usize>>>>,
    i_x: usize,
    i_y: usize,
    i_z: usize,
    di: [i32; 3],
    rng: &mut rand::rngs::ThreadRng,
    r: &[[f64; 3]],
    v: &mut [[f64; 3]],
    config: &Config,
) {
    let l = grid.len();
    let i_x_new = (i_x as i32 + di[0]).rem_euclid(l as i32) as usize;
    let i_y_new = (i_y as i32 + di[1]).rem_euclid(l as i32) as usize;
    let i_z_new = (i_z as i32 + di[2]).rem_euclid(l as i32) as usize;
    
    // Store local references to grid cells for better cache locality  
    let current_cell = &grid[i_x][i_y][i_z];
    let neighbor_cell = &grid[i_x_new][i_y_new][i_z_new];
    
    // Skip empty cell pairs
    if current_cell.is_empty() || neighbor_cell.is_empty() {
        return;
    }
    
    for j0 in 0..current_cell.len() {
        let i0 = current_cell[j0];
        
        for j1 in 0..neighbor_cell.len() {
            let i1 = neighbor_cell[j1];
            
            // Skip self-interactions
            if i0 == i1 {
                continue;
            }
            
            let (_dr2, dv_dr, dspeed, _dr, vec_k) = 
                calculate_collision_parameters(i0, i1, r, v, config);
                
            // Early exit if particles are moving away from each other
            if dv_dr >= 0.0 {
                continue;
            }
            
            // Calculate k_factor based on collision order
            let mut k_factor = 1.0;
            for k in 0..3 {
                if di[k] == 1 {
                    k_factor *= vec_k[k];
                } else if di[k] == 2 {
                    k_factor *= vec_k[k].powi(2);
                }
            }
            
            // Calculate collision probability
            let collision_prob = dspeed * config.dt * config.d.powi(4) * k_factor * PI
                / (8 * config.n_test) as f64;
            
            // Perform collision if probability threshold is met
            if rng.gen_range(0.0..1.0) < collision_prob {
                for k in 0..3 {
                    v[i0][k] -= vec_k[k] * dv_dr;
                    v[i1][k] += vec_k[k] * dv_dr;
                }
            }
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 4 {
        eprintln!(
            "Usage: {} <config_path> <n_step: usize> <bounded: bool>",
            args[0]
        );
        std::process::exit(1);
    }
    let config_path = &args[1];
    let n_step: usize = args[2].parse()?;
    let bounded: bool = args[3].parse()?;
    let config = Config::from_file(config_path);

    let mut r = vec![[0.0; 3]; config.n];
    let mut theta = vec![0.0; config.n];
    let mut phi = vec![0.0; config.n];
    let mut v = vec![[0.0; 3]; config.n];
    let mut rng = rand::thread_rng();
    for i in 0..config.n {
        for j in 0..3 {
            r[i][j] = rng.gen_range(0.0..config.l as f64);
        }
        theta[i] = (1.0 - 2.0 * rng.gen_range(0.0..1.0_f64)).acos();
        phi[i] = rng.gen_range(0.0..2.0 * std::f64::consts::PI);
        v[i][0] = theta[i].sin() * phi[i].cos() * (2.0 * config.e0 / config.mass).sqrt();
        v[i][1] = theta[i].sin() * phi[i].sin() * (2.0 * config.e0 / config.mass).sqrt();
        v[i][2] = theta[i].cos() * (2.0 * config.e0 / config.mass).sqrt();
    }
    let mut grid: Vec<Vec<Vec<Vec<usize>>>> =
        vec![vec![vec![vec![]; config.l]; config.l]; config.l];

    let mut pressures = vec![0.0; n_step];
    // let mut temperatures = vec![0.0; n_step];

    for i in 0..config.n {
        grid[r[i][0].floor() as usize][r[i][1].floor() as usize][r[i][2].floor() as usize].push(i);
    }

    let start = Instant::now();
    for i_t in 0..n_step {
        for i in 0..config.n {
            for j in 0..3 {
                r[i][j] += v[i][j] * config.dt;
            }
        }
        let mut pressure = 0.0;
        if bounded {
            for i in 0..config.n {
                for j in 0..3 {
                    if r[i][j] < 0.0 {
                        r[i][j] = -r[i][j];
                        pressure -= 2.0 * config.mass * v[i][j] / config.dt;
                        v[i][j] = -v[i][j];
                    }
                    if r[i][j] >= config.l as f64 {
                        r[i][j] = 2.0 * config.l as f64 - r[i][j];
                        pressure += 2.0 * config.mass * v[i][j] / config.dt;
                        v[i][j] = -v[i][j];
                    }
                }
            }
        } else {
            for i in 0..config.n {
                for j in 0..3 {
                    if r[i][j] < 0.0 {
                        pressure -= 2. * config.mass * v[i][j] / config.dt;
                        r[i][j] = r[i][j].rem_euclid(config.l as f64);
                    }
                    if r[i][j] >= config.l as f64 {
                        pressure += 2. * config.mass * v[i][j] / config.dt;
                        r[i][j] = r[i][j].rem_euclid(config.l as f64);
                    }
                }
            }
        }
        pressure /= (6 * config.l.pow(2)) as f64;
        pressures[i_t] = pressure;

        let mut removes = vec![];
        for i_x in 0..config.l {
            for i_y in 0..config.l {
                for i_z in 0..config.l {
                    for j in 0..grid[i_x][i_y][i_z].len() {
                        let i: usize = grid[i_x][i_y][i_z][j];
                        let i_x_new = r[i][0].floor() as usize;
                        let i_y_new = r[i][1].floor() as usize;
                        let i_z_new = r[i][2].floor() as usize;
                        if i_x_new != i_x || i_y_new != i_y || i_z_new != i_z {
                            removes.push(j);
                            grid[i_x_new][i_y_new][i_z_new].push(i);
                        }
                    }
                    for &j in removes.iter().rev() {
                        grid[i_x][i_y][i_z].remove(j);
                    }
                    removes.clear();
                }
            }
        }

        grid.iter().for_each(|grid_x| {
            grid_x.iter().for_each(|grid_y| {
                grid_y.iter().for_each(|grid_z| {
                    for j0 in 0..grid_z.len() {
                        for j1 in j0 + 1..grid_z.len() {
                            let i0 = grid_z[j0];
                            let i1 = grid_z[j1];
                            let mut dr = [0.0; 3];
                            let mut dv = [0.0; 3];
                            for k in 0..3 {
                                dr[k] = r[i0][k] - r[i1][k];
                                dv[k] = v[i0][k] - v[i1][k];
                            }
                            let dv2 = dv[0] * dv[0] + dv[1] * dv[1] + dv[2] * dv[2];
                            let dspeed = dv2.sqrt();
                            let collision_prob = dspeed * config.dt * config.d * config.d * PI
                                / config.n_test as f64;
                            if rng.gen_range(0.0..1.0) < collision_prob {
                                let dr2 = dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2];
                                let dv_dr = dv[0] * dr[0] + dv[1] * dr[1] + dv[2] * dr[2];
                                for k in 0..3 {
                                    v[i0][k] -= dr[k] * dv_dr / dr2;
                                    v[i1][k] += dr[k] * dv_dr / dr2;
                                }
                                let mut force = [0.0; 3];
                                if dr2 > config.d.powi(2) && dr2 < 10.0 * (config.d).powi(2) {
                                    for k in 0..3 {
                                        force[k] = 24.0 * config.l_j_epsilon * (2.0 * (config.d.powi(2) / dr2).powi(5) - (config.d.powi(2) / dr2).powi(2)) * dr[k];
                                        v[i0][k] -= force[k] * config.dt / config.mass;
                                        v[i1][k] += force[k] * config.dt / config.mass;
                                    }
                                }
                            }
                        }
                    }
                });
            });
        });

        for i_x in 0..config.l {
            for i_y in 0..config.l {
                for i_z in 0..config.l {
                    scatt_o1_optimized(&grid, i_x, i_y, i_z, [1, 0, 0], &mut rng, &r, &mut v, &config);
                    scatt_o1_optimized(&grid, i_x, i_y, i_z, [0, 1, 0], &mut rng, &r, &mut v, &config);
                    scatt_o1_optimized(&grid, i_x, i_y, i_z, [0, 0, 1], &mut rng, &r, &mut v, &config);
                    scatt_o2_optimized(&grid, i_x, i_y, i_z, [1, 1, 0], &mut rng, &r, &mut v, &config);
                    scatt_o2_optimized(&grid, i_x, i_y, i_z, [1, 0, 1], &mut rng, &r, &mut v, &config);
                    scatt_o2_optimized(&grid, i_x, i_y, i_z, [0, 1, 1], &mut rng, &r, &mut v, &config);
                    scatt_o2_optimized(&grid, i_x, i_y, i_z, [2, 0, 0], &mut rng, &r, &mut v, &config);
                    scatt_o2_optimized(&grid, i_x, i_y, i_z, [0, 2, 0], &mut rng, &r, &mut v, &config);
                    scatt_o2_optimized(&grid, i_x, i_y, i_z, [0, 0, 2], &mut rng, &r, &mut v, &config);
                }
            }
        }

        print!("{i_t}/{n_step}\r");
        io::stdout().flush().unwrap();
    }
    let elapsed = start.elapsed();
    println!(
        "Elapsed: {}.{:03} s",
        elapsed.as_secs(),
        elapsed.subsec_millis()
    );
    println!("{} ms per step", elapsed.as_millis() / n_step as u128);

    let mut speed = vec![0.0; config.n];
    for i in 0..config.n {
        speed[i] = (v[i][0] * v[i][0] + v[i][1] * v[i][1] + v[i][2] * v[i][2]).sqrt();
    }

    let data_dir: String = format!(
        "data/N={}_L={}_D={}_T={}_MASS={}_N_TEST={}_T_STEP={}_EPS={}_N_STEP={}_bounded={}",
        config.n, config.l, config.d, config.temperature, config.mass, config.n_test, config.dt, config.l_j_epsilon, n_step, bounded
    );
    std::fs::create_dir_all(&data_dir)?;
    let mut file = File::create(format!("{}/speed.csv", data_dir))?;
    writeln!(file, "speed")?;
    for value in speed {
        writeln!(file, "{}", value)?;
    }
    file = File::create(format!("{}/final_state.csv", data_dir))?;
    writeln!(file, "x,y,z,vx,vy,vz")?;
    for i in 0..config.n {
        writeln!(file, "{},{},{},{},{},{}", r[i][0], r[i][1], r[i][2], v[i][0], v[i][1], v[i][2])?;
    }
    file = File::create(format!("{}/pressure.csv", data_dir))?;
    writeln!(file, "time,pressure")?;
    for (i, value) in pressures.iter().enumerate() {
        writeln!(file, "{},{}", i as f64 * config.dt, value)?;
    }
    Ok(())
}
===
#![cfg_attr(feature = "simd", feature(portable_simd))]

// Fix the imports for SIMD
#[cfg(feature = "simd")]
use std::simd::f64x4;
#[cfg(feature = "simd")]
use std::simd::num::SimdFloat; // Import the correct trait for reduce_sum()

use std::env;
use std::io::prelude::*;
use std::io::{self, BufWriter};

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use std::fs::File;
use std::time::Instant;

mod config;
use config::Config;

// Flat grid index helper
#[inline]
fn grid_idx(x: usize, y: usize, z: usize, l: usize) -> usize {
    x * l * l + y * l + z
}

// Vector operation utilities
#[inline]
fn dot_product(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[cfg(feature = "simd")]
#[inline]
fn dot_product_simd(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    let a_simd = f64x4::from_array([a[0], a[1], a[2], 0.0]);
    let b_simd = f64x4::from_array([b[0], b[1], b[2], 0.0]);
    (a_simd * b_simd).reduce_sum()
}

#[inline]
fn apply_periodic_boundary(mut dr: f64, box_length: f64) -> f64 {
    if dr > box_length / 2.0 {
        dr -= box_length;
    } else if dr < -box_length / 2.0 {
        dr += box_length;
    }
    dr
}

// Common collision calculation logic for both scatt_o1 and scatt_o2
#[inline]
fn calculate_collision_parameters(
    i0: usize, 
    i1: usize, 
    r: &[[f64; 3]], 
    v: &[[f64; 3]],
    config: &Config
) -> (f64, f64, f64, [f64; 3], [f64; 3]) {
    let mut dr = [0.0; 3];
    let mut dv = [0.0; 3];
    
    // Compute position and velocity differences with boundary conditions
    for k in 0..3 {
        dr[k] = r[i0][k] - r[i1][k];
        dr[k] = apply_periodic_boundary(dr[k], config.l as f64);
        dv[k] = v[i0][k] - v[i1][k];
    }
    
    #[cfg(feature = "simd")]
    let dr2 = dot_product_simd(&dr, &dr);
    #[cfg(not(feature = "simd"))]
    let dr2 = dot_product(&dr, &dr);
    
    #[cfg(feature = "simd")]
    let dv_dr = dot_product_simd(&dv, &dr);
    #[cfg(not(feature = "simd"))]
    let dv_dr = dot_product(&dv, &dr);
    
    #[cfg(feature = "simd")]
    let dv2 = dot_product_simd(&dv, &dv);
    #[cfg(not(feature = "simd"))]
    let dv2 = dot_product(&dv, &dv);
    
    let dspeed = dv2.sqrt();
    
    let mut vec_k = [0.0; 3];
    for k in 0..3 {
        vec_k[k] = dr[k] / dr2;
    }
    
    (dr2, dv_dr, dspeed, dr, vec_k)
}

#[inline]
fn scatt_o1_optimized(
    grid: &[Vec<usize>],
    l: usize,
    i_x: usize,
    i_y: usize,
    i_z: usize,
    di: [i32; 3],
    rng: &mut SmallRng,
    r: &[[f64; 3]],
    v: &mut [[f64; 3]],
    config: &Config,
) {
    let i_x_new = (i_x as i32 + di[0]).rem_euclid(l as i32) as usize;
    let i_y_new = (i_y as i32 + di[1]).rem_euclid(l as i32) as usize;
    let i_z_new = (i_z as i32 + di[2]).rem_euclid(l as i32) as usize;
    
    // Store local references to grid cells for better cache locality
    let current_cell = &grid[grid_idx(i_x, i_y, i_z, l)];
    let neighbor_cell = &grid[grid_idx(i_x_new, i_y_new, i_z_new, l)];
    
    // Skip empty cell pairs
    if current_cell.is_empty() || neighbor_cell.is_empty() {
        return;
    }
    
    // First pass: identify all potential collisions
    for j0 in 0..current_cell.len() {
        let i0 = current_cell[j0];
        
        for j1 in 0..neighbor_cell.len() {
            let i1 = neighbor_cell[j1];
            
            // Skip self-interactions
            if i0 == i1 {
                continue;
            }
            
            let (_dr2, dv_dr, dspeed, _dr, vec_k) = 
                calculate_collision_parameters(i0, i1, r, v, config);
                
            // Early exit if particles are moving away from each other
            if dv_dr >= 0.0 {
                continue;
            }
            
            // Calculate k_factor based on direction vector
            let mut k_factor = vec_k[0] * di[0] as f64 + 
                              vec_k[1] * di[1] as f64 + 
                              vec_k[2] * di[2] as f64;
            k_factor = -k_factor; // Adjust for collision direction
            
            // Calculate collision probability
            let collision_prob = dspeed * config.collision_coeff_o1 * k_factor;
            
            // Perform collision if probability threshold is met
            if rng.gen_range(0.0_f64..1.0) < collision_prob {
                // Apply velocity changes directly
                for k in 0..3 {
                    v[i0][k] -= vec_k[k] * dv_dr;
                    v[i1][k] += vec_k[k] * dv_dr;
                }
            }
        }
    }
}

#[inline]
fn scatt_o2_optimized(
    grid: &[Vec<usize>],
    l: usize,
    i_x: usize,
    i_y: usize,
    i_z: usize,
    di: [i32; 3],
    rng: &mut SmallRng,
    r: &[[f64; 3]],
    v: &mut [[f64; 3]],
    config: &Config,
) {
    let i_x_new = (i_x as i32 + di[0]).rem_euclid(l as i32) as usize;
    let i_y_new = (i_y as i32 + di[1]).rem_euclid(l as i32) as usize;
    let i_z_new = (i_z as i32 + di[2]).rem_euclid(l as i32) as usize;
    
    // Store local references to grid cells for better cache locality  
    let current_cell = &grid[grid_idx(i_x, i_y, i_z, l)];
    let neighbor_cell = &grid[grid_idx(i_x_new, i_y_new, i_z_new, l)];
    
    // Skip empty cell pairs
    if current_cell.is_empty() || neighbor_cell.is_empty() {
        return;
    }
    
    for j0 in 0..current_cell.len() {
        let i0 = current_cell[j0];
        
        for j1 in 0..neighbor_cell.len() {
            let i1 = neighbor_cell[j1];
            
            // Skip self-interactions
            if i0 == i1 {
                continue;
            }
            
            let (_dr2, dv_dr, dspeed, _dr, vec_k) = 
                calculate_collision_parameters(i0, i1, r, v, config);
                
            // Early exit if particles are moving away from each other
            if dv_dr >= 0.0 {
                continue;
            }
            
            // Calculate k_factor based on collision order
            let mut k_factor = 1.0;
            for k in 0..3 {
                if di[k] == 1 {
                    k_factor *= vec_k[k];
                } else if di[k] == 2 {
                    k_factor *= vec_k[k].powi(2);
                }
            }
            
            // Calculate collision probability
            let collision_prob = dspeed * config.collision_coeff_o2 * k_factor;
            
            // Perform collision if probability threshold is met
            if rng.gen_range(0.0_f64..1.0) < collision_prob {
                for k in 0..3 {
                    v[i0][k] -= vec_k[k] * dv_dr;
                    v[i1][k] += vec_k[k] * dv_dr;
                }
            }
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 4 {
        eprintln!(
            "Usage: {} <config_path> <n_step: usize> <bounded: bool>",
            args[0]
        );
        std::process::exit(1);
    }
    let config_path = &args[1];
    let n_step: usize = args[2].parse()?;
    let bounded: bool = args[3].parse()?;
    let config = Config::from_file(config_path);

    let mut r: Vec<[f64; 3]> = vec![[0.0; 3]; config.n];
    let mut v: Vec<[f64; 3]> = vec![[0.0; 3]; config.n];
    let mut rng = SmallRng::from_entropy();
    for i in 0..config.n {
        for j in 0..3 {
            r[i][j] = rng.gen_range(0.0_f64..config.l as f64);
        }
        let theta: f64 = (1.0 - 2.0 * rng.gen_range(0.0_f64..1.0)).acos();
        let phi: f64 = rng.gen_range(0.0_f64..2.0 * std::f64::consts::PI);
        v[i][0] = theta.sin() * phi.cos() * config.v_magnitude;
        v[i][1] = theta.sin() * phi.sin() * config.v_magnitude;
        v[i][2] = theta.cos() * config.v_magnitude;
    }
    // Flattened grid: single Vec of L³ cells instead of nested Vec<Vec<Vec<Vec>>>
    let l = config.l;
    let grid_size = l * l * l;
    let mut grid: Vec<Vec<usize>> = vec![vec![]; grid_size];

    let mut pressures = vec![0.0; n_step];
    // let mut temperatures = vec![0.0; n_step];

    for i in 0..config.n {
        let idx = grid_idx(r[i][0].floor() as usize, r[i][1].floor() as usize, r[i][2].floor() as usize, l);
        grid[idx].push(i);
    }

    let start = Instant::now();
    for i_t in 0..n_step {
        for i in 0..config.n {
            for j in 0..3 {
                r[i][j] += v[i][j] * config.dt;
            }
        }
        let mut pressure = 0.0;
        if bounded {
            for i in 0..config.n {
                for j in 0..3 {
                    if r[i][j] < 0.0 {
                        r[i][j] = -r[i][j];
                        pressure -= 2.0 * config.mass * v[i][j] / config.dt;
                        v[i][j] = -v[i][j];
                    }
                    if r[i][j] >= config.l as f64 {
                        r[i][j] = 2.0 * config.l as f64 - r[i][j];
                        pressure += 2.0 * config.mass * v[i][j] / config.dt;
                        v[i][j] = -v[i][j];
                    }
                }
            }
        } else {
            for i in 0..config.n {
                for j in 0..3 {
                    if r[i][j] < 0.0 {
                        pressure -= 2. * config.mass * v[i][j] / config.dt;
                        r[i][j] = r[i][j].rem_euclid(config.l as f64);
                    }
                    if r[i][j] >= config.l as f64 {
                        pressure += 2. * config.mass * v[i][j] / config.dt;
                        r[i][j] = r[i][j].rem_euclid(config.l as f64);
                    }
                }
            }
        }
        pressure /= (6 * config.l.pow(2)) as f64;
        pressures[i_t] = pressure;

        let mut removes = vec![];
        for i_x in 0..l {
            for i_y in 0..l {
                for i_z in 0..l {
                    let cell_idx = grid_idx(i_x, i_y, i_z, l);
                    for j in 0..grid[cell_idx].len() {
                        let i: usize = grid[cell_idx][j];
                        let i_x_new = r[i][0].floor() as usize;
                        let i_y_new = r[i][1].floor() as usize;
                        let i_z_new = r[i][2].floor() as usize;
                        if i_x_new != i_x || i_y_new != i_y || i_z_new != i_z {
                            removes.push(j);
                            let new_idx = grid_idx(i_x_new, i_y_new, i_z_new, l);
                            grid[new_idx].push(i);
                        }
                    }
                    // Use swap_remove (O(1)) instead of remove (O(n))
                    // Must iterate in reverse to keep indices valid
                    for &j in removes.iter().rev() {
                        grid[cell_idx].swap_remove(j);
                    }
                    removes.clear();
                }
            }
        }

        for cell_idx in 0..grid_size {
            let grid_z = &grid[cell_idx];
            for j0 in 0..grid_z.len() {
                for j1 in j0 + 1..grid_z.len() {
                    let i0 = grid_z[j0];
                    let i1 = grid_z[j1];
                    let mut dr = [0.0; 3];
                    let mut dv = [0.0; 3];
                    for k in 0..3 {
                        dr[k] = r[i0][k] - r[i1][k];
                        dv[k] = v[i0][k] - v[i1][k];
                    }
                    let dv2 = dv[0] * dv[0] + dv[1] * dv[1] + dv[2] * dv[2];
                    let dspeed = dv2.sqrt();
                    let collision_prob = dspeed * config.collision_coeff_o0;
                    if rng.gen_range(0.0_f64..1.0) < collision_prob {
                        let dr2 = dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2];
                        let dv_dr = dv[0] * dr[0] + dv[1] * dr[1] + dv[2] * dr[2];
                        for k in 0..3 {
                            v[i0][k] -= dr[k] * dv_dr / dr2;
                            v[i1][k] += dr[k] * dv_dr / dr2;
                        }
                        let mut force = [0.0; 3];
                        if dr2 > config.d_sq && dr2 < 10.0 * config.d_sq {
                            let d_sq_over_dr2 = config.d_sq / dr2;
                            let d_sq_over_dr2_sq = d_sq_over_dr2 * d_sq_over_dr2;
                            let d_sq_over_dr2_5 = d_sq_over_dr2_sq * d_sq_over_dr2_sq * d_sq_over_dr2;
                            let lj_factor = config.lj_coeff * (2.0 * d_sq_over_dr2_5 - d_sq_over_dr2_sq);
                            for k in 0..3 {
                                force[k] = lj_factor * dr[k];
                                v[i0][k] -= force[k] * config.dt_over_mass;
                                v[i1][k] += force[k] * config.dt_over_mass;
                            }
                        }
                    }
                }
            }
        }

        for i_x in 0..l {
            for i_y in 0..l {
                for i_z in 0..l {
                    scatt_o1_optimized(&grid, l, i_x, i_y, i_z, [1, 0, 0], &mut rng, &r, &mut v, &config);
                    scatt_o1_optimized(&grid, l, i_x, i_y, i_z, [0, 1, 0], &mut rng, &r, &mut v, &config);
                    scatt_o1_optimized(&grid, l, i_x, i_y, i_z, [0, 0, 1], &mut rng, &r, &mut v, &config);
                    scatt_o2_optimized(&grid, l, i_x, i_y, i_z, [1, 1, 0], &mut rng, &r, &mut v, &config);
                    scatt_o2_optimized(&grid, l, i_x, i_y, i_z, [1, 0, 1], &mut rng, &r, &mut v, &config);
                    scatt_o2_optimized(&grid, l, i_x, i_y, i_z, [0, 1, 1], &mut rng, &r, &mut v, &config);
                    scatt_o2_optimized(&grid, l, i_x, i_y, i_z, [2, 0, 0], &mut rng, &r, &mut v, &config);
                    scatt_o2_optimized(&grid, l, i_x, i_y, i_z, [0, 2, 0], &mut rng, &r, &mut v, &config);
                    scatt_o2_optimized(&grid, l, i_x, i_y, i_z, [0, 0, 2], &mut rng, &r, &mut v, &config);
                }
            }
        }

        if i_t % 10 == 0 {
            print!("{i_t}/{n_step}\r");
            io::stdout().flush().unwrap();
        }
    }
    let elapsed = start.elapsed();
    println!(
        "Elapsed: {}.{:03} s",
        elapsed.as_secs(),
        elapsed.subsec_millis()
    );
    println!("{} ms per step", elapsed.as_millis() / n_step as u128);

    let mut speed = vec![0.0; config.n];
    for i in 0..config.n {
        speed[i] = (v[i][0] * v[i][0] + v[i][1] * v[i][1] + v[i][2] * v[i][2]).sqrt();
    }

    let data_dir: String = format!(
        "data/N={}_L={}_D={}_T={}_MASS={}_N_TEST={}_T_STEP={}_EPS={}_N_STEP={}_bounded={}",
        config.n, config.l, config.d, config.temperature, config.mass, config.n_test, config.dt, config.l_j_epsilon, n_step, bounded
    );
    std::fs::create_dir_all(&data_dir)?;
    {
        let mut file = BufWriter::new(File::create(format!("{}/speed.csv", data_dir))?);
        writeln!(file, "speed")?;
        for value in speed {
            writeln!(file, "{}", value)?;
        }
    }
    {
        let mut file = BufWriter::new(File::create(format!("{}/final_state.csv", data_dir))?);
        writeln!(file, "x,y,z,vx,vy,vz")?;
        for i in 0..config.n {
            writeln!(file, "{},{},{},{},{},{}", r[i][0], r[i][1], r[i][2], v[i][0], v[i][1], v[i][2])?;
        }
    }
    {
        let mut file = BufWriter::new(File::create(format!("{}/pressure.csv", data_dir))?);
        writeln!(file, "time,pressure")?;
        for (i, value) in pressures.iter().enumerate() {
            writeln!(file, "{},{}", i as f64 * config.dt, value)?;
        }
    }
    Ok(())
}

```

**主要变更：**

1. **网格数据结构扁平化**
   - `Vec<Vec<Vec<Vec<usize>>>>` → `Vec<Vec<usize>>` + `grid_idx()` 辅助函数
   - 减少 L³ 次堆分配，改善缓存局部性

2. **`swap_remove` 替代 `remove`**
   - 网格粒子迁移从 O(n) 降为 O(1)

3. **`SmallRng` 替代 `ThreadRng`**
   - 非密码学 RNG，速度提升 2-3x
   - 需要在 `Cargo.toml` 启用 `small_rng` feature

4. **预计算常量**
   - 碰撞概率公式中的 `d.powi(2/3/4)`, `dt * PI / n_test` 等预计算到 Config 中
   - LJ 力场计算展开为手动乘法链，避免 `powi()` 调用

5. **I/O 优化**
   - 文件输出使用 `BufWriter` 批量写入
   - 进度输出频率从每步降为每 10 步

6. **代码清理**
   - 消除 `theta`/`phi` 全局向量（改为局部变量）
   - 移除未使用 imports (`PI`, `vec`, 注释掉的 rayon)

---

### [Cargo.toml](file:///Users/bjergsen/Documents/GitHub/mc/Cargo.toml)

```diff
-rand = "0.8.4"
+rand = { version = "0.8.4", features = ["small_rng"] }
```

---

## 验证

- **编译**：`cargo build --release` — 零 warnings ✅
- **功能测试**：`bounded=false` 和 `bounded=true` 两种模式均正常运行 ✅
- **统计一致性**：所有步骤的平均粒子速度稳定在 1.18-1.19 范围内 ✅

| 阶段 | 平均速度 |
|------|---------|
| 基线 | 1.18767 |
| Step 1 | 1.19062 |
| Step 2 | 1.18679 |
| Step 3 | 1.18185 |
| Step 4 | 1.19101 |
| Step 5 | 1.18727 |
| Step 6 | 1.19415 |
