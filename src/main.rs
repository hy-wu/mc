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
