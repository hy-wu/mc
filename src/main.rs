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
use rayon::prelude::*;
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
    v_ptr: *const [f64; 3],
    config: &Config,
) -> (f64, f64, f64, [f64; 3], [f64; 3]) {
    let mut dr = [0.0; 3];
    let mut dv = [0.0; 3];

    // Compute position and velocity differences with boundary conditions
    unsafe {
        let vi0 = &*v_ptr.add(i0);
        let vi1 = &*v_ptr.add(i1);
        for k in 0..3 {
            dr[k] = r[i0][k] - r[i1][k];
            dr[k] = apply_periodic_boundary(dr[k], config.l as f64);
            dv[k] = vi0[k] - vi1[k];
        }
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

// Wrapper to allow sending raw pointer across threads
// SAFETY: We guarantee non-overlapping access via graph coloring / disjoint cell particles
#[derive(Copy, Clone)]
struct SendPtr<T>(usize, std::marker::PhantomData<*mut T>);
unsafe impl<T> Send for SendPtr<T> {}
unsafe impl<T> Sync for SendPtr<T> {}

impl<T> SendPtr<T> {
    fn new(ptr: *mut T) -> Self {
        SendPtr(ptr as usize, std::marker::PhantomData)
    }
    #[inline]
    unsafe fn get(&self, i: usize) -> &T {
        &*((self.0 as *const T).add(i))
    }
    #[inline]
    unsafe fn get_mut(&self, i: usize) -> &mut T {
        &mut *((self.0 as *mut T).add(i))
    }
    #[inline]
    fn as_const_ptr(&self) -> *const T {
        self.0 as *const T
    }
}

/// Process cross-cell collisions for a given direction with graph coloring.
/// `color_fn` determines the coloring (number of passes) to avoid data races.
fn process_cross_cell_collisions(
    grid: &[Vec<usize>],
    l: usize,
    di: [i32; 3],
    r: &[[f64; 3]],
    v_ptr: SendPtr<[f64; 3]>,
    config: &Config,
    is_o1: bool,
    n_colors: usize,
    color_key_fn: fn(usize, usize, usize) -> usize,
) {
    for color in 0..n_colors {
        // Collect cells matching this color
        let cells: Vec<(usize, usize, usize)> = (0..l)
            .flat_map(|x| (0..l).flat_map(move |y| (0..l).map(move |z| (x, y, z))))
            .filter(|&(x, y, z)| color_key_fn(x, y, z) % n_colors == color)
            .collect();

        let vp = v_ptr; // Copy the SendPtr so closure captures a Sync type
        cells.par_iter().for_each_init(
            || SmallRng::from_entropy(),
            move |rng, &(i_x, i_y, i_z)| {
                let i_x_new = (i_x as i32 + di[0]).rem_euclid(l as i32) as usize;
                let i_y_new = (i_y as i32 + di[1]).rem_euclid(l as i32) as usize;
                let i_z_new = (i_z as i32 + di[2]).rem_euclid(l as i32) as usize;

                let current_cell = &grid[grid_idx(i_x, i_y, i_z, l)];
                let neighbor_cell = &grid[grid_idx(i_x_new, i_y_new, i_z_new, l)];

                if current_cell.is_empty() || neighbor_cell.is_empty() {
                    return;
                }

                for j0 in 0..current_cell.len() {
                    let i0 = current_cell[j0];
                    for j1 in 0..neighbor_cell.len() {
                        let i1 = neighbor_cell[j1];
                        if i0 == i1 {
                            continue;
                        }

                        let (_dr2, dv_dr, dspeed, _dr, vec_k) =
                            calculate_collision_parameters(i0, i1, r, vp.as_const_ptr(), config);

                        if dv_dr >= 0.0 {
                            continue;
                        }

                        let collision_prob;

                        if is_o1 {
                            let mut kf = vec_k[0] * di[0] as f64
                                + vec_k[1] * di[1] as f64
                                + vec_k[2] * di[2] as f64;
                            kf = -kf;
                            collision_prob = dspeed * config.collision_coeff_o1 * kf;
                        } else {
                            let mut kf = 1.0;
                            for k in 0..3 {
                                if di[k] == 1 {
                                    kf *= vec_k[k];
                                } else if di[k] == 2 {
                                    kf *= vec_k[k].powi(2);
                                }
                            }
                            collision_prob = dspeed * config.collision_coeff_o2 * kf;
                        }

                        if rng.gen_range(0.0_f64..1.0) < collision_prob {
                            unsafe {
                                let vi0 = vp.get_mut(i0);
                                let vi1 = vp.get_mut(i1);
                                for k in 0..3 {
                                    vi0[k] -= vec_k[k] * dv_dr;
                                    vi1[k] += vec_k[k] * dv_dr;
                                }
                            }
                        }
                    }
                }
            },
        );
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
    // Flattened grid: single Vec of L³ cells
    let l = config.l;
    let grid_size = l * l * l;
    let mut grid: Vec<Vec<usize>> = vec![vec![]; grid_size];

    let mut pressures = vec![0.0; n_step];

    for i in 0..config.n {
        let idx = grid_idx(
            r[i][0].floor() as usize,
            r[i][1].floor() as usize,
            r[i][2].floor() as usize,
            l,
        );
        grid[idx].push(i);
    }

    let l_f64 = config.l as f64;

    let start = Instant::now();
    for i_t in 0..n_step {
        // === 1. Position update (parallel) ===
        r.par_iter_mut()
            .zip(v.par_iter())
            .for_each(|(ri, vi)| {
                for j in 0..3 {
                    ri[j] += vi[j] * config.dt;
                }
            });

        // === 2. Boundary conditions + pressure (parallel reduction) ===
        let pressure: f64 = if bounded {
            r.par_iter_mut()
                .zip(v.par_iter_mut())
                .map(|(ri, vi)| {
                    let mut p = 0.0f64;
                    for j in 0..3 {
                        if ri[j] < 0.0 {
                            ri[j] = -ri[j];
                            p -= 2.0 * config.mass * vi[j] / config.dt;
                            vi[j] = -vi[j];
                        }
                        if ri[j] >= l_f64 {
                            ri[j] = 2.0 * l_f64 - ri[j];
                            p += 2.0 * config.mass * vi[j] / config.dt;
                            vi[j] = -vi[j];
                        }
                    }
                    p
                })
                .sum()
        } else {
            r.par_iter_mut()
                .zip(v.par_iter_mut())
                .map(|(ri, vi)| {
                    let mut p = 0.0f64;
                    for j in 0..3 {
                        if ri[j] < 0.0 {
                            p -= 2.0 * config.mass * vi[j] / config.dt;
                            ri[j] = ri[j].rem_euclid(l_f64);
                        }
                        if ri[j] >= l_f64 {
                            p += 2.0 * config.mass * vi[j] / config.dt;
                            ri[j] = ri[j].rem_euclid(l_f64);
                        }
                    }
                    p
                })
                .sum()
        };
        pressures[i_t] = pressure / (6 * config.l.pow(2)) as f64;

        // === 3. Grid rebuild (sequential O(N) — simpler than incremental) ===
        for cell in grid.iter_mut() {
            cell.clear();
        }
        for i in 0..config.n {
            let idx = grid_idx(
                r[i][0].floor() as usize,
                r[i][1].floor() as usize,
                r[i][2].floor() as usize,
                l,
            );
            grid[idx].push(i);
        }

        // === 4. Same-cell collisions (parallel over cells) ===
        // SAFETY: Each particle belongs to exactly one cell, so parallel cell processing
        // accesses disjoint particle indices in `v`. No data race occurs.
        let v_ptr = SendPtr::new(v.as_mut_ptr());
        let r_ref = r.as_slice();
        grid.par_iter().for_each_init(
            || SmallRng::from_entropy(),
            move |rng, cell| {
                for j0 in 0..cell.len() {
                    for j1 in j0 + 1..cell.len() {
                        let i0 = cell[j0];
                        let i1 = cell[j1];
                        let mut dr = [0.0; 3];
                        let mut dv = [0.0; 3];
                        unsafe {
                            let vi0 = v_ptr.get(i0);
                            let vi1 = v_ptr.get(i1);
                            for k in 0..3 {
                                dr[k] = r_ref[i0][k] - r_ref[i1][k];
                                dv[k] = vi0[k] - vi1[k];
                            }
                        }
                        let dv2 = dv[0] * dv[0] + dv[1] * dv[1] + dv[2] * dv[2];
                        let dspeed = dv2.sqrt();
                        let collision_prob = dspeed * config.collision_coeff_o0;
                        if rng.gen_range(0.0_f64..1.0) < collision_prob {
                            let dr2 = dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2];
                            let dv_dr = dv[0] * dr[0] + dv[1] * dr[1] + dv[2] * dr[2];
                            unsafe {
                                let vi0 = v_ptr.get_mut(i0);
                                let vi1 = v_ptr.get_mut(i1);
                                for k in 0..3 {
                                    vi0[k] -= dr[k] * dv_dr / dr2;
                                    vi1[k] += dr[k] * dv_dr / dr2;
                                }
                                let mut force = [0.0; 3];
                                if dr2 > config.d_sq && dr2 < 10.0 * config.d_sq {
                                    let d_sq_over_dr2 = config.d_sq / dr2;
                                    let d_sq_over_dr2_sq = d_sq_over_dr2 * d_sq_over_dr2;
                                    let d_sq_over_dr2_5 =
                                        d_sq_over_dr2_sq * d_sq_over_dr2_sq * d_sq_over_dr2;
                                    let lj_factor = config.lj_coeff
                                        * (2.0 * d_sq_over_dr2_5 - d_sq_over_dr2_sq);
                                    for k in 0..3 {
                                        force[k] = lj_factor * dr[k];
                                        vi0[k] -= force[k] * config.dt_over_mass;
                                        vi1[k] += force[k] * config.dt_over_mass;
                                    }
                                }
                            }
                        }
                    }
                }
            },
        );

        // === 5. Cross-cell collisions (parallel with graph coloring) ===
        // For each direction, we color cells so that no two cells processed in
        // parallel share any particle indices (current cell ∪ neighbor cell).
        let v_ptr = SendPtr::new(v.as_mut_ptr());

        // --- Order 1 directions: offset 1 along one axis → 2-coloring on that axis ---
        // [1,0,0]: color by x%2
        process_cross_cell_collisions(
            &grid, l, [1, 0, 0], &r, v_ptr, &config, true, 2,
            |x, _y, _z| x,
        );
        // [0,1,0]: color by y%2
        process_cross_cell_collisions(
            &grid, l, [0, 1, 0], &r, v_ptr, &config, true, 2,
            |_x, y, _z| y,
        );
        // [0,0,1]: color by z%2
        process_cross_cell_collisions(
            &grid, l, [0, 0, 1], &r, v_ptr, &config, true, 2,
            |_x, _y, z| z,
        );

        // --- Order 2 directions: diagonal offset → 2-coloring on axis sum parity ---
        // [1,1,0]: color by (x+y)%2
        process_cross_cell_collisions(
            &grid, l, [1, 1, 0], &r, v_ptr, &config, false, 2,
            |x, y, _z| x + y,
        );
        // [1,0,1]: color by (x+z)%2
        process_cross_cell_collisions(
            &grid, l, [1, 0, 1], &r, v_ptr, &config, false, 2,
            |x, _y, z| x + z,
        );
        // [0,1,1]: color by (y+z)%2
        process_cross_cell_collisions(
            &grid, l, [0, 1, 1], &r, v_ptr, &config, false, 2,
            |_x, y, z| y + z,
        );

        // --- Order 2 directions: offset 2 along one axis → 3-coloring ---
        // [2,0,0]: color by x%3
        process_cross_cell_collisions(
            &grid, l, [2, 0, 0], &r, v_ptr, &config, false, 3,
            |x, _y, _z| x,
        );
        // [0,2,0]: color by y%3
        process_cross_cell_collisions(
            &grid, l, [0, 2, 0], &r, v_ptr, &config, false, 3,
            |_x, y, _z| y,
        );
        // [0,0,2]: color by z%3
        process_cross_cell_collisions(
            &grid, l, [0, 0, 2], &r, v_ptr, &config, false, 3,
            |_x, _y, z| z,
        );

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
        config.n,
        config.l,
        config.d,
        config.temperature,
        config.mass,
        config.n_test,
        config.dt,
        config.l_j_epsilon,
        n_step,
        bounded
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
            writeln!(
                file,
                "{},{},{},{},{},{}",
                r[i][0], r[i][1], r[i][2], v[i][0], v[i][1], v[i][2]
            )?;
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
