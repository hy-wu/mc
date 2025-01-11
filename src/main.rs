use std::env;
use std::f64::consts::PI;
use std::io::prelude::*;
use std::{io, vec};

use rand::Rng;
use std::fs::File;
use std::time::Instant;

mod config;
use config::Config;

fn scatt_o1(
    grid: &mut Vec<Vec<Vec<Vec<usize>>>>,
    i_x: usize,
    i_y: usize,
    i_z: usize,
    di: [i32; 3],
    rng: &mut rand::rngs::ThreadRng,
    r: &Vec<[f64; 3]>,
    v: &mut Vec<[f64; 3]>,
    config: &Config,
) {
    let l = grid.len();
    let i_x_new = (i_x as i32 + di[0]).rem_euclid(l as i32) as usize;
    let i_y_new = (i_y as i32 + di[1]).rem_euclid(l as i32) as usize;
    let i_z_new = (i_z as i32 + di[2]).rem_euclid(l as i32) as usize;
    for j0 in 0..grid[i_x][i_y][i_z].len() {
        for j1 in 0..grid[i_x_new][i_y_new][i_z_new].len() {
            let i0 = grid[i_x][i_y][i_z][j0];
            let i1 = grid[i_x_new][i_y_new][i_z_new][j1];
            let mut dr = [0.0; 3];
            let mut dv = [0.0; 3];
            for k in 0..3 {
                dr[k] = r[i0][k] - r[i1][k];
                if dr[k] > config.L as f64 / 2.0 {
                    dr[k] -= config.L as f64;
                } else if dr[k] < -(config.L as f64) / 2.0 {
                    dr[k] += config.L as f64;
                }
                dv[k] = v[i0][k] - v[i1][k];
            }
            let dr2 = dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2];
            let dv2 = dv[0] * dv[0] + dv[1] * dv[1] + dv[2] * dv[2];
            let dspeed = dv2.sqrt();
            // \vec{k} = \Delta\vec{v}_{rel} / |\Deltad\vec{v}_{rel}|
            //         = (\Delta\vec{v}_2 - \Delta\vec{v}_1) / |\Delta\vec{v}_2 - \Delta\vec{v}_1|
            let mut vec_k: [f64; 3] = [0.0; 3];
            for k in 0..3 {
                vec_k[k] = dr[k] / dr2
            }
            let mut k_factor =
                vec_k[0] * di[0] as f64 + vec_k[1] * di[1] as f64 + vec_k[2] * di[2] as f64;
            if k_factor >= 0.0 {
                continue;
            } else {
                k_factor = -k_factor;
            }
            let collision_prob = dspeed * config.T_STEP * config.D.powi(3) * k_factor * PI
                / (2 * config.N_TEST) as f64;
            if rng.gen_range(0.0..1.0) < collision_prob {
                let dv_dr = dv[0] * dr[0] + dv[1] * dr[1] + dv[2] * dr[2];
                for k in 0..3 {
                    v[i0][k] -= vec_k[k] * dv_dr;
                    v[i1][k] += vec_k[k] * dv_dr;
                }
            }
        }
    }
}

fn scatt_o2(
    grid: &mut Vec<Vec<Vec<Vec<usize>>>>,
    i_x: usize,
    i_y: usize,
    i_z: usize,
    di: [i32; 3],
    rng: &mut rand::rngs::ThreadRng,
    r: &Vec<[f64; 3]>,
    v: &mut Vec<[f64; 3]>,
    config: &Config,
) {
    let l = grid.len();
    let i_x_new = (i_x as i32 + di[0]).rem_euclid(l as i32) as usize;
    let i_y_new = (i_y as i32 + di[1]).rem_euclid(l as i32) as usize;
    let i_z_new = (i_z as i32 + di[2]).rem_euclid(l as i32) as usize;
    for j0 in 0..grid[i_x][i_y][i_z].len() {
        for j1 in 0..grid[i_x_new][i_y_new][i_z_new].len() {
            let i0 = grid[i_x][i_y][i_z][j0];
            let i1 = grid[i_x_new][i_y_new][i_z_new][j1];
            let mut dr = [0.0; 3];
            let mut dv = [0.0; 3];

            for k in 0..3 {
                dr[k] = r[i0][k] - r[i1][k];
                if dr[k] > config.L as f64 / 2.0 {
                    dr[k] -= config.L as f64;
                } else if dr[k] < -(config.L as f64) / 2.0 {
                    dr[k] += config.L as f64;
                }
                // elsewise, dr[k] = (dr[k] + L / 2).rem_euclid(L as f64) - L / 2;, which is faster?
                dv[k] = v[i0][k] - v[i1][k];
            }
            let dr2 = dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2];
            let dv2 = dv[0] * dv[0] + dv[1] * dv[1] + dv[2] * dv[2];
            let dspeed = dv2.sqrt();
            // \vec{k} = \Delta\vec{v}_{rel} / |\Deltad\vec{v}_{rel}|
            //         = (\Delta\vec{v}_2 - \Delta\vec{v}_1) / |\Delta\vec{v}_2 - \Delta\vec{v}_1|
            let mut vec_k: [f64; 3] = [0.0; 3];
            for k in 0..3 {
                vec_k[k] = dr[k] / dr2
            }
            let mut k_factor = 1.;
            for k in 0..3 {
                if di[k] == 1 {
                    if vec_k[k] >= 0.0 {
                        k_factor = -1.;
                        break;
                    }
                    k_factor *= vec_k[k];
                } else if di[k] == 2 {
                    if vec_k[k] >= 0.0 {
                        k_factor = -1.;
                        break;
                    }
                    k_factor *= vec_k[k].powi(2);
                }
            }
            if k_factor <= 0.0 {
                continue;
            }
            let collision_prob = dspeed * config.T_STEP * config.D.powi(4) * k_factor * PI
                / (8 * config.N_TEST) as f64;
            if rng.gen_range(0.0..1.0) < collision_prob {
                let dv_dr = dv[0] * dr[0] + dv[1] * dr[1] + dv[2] * dr[2];
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

    let mut r = vec![[0.0; 3]; config.N];
    let mut theta = vec![0.0; config.N];
    let mut phi = vec![0.0; config.N];
    let mut v = vec![[0.0; 3]; config.N];
    let mut rng = rand::thread_rng();
    for i in 0..config.N {
        for j in 0..3 {
            r[i][j] = rng.gen_range(0.0..config.L as f64);
        }
        theta[i] = (1.0 - 2.0 * rng.gen_range(0.0..1.0_f64)).acos();
        phi[i] = rng.gen_range(0.0..2.0 * std::f64::consts::PI);
        v[i][0] = theta[i].sin() * phi[i].cos() * (2.0 * config.E0 / config.MASS).sqrt();
        v[i][1] = theta[i].sin() * phi[i].sin() * (2.0 * config.E0 / config.MASS).sqrt();
        v[i][2] = theta[i].cos() * (2.0 * config.E0 / config.MASS).sqrt();
    }
    let mut grid: Vec<Vec<Vec<Vec<usize>>>> =
        vec![vec![vec![vec![]; config.L]; config.L]; config.L];

    let mut pressures = vec![0.0; n_step];
    // let mut temperatures = vec![0.0; n_step];

    for i in 0..config.N {
        grid[r[i][0].floor() as usize][r[i][1].floor() as usize][r[i][2].floor() as usize].push(i);
    }

    let start = Instant::now();
    for i_t in 0..n_step {
        for i in 0..config.N {
            for j in 0..3 {
                r[i][j] += v[i][j] * config.T_STEP;
            }
        }
        let mut pressure = 0.0;
        if bounded {
            for i in 0..config.N {
                for j in 0..3 {
                    if r[i][j] < 0.0 {
                        r[i][j] = -r[i][j];
                        pressure -= 2.0 * config.MASS * v[i][j] / config.T_STEP;
                        v[i][j] = -v[i][j];
                    }
                    if r[i][j] >= config.L as f64 {
                        r[i][j] = 2.0 * config.L as f64 - r[i][j];
                        pressure += 2.0 * config.MASS * v[i][j] / config.T_STEP;
                        v[i][j] = -v[i][j];
                    }
                }
            }
        } else {
            for i in 0..config.N {
                for j in 0..3 {
                    if r[i][j] < 0.0 {
                        pressure -= 2. * config.MASS * v[i][j] / config.T_STEP;
                        r[i][j] = r[i][j].rem_euclid(config.L as f64);
                    }
                    if r[i][j] >= config.L as f64 {
                        pressure += 2. * config.MASS * v[i][j] / config.T_STEP;
                        r[i][j] = r[i][j].rem_euclid(config.L as f64);
                    }
                }
            }
        }
        pressure /= (6 * config.L.pow(2)) as f64;
        pressures[i_t] = pressure;

        let mut removes = vec![];
        for i_x in 0..config.L {
            for i_y in 0..config.L {
                for i_z in 0..config.L {
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
                            let collision_prob = dspeed * config.T_STEP * config.D * config.D * PI
                                / config.N_TEST as f64;
                            if rng.gen_range(0.0..1.0) < collision_prob {
                                let dr2 = dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2];
                                let dv_dr = dv[0] * dr[0] + dv[1] * dr[1] + dv[2] * dr[2];
                                for k in 0..3 {
                                    v[i0][k] -= dr[k] * dv_dr / dr2;
                                    v[i1][k] += dr[k] * dv_dr / dr2;
                                }
                            }
                        }
                    }
                });
            });
        });

        for i_x in 0..config.L {
            for i_y in 0..config.L {
                for i_z in 0..config.L {
                    scatt_o1(
                        &mut grid,
                        i_x,
                        i_y,
                        i_z,
                        [1, 0, 0],
                        &mut rng,
                        &r,
                        &mut v,
                        &config,
                    );
                    scatt_o1(
                        &mut grid,
                        i_x,
                        i_y,
                        i_z,
                        [0, 1, 0],
                        &mut rng,
                        &r,
                        &mut v,
                        &config,
                    );
                    scatt_o1(
                        &mut grid,
                        i_x,
                        i_y,
                        i_z,
                        [0, 0, 1],
                        &mut rng,
                        &r,
                        &mut v,
                        &config,
                    );
                    scatt_o2(
                        &mut grid,
                        i_x,
                        i_y,
                        i_z,
                        [1, 1, 0],
                        &mut rng,
                        &r,
                        &mut v,
                        &config,
                    );
                    scatt_o2(
                        &mut grid,
                        i_x,
                        i_y,
                        i_z,
                        [1, 0, 1],
                        &mut rng,
                        &r,
                        &mut v,
                        &config,
                    );
                    scatt_o2(
                        &mut grid,
                        i_x,
                        i_y,
                        i_z,
                        [0, 1, 1],
                        &mut rng,
                        &r,
                        &mut v,
                        &config,
                    );
                    scatt_o2(
                        &mut grid,
                        i_x,
                        i_y,
                        i_z,
                        [2, 0, 0],
                        &mut rng,
                        &r,
                        &mut v,
                        &config,
                    );
                    scatt_o2(
                        &mut grid,
                        i_x,
                        i_y,
                        i_z,
                        [0, 2, 0],
                        &mut rng,
                        &r,
                        &mut v,
                        &config,
                    );
                    scatt_o2(
                        &mut grid,
                        i_x,
                        i_y,
                        i_z,
                        [0, 0, 2],
                        &mut rng,
                        &r,
                        &mut v,
                        &config,
                    );
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

    let mut speed = vec![0.0; config.N];
    for i in 0..config.N {
        speed[i] = (v[i][0] * v[i][0] + v[i][1] * v[i][1] + v[i][2] * v[i][2]).sqrt();
    }

    let data_dir: String = format!(
        "data/N={}_L={}_D={}_T={}_MASS={}_N_TEST={}",
        config.N, config.L, config.D, config.T, config.MASS, config.N_TEST
    );
    std::fs::create_dir_all(&data_dir)?;
    let mut file = File::create(format!("{}/speed.csv", data_dir))?;
    writeln!(file, "speed")?;
    for value in speed {
        writeln!(file, "{}", value)?;
    }
    file = File::create(format!("{}/pressure.csv", data_dir))?;
    writeln!(file, "time,pressure")?;
    for (i, value) in pressures.iter().enumerate() {
        writeln!(file, "{},{}", i as f64 * config.T_STEP, value)?;
    }
    Ok(())
}
