use std::f64::consts::PI;
use std::{io, vec};
use std::io::prelude::*;

use rand::Rng;
use std::time::Instant;
use std::fs::File;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    const N: usize = 131072;
    const L: usize = 16;
    const D: f64 = 0.1;
    const T: f64 = 1.0;
    const MASS: f64 = 200.0;
    const N_TEST: usize = 1;

    const E0: f64 = 1.5 * T;
    let mut r = vec![[0.0; 3]; N];
    let mut theta = vec![0.0; N];
    let mut phi = vec![0.0; N];
    let mut v = vec![[0.0; 3]; N];
    let mut rng = rand::thread_rng();
    for i in 0..N {
        for j in 0..3 {
            r[i][j] = rng.gen_range(0.0..L as f64);
        }
        theta[i] = (1.0 - 2.0 * rng.gen_range(0.0..1.0_f64)).acos();
        phi[i] = rng.gen_range(0.0..2.0 * std::f64::consts::PI);
        v[i][0] = theta[i].sin() * phi[i].cos() * (2.0 * E0 / MASS).sqrt();
        v[i][1] = theta[i].sin() * phi[i].sin() * (2.0 * E0 / MASS).sqrt();
        v[i][2] = theta[i].cos() * (2.0 * E0 / MASS).sqrt();
    }
    let mut grid: Vec<Vec<Vec<Vec<usize>>>> = vec![vec![vec![vec![]; L]; L]; L];
    for i in 0..N {
        grid[r[i][0].floor() as usize][r[i][1].floor() as usize][r[i][2].floor() as usize].push(i);
    }

    const T_STEP: f64 = 0.2;
    const N_STEP: usize = 500;
    let start = Instant::now();
    for i_t in 0..N_STEP {
        for i in 0..N {
            for j in 0..3 {
                r[i][j] += v[i][j] * T_STEP;
                if r[i][j] < 0.0 {
                    r[i][j] = -r[i][j];
                    v[i][j] = -v[i][j];
                }
                if r[i][j] >= L as f64 {
                    r[i][j] = 2.0 * L as f64 - r[i][j];
                    v[i][j] = -v[i][j];
                }
            }
        }
        let mut removes = vec![];
        for i_x in 0..L {
            for i_y in 0..L {
                for i_z in 0..L {
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
                            let collision_prob = dspeed * T_STEP * D * D * PI / N_TEST as f64;
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
        print!("{i_t}/{N_STEP}\r");
        io::stdout().flush().unwrap();
    }
    let elapsed = start.elapsed();
    println!(
        "Elapsed: {}.{:03} s",
        elapsed.as_secs(),
        elapsed.subsec_millis()
    );
    println!("{} ms per step", elapsed.as_millis() / N_STEP as u128);

    let mut speed = vec![0.0; N];
    for i in 0..N {
        speed[i] = (v[i][0] * v[i][0] + v[i][1] * v[i][1] + v[i][2] * v[i][2]).sqrt();
    }

    let mut file = File::create("data/speed.csv")?;
    writeln!(file, "speed")?;
    for value in speed {
        writeln!(file, "{}", value)?;
    }
    Ok(())
}
