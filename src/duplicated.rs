        // let mut len: usize;
        // let mut i0: usize;
        // let mut i1: usize;
        // let mut dr: Vec<f64>;
        // let mut dv: Vec<f64>;
        // let mut dv2: f64;
        // let mut dspeed: f64;
        // let mut collision_prob: f64;
        // let mut dr2: f64;
        // let mut dv_dr: f64;

                    // let mut len: usize;
                    // let mut i0: usize;
                    // let mut i1: usize;
                    // let mut dr: Vec<f64>;
                    // let mut dv: Vec<f64>;
                    // let mut dv2: f64;
                    // let mut dspeed: f64;
                    // let mut collision_prob: f64;
                    // let mut dr2: f64;
                    // let mut dv_dr: f64;
                    //
                    // len = grid[i_x][i_y][i_z].len();
                    // for j0 in 0..len {
                    //     for j1 in j0+1..len {
                    //         i0 = grid[i_x][i_y][i_z][j0];
                    //         i1 = grid[i_x][i_y][i_z][j1];
                    //         dr = (0..3).map(|k| r[i0][k] - r[i1][k]).collect();
                    //         dv = (0..3).map(|k| v[i0][k] - v[i1][k]).collect();
                    //         dv2 = dv.iter().map(|&x| x * x).sum();
                    //         dspeed = dv2.sqrt();
                    //         collision_prob = dspeed * STEP * D * D * PI / N_TEST as f64;
                    //         if rng.gen_range(0.0..1.0) < collision_prob {
                    //             dr2 = dr.iter().map(|&x| x * x).sum();
                    //             dv_dr = dv.iter().zip(dr.iter()).map(|(&dv_k, &dr_k)| dv_k * dr_k).sum();
                    //             for k in 0..3 {
                    //                 v[i0][k] -= dr[k] * dv_dr / dr2;
                    //                 v[i1][k] += dr[k] * dv_dr / dr2;
                    //             }
                    //         }
                    //     }
                    // }

                //     for i_x in 0..L {
                //         for i_y in 0..L {
                //             for i_z in 0..L {
                //                 for j0 in 0..grid[i_x][i_y][i_z].len() {
                //                     for j1 in j0 + 1..grid[i_x][i_y][i_z].len() {
                //                         let i0 = grid[i_x][i_y][i_z][j0];
                //                         let i1 = grid[i_x][i_y][i_z][j1];
                //                         let mut dr = [0.0; 3];
                //                         let mut dv = [0.0; 3];
                //                         for k in 0..3 {
                //                             dr[k] = r[i0][k] - r[i1][k];
                //                             dv[k] = v[i0][k] - v[i1][k];
                //                         }
                //                         let dv2 = dv[0] * dv[0] + dv[1] * dv[1] + dv[2] * dv[2];
                //                         let dspeed = dv2.sqrt();
                //                         let collision_prob = dspeed * STEP * D * D * PI / N_TEST as f64;
                //                         if rng.gen_range(0.0..1.0) < collision_prob {
                //                             let dr2 = dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2];
                //                             let dv_dr = dv[0] * dr[0] + dv[1] * dr[1] + dv[2] * dr[2];
                //                             for k in 0..3 {
                //                                 v[i0][k] -= dr[k] * dv_dr / dr2;
                //                                 v[i1][k] += dr[k] * dv_dr / dr2;
                //                             }
                //                         }
                //                     }
                //                 }
                //             }
                //         }
                //     }