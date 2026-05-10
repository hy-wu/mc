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
