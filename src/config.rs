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
