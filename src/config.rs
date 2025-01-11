use serde_derive::Deserialize;
use std::fs;

#[derive(Deserialize)]
pub struct Config {
    pub N: usize,
    pub L: usize,
    pub D: f64,
    pub T: f64,
    pub MASS: f64,
    pub N_TEST: usize,
    pub E0: f64,
    pub T_STEP: f64,
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
