pub mod circuit;
pub mod params;
pub mod trace;
pub mod util;

pub use circuit::AffineCircuit;
pub use params::AffineParams;
pub use trace::{generate_affine_trace, AffineIteration};

pub mod fixed_point;
