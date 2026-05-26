pub mod circuit;
pub mod params;
pub mod runner;
pub mod trace;

pub use circuit::AffineClippedReluLookupCircuit;
pub use params::{AffineClippedReluLookupParams, IntAffineClippedReluLookupParams};
pub use runner::{
    build_placeholder_circuit as build_affine_clipped_relu_lookup_placeholder_circuit,
    build_step_circuits as build_affine_clipped_relu_lookup_step_circuits,
    compress_and_verify as compress_affine_clipped_relu_lookup_and_verify,
    run_recursive as run_affine_clipped_relu_lookup_recursive,
    setup_public_params as setup_affine_clipped_relu_lookup_public_params,
    verify_recursive as verify_affine_clipped_relu_lookup_recursive,
};
pub use trace::{generate_affine_clipped_relu_lookup_trace, AffineClippedReluLookupIteration};
