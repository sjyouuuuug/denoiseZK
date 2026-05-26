pub mod circuit;
pub mod params;
pub mod runner;
pub mod trace;

pub use circuit::PublicMlpClippedReluCircuit;
pub use params::{IntMlpClippedReluStepParams, MlpClippedReluPublicParams};
pub use runner::{
    build_placeholder_circuit as build_mlp_clipped_relu_placeholder_circuit,
    build_step_circuits as build_mlp_clipped_relu_step_circuits,
    compress_and_verify as compress_mlp_clipped_relu_and_verify,
    run_recursive as run_mlp_clipped_relu_recursive,
    setup_public_params as setup_mlp_clipped_relu_public_params,
    verify_recursive as verify_mlp_clipped_relu_recursive,
};
pub use trace::{generate_mlp_clipped_relu_trace, MlpClippedReluIteration};
