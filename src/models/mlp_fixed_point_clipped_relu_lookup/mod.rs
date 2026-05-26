pub mod circuit;
pub mod commitment_circuit;
pub mod params;
pub mod runner;
pub mod trace;

pub use circuit::PublicFixedPointMlpClippedReluCircuit;
pub use commitment_circuit::PublicFixedPointMlpCommitmentCircuit;
pub use params::{FixedMlpClippedReluPublicParams, FixedMlpClippedReluStepParams};
pub use runner::{
    build_placeholder_circuit as build_fixed_point_mlp_placeholder_circuit,
    build_step_circuits as build_fixed_point_mlp_step_circuits,
    compress_and_verify as compress_fixed_point_mlp_and_verify,
    run_recursive as run_fixed_point_mlp_recursive,
    setup_public_params as setup_fixed_point_mlp_public_params,
    verify_recursive as verify_fixed_point_mlp_recursive,
};
pub use trace::{generate_fixed_point_mlp_trace, FixedPointMlpClippedReluIteration};
