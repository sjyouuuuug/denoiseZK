use crate::{
    experiments::{
        runner::{run_conv_case, run_mlp_case},
        schema::{ExperimentResult, RangeMode, RunMode},
    },
    models::denoise_update::DenoiseUpdateMode,
};

pub const NAME: &str = "steps_scaling";
pub const TITLE: &str = "Recursive step scaling";

pub fn run() -> Vec<ExperimentResult> {
    let mut results = Vec::new();
    for (case, total_iters, num_steps) in [
        ("steps_mlp_t2", 2, 1),
        ("steps_mlp_t4", 4, 2),
        ("steps_mlp_t8", 8, 4),
        ("steps_mlp_t16", 16, 8),
        ("steps_mlp_t24", 24, 12),
        ("steps_mlp_t32", 32, 16),
        ("steps_mlp_t48", 48, 24),
        ("steps_mlp_t64", 64, 32),
    ] {
        results.push(run_mlp_case::<32, 4, 36, 32>(
            case,
            NAME,
            DenoiseUpdateMode::FusedFloor,
            RangeMode::Bits,
            RunMode::BuildOnly,
            total_iters,
            num_steps,
            2,
        ));
    }
    for (case, total_iters, num_steps) in [
        ("steps_conv_t2", 2, 1),
        ("steps_conv_t4", 4, 2),
        ("steps_conv_t8", 8, 4),
        ("steps_conv_t16", 16, 8),
        ("steps_conv_t24", 24, 12),
        ("steps_conv_t32", 32, 16),
        ("steps_conv_t48", 48, 24),
        ("steps_conv_t64", 64, 32),
    ] {
        results.push(run_conv_case::<64, 8, 8, 2, 3, 3, 8, 8>(
            case,
            NAME,
            DenoiseUpdateMode::FusedFloor,
            RangeMode::Bits,
            RunMode::BuildOnly,
            total_iters,
            num_steps,
            2,
        ));
    }
    results
}
