use crate::{
    experiments::{
        runner::run_mlp_case,
        schema::{ExperimentResult, RangeMode, RunMode},
    },
    models::denoise_update::DenoiseUpdateMode,
};

pub const NAME: &str = "mlp_scaling";
pub const TITLE: &str = "MLP scaling";

pub fn run() -> Vec<ExperimentResult> {
    vec![
        run_mlp_case::<16, 4, 20, 16>(
            "mlp_scale_16",
            NAME,
            DenoiseUpdateMode::FusedFloor,
            RangeMode::Bits,
            RunMode::BuildOnly,
            4,
            2,
            2,
        ),
        run_mlp_case::<32, 4, 36, 32>(
            "mlp_scale_32",
            NAME,
            DenoiseUpdateMode::FusedFloor,
            RangeMode::Bits,
            RunMode::BuildOnly,
            4,
            2,
            2,
        ),
        run_mlp_case::<64, 4, 68, 64>(
            "mlp_scale_64",
            NAME,
            DenoiseUpdateMode::FusedFloor,
            RangeMode::Bits,
            RunMode::BuildOnly,
            2,
            1,
            2,
        ),
        run_mlp_case::<128, 4, 132, 128>(
            "mlp_scale_128",
            NAME,
            DenoiseUpdateMode::FusedFloor,
            RangeMode::Bits,
            RunMode::BuildOnly,
            1,
            1,
            1,
        ),
    ]
}
