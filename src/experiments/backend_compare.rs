use crate::{
    experiments::{
        runner::{run_conv_case, run_mlp_case},
        schema::{ExperimentResult, RangeMode, RunMode},
    },
    models::denoise_update::DenoiseUpdateMode,
};

pub const NAME: &str = "backend_compare";
pub const TITLE: &str = "MLP vs Conv";

pub fn run() -> Vec<ExperimentResult> {
    vec![
        run_mlp_case::<16, 4, 20, 16>(
            "compare_16_mlp",
            NAME,
            DenoiseUpdateMode::FusedFloor,
            RangeMode::Bits,
            RunMode::BuildOnly,
            4,
            2,
            2,
        ),
        run_conv_case::<16, 4, 4, 2, 3, 3, 4, 4>(
            "compare_16_conv",
            NAME,
            DenoiseUpdateMode::FusedFloor,
            RangeMode::Bits,
            RunMode::BuildOnly,
            4,
            2,
            2,
        ),
        run_mlp_case::<64, 4, 68, 64>(
            "compare_64_mlp",
            NAME,
            DenoiseUpdateMode::FusedFloor,
            RangeMode::Bits,
            RunMode::BuildOnly,
            2,
            1,
            2,
        ),
        run_conv_case::<64, 8, 8, 2, 3, 3, 8, 8>(
            "compare_64_conv",
            NAME,
            DenoiseUpdateMode::FusedFloor,
            RangeMode::Bits,
            RunMode::BuildOnly,
            2,
            1,
            2,
        ),
        run_conv_case::<256, 16, 16, 2, 3, 3, 16, 16>(
            "compare_256_conv",
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
