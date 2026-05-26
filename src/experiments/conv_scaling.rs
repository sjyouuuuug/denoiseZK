use crate::{
    experiments::{
        runner::run_conv_case,
        schema::{ExperimentResult, RangeMode, RunMode},
    },
    models::denoise_update::DenoiseUpdateMode,
};

pub const NAME: &str = "conv_scaling";
pub const TITLE: &str = "Conv scaling";

pub fn run() -> Vec<ExperimentResult> {
    vec![
        run_conv_case::<16, 4, 4, 2, 3, 3, 4, 4>(
            "conv_scale_4x4",
            NAME,
            DenoiseUpdateMode::FusedFloor,
            RangeMode::Bits,
            RunMode::BuildOnly,
            4,
            2,
            2,
        ),
        run_conv_case::<64, 8, 8, 2, 3, 3, 8, 8>(
            "conv_scale_8x8",
            NAME,
            DenoiseUpdateMode::FusedFloor,
            RangeMode::Bits,
            RunMode::BuildOnly,
            4,
            2,
            2,
        ),
        run_conv_case::<256, 16, 16, 2, 3, 3, 16, 16>(
            "conv_scale_16x16",
            NAME,
            DenoiseUpdateMode::FusedFloor,
            RangeMode::Bits,
            RunMode::BuildOnly,
            1,
            1,
            1,
        ),
        run_conv_case::<1024, 32, 32, 2, 3, 3, 32, 32>(
            "conv_scale_32x32",
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
