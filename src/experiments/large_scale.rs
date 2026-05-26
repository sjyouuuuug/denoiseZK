use crate::{
    experiments::{
        runner::run_conv_case,
        schema::{ExperimentResult, RangeMode, RunMode},
    },
    models::denoise_update::DenoiseUpdateMode,
};

pub const NAME: &str = "large_scale";
pub const TITLE: &str = "Large dimension cases";

pub fn run() -> Vec<ExperimentResult> {
    vec![
        run_conv_case::<256, 16, 16, 2, 3, 3, 16, 16>(
            "large_conv_256",
            NAME,
            DenoiseUpdateMode::FusedFloor,
            RangeMode::Bits,
            RunMode::BuildOnly,
            1,
            1,
            1,
        ),
        run_conv_case::<1024, 32, 32, 2, 3, 3, 32, 32>(
            "large_conv_1024",
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
