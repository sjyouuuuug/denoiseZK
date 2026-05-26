use crate::{
    experiments::{
        runner::{run_conv_case, run_mlp_case},
        schema::{ExperimentResult, RangeMode, RunMode},
    },
    models::denoise_update::DenoiseUpdateMode,
};

pub const NAME: &str = "correctness";
pub const TITLE: &str = "End-to-end correctness";

pub fn run() -> Vec<ExperimentResult> {
    vec![
        run_mlp_case::<16, 4, 20, 16>(
            "correctness_mlp_16",
            NAME,
            DenoiseUpdateMode::DoubleFloor,
            RangeMode::Bits,
            RunMode::FullProof,
            4,
            2,
            2,
        ),
        run_mlp_case::<32, 4, 36, 32>(
            "correctness_mlp_32",
            NAME,
            DenoiseUpdateMode::DoubleFloor,
            RangeMode::Bits,
            RunMode::FullProof,
            4,
            2,
            2,
        ),
        run_conv_case::<16, 4, 4, 2, 3, 3, 4, 4>(
            "correctness_conv_4x4",
            NAME,
            DenoiseUpdateMode::DoubleFloor,
            RangeMode::Bits,
            RunMode::FullProof,
            4,
            2,
            2,
        ),
        run_conv_case::<64, 8, 8, 2, 3, 3, 8, 8>(
            "correctness_conv_8x8",
            NAME,
            DenoiseUpdateMode::DoubleFloor,
            RangeMode::Bits,
            RunMode::FullProof,
            4,
            2,
            2,
        ),
        run_conv_case::<100, 10, 10, 2, 3, 3, 10, 10>(
            "correctness_conv_10x10",
            NAME,
            DenoiseUpdateMode::DoubleFloor,
            RangeMode::Bits,
            RunMode::FullProof,
            4,
            2,
            2,
        ),
        run_conv_case::<128, 8, 16, 2, 3, 3, 8, 16>(
            "correctness_conv_8x16_build",
            NAME,
            DenoiseUpdateMode::DoubleFloor,
            RangeMode::Bits,
            RunMode::BuildOnly,
            4,
            2,
            2,
        ),
        run_conv_case::<256, 16, 16, 2, 3, 3, 16, 16>(
            "correctness_conv_16x16_build",
            NAME,
            DenoiseUpdateMode::DoubleFloor,
            RangeMode::Bits,
            RunMode::BuildOnly,
            1,
            1,
            1,
        ),
        run_conv_case::<1024, 32, 32, 2, 3, 3, 32, 32>(
            "correctness_conv_32x32_build",
            NAME,
            DenoiseUpdateMode::DoubleFloor,
            RangeMode::Bits,
            RunMode::BuildOnly,
            1,
            1,
            1,
        ),
    ]
}
