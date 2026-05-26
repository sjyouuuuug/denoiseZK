use crate::{
    experiments::{
        runner::run_conv_case,
        schema::{ExperimentResult, RangeMode, RunMode},
    },
    models::denoise_update::DenoiseUpdateMode,
};

pub const NAME: &str = "ablation";
pub const TITLE: &str = "Range check and fused update ablation";

pub fn run() -> Vec<ExperimentResult> {
    vec![
        run_conv_case::<64, 8, 8, 2, 3, 3, 8, 8>(
            "ablation_baseline_onehot_double",
            NAME,
            DenoiseUpdateMode::DoubleFloor,
            RangeMode::OneHot,
            RunMode::BuildOnly,
            4,
            2,
            2,
        ),
        run_conv_case::<64, 8, 8, 2, 3, 3, 8, 8>(
            "ablation_a_bits_double",
            NAME,
            DenoiseUpdateMode::DoubleFloor,
            RangeMode::Bits,
            RunMode::BuildOnly,
            4,
            2,
            2,
        ),
        run_conv_case::<64, 8, 8, 2, 3, 3, 8, 8>(
            "ablation_b_onehot_fused",
            NAME,
            DenoiseUpdateMode::FusedFloor,
            RangeMode::OneHot,
            RunMode::BuildOnly,
            4,
            2,
            2,
        ),
        run_conv_case::<64, 8, 8, 2, 3, 3, 8, 8>(
            "ablation_ab_bits_fused",
            NAME,
            DenoiseUpdateMode::FusedFloor,
            RangeMode::Bits,
            RunMode::BuildOnly,
            4,
            2,
            2,
        ),
    ]
}
