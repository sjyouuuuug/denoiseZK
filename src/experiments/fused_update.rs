use crate::{
    experiments::{
        runner::{run_conv_case, run_mlp_case},
        schema::{ExperimentResult, RangeMode, RunMode},
    },
    models::denoise_update::DenoiseUpdateMode,
};

pub const NAME: &str = "fused_update";
pub const TITLE: &str = "DoubleFloor vs FusedFloor";

pub fn run() -> Vec<ExperimentResult> {
    let mut results = Vec::new();
    for (case, mode) in [
        ("fused_mlp_16_double", DenoiseUpdateMode::DoubleFloor),
        ("fused_mlp_16_fused", DenoiseUpdateMode::FusedFloor),
    ] {
        results.push(run_mlp_case::<16, 4, 20, 16>(
            case,
            NAME,
            mode,
            RangeMode::Bits,
            RunMode::BuildOnly,
            4,
            2,
            2,
        ));
    }
    for (case, mode) in [
        ("fused_mlp_64_double", DenoiseUpdateMode::DoubleFloor),
        ("fused_mlp_64_fused", DenoiseUpdateMode::FusedFloor),
    ] {
        results.push(run_mlp_case::<64, 4, 68, 64>(
            case,
            NAME,
            mode,
            RangeMode::Bits,
            RunMode::BuildOnly,
            2,
            1,
            2,
        ));
    }
    for (case, mode) in [
        ("fused_mlp_128_double", DenoiseUpdateMode::DoubleFloor),
        ("fused_mlp_128_fused", DenoiseUpdateMode::FusedFloor),
    ] {
        results.push(run_mlp_case::<128, 4, 132, 128>(
            case,
            NAME,
            mode,
            RangeMode::Bits,
            RunMode::BuildOnly,
            1,
            1,
            1,
        ));
    }
    results.push(run_conv_case::<16, 4, 4, 2, 3, 3, 4, 4>(
        "fused_conv_4x4_double",
        NAME,
        DenoiseUpdateMode::DoubleFloor,
        RangeMode::Bits,
        RunMode::BuildOnly,
        4,
        2,
        2,
    ));
    results.push(run_conv_case::<16, 4, 4, 2, 3, 3, 4, 4>(
        "fused_conv_4x4_fused",
        NAME,
        DenoiseUpdateMode::FusedFloor,
        RangeMode::Bits,
        RunMode::BuildOnly,
        4,
        2,
        2,
    ));
    for (case, mode) in [
        ("fused_conv_8x8_double", DenoiseUpdateMode::DoubleFloor),
        ("fused_conv_8x8_fused", DenoiseUpdateMode::FusedFloor),
    ] {
        results.push(run_conv_case::<64, 8, 8, 2, 3, 3, 8, 8>(
            case,
            NAME,
            mode,
            RangeMode::Bits,
            RunMode::BuildOnly,
            4,
            2,
            2,
        ));
    }
    for (case, mode) in [
        ("fused_conv_16x16_double", DenoiseUpdateMode::DoubleFloor),
        ("fused_conv_16x16_fused", DenoiseUpdateMode::FusedFloor),
    ] {
        results.push(run_conv_case::<256, 16, 16, 2, 3, 3, 16, 16>(
            case,
            NAME,
            mode,
            RangeMode::Bits,
            RunMode::BuildOnly,
            1,
            1,
            1,
        ));
    }
    results
}
