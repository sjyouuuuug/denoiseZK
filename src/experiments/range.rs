use crate::{
    experiments::{
        runner::run_conv_case,
        schema::{ExperimentResult, RangeMode, RunMode},
    },
    models::denoise_update::DenoiseUpdateMode,
};

pub const NAME: &str = "range";
pub const TITLE: &str = "Range check optimization";

pub fn run() -> Vec<ExperimentResult> {
    let mut results = Vec::new();
    // for (case, range_mode) in [
    //     ("range_mlp_16_onehot", RangeMode::OneHot),
    //     ("range_mlp_16_bits", RangeMode::Bits),
    // ] {
    //     results.push(run_mlp_case::<16, 4, 20, 16>(
    //         case,
    //         NAME,
    //         DenoiseUpdateMode::DoubleFloor,
    //         range_mode,
    //         RunMode::FullProof,
    //         2,
    //         1,
    //         2,
    //     ));
    // }
    // for (case, range_mode) in [
    //     ("range_mlp_64_onehot", RangeMode::OneHot),
    //     ("range_mlp_64_bits", RangeMode::Bits),
    // ] {
    //     results.push(run_mlp_case::<64, 4, 68, 64>(
    //         case,
    //         NAME,
    //         DenoiseUpdateMode::DoubleFloor,
    //         range_mode,
    //         RunMode::FullProof,
    //         2,
    //         1,
    //         2,
    //     ));
    // }
    for (case, range_mode) in [
        ("range_conv_8x8_onehot", RangeMode::OneHot),
        ("range_conv_8x8_bits", RangeMode::Bits),
    ] {
        results.push(run_conv_case::<64, 8, 8, 2, 3, 3, 8, 8>(
            case,
            NAME,
            DenoiseUpdateMode::DoubleFloor,
            range_mode,
            RunMode::FullProof,
            2,
            1,
            2,
        ));
    }
    results
}
