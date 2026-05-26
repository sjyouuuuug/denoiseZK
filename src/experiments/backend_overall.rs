use crate::{
    experiments::{
        runner::{run_conv_case, run_mlp_case},
        schema::{ExperimentResult, RangeMode, RunMode},
    },
    models::denoise_update::DenoiseUpdateMode,
};

pub const NAME: &str = "backend_overall";
pub const TITLE: &str = "Overall backend performance";

const TOTAL_ITERS: usize = 4;
const NUM_STEPS: usize = 2;
const ITERS_PER_STEP: usize = 2;

pub fn run() -> Vec<ExperimentResult> {
    vec![
        run_mlp_case::<16, 4, 20, 16>(
            "overall_mlp_16",
            NAME,
            DenoiseUpdateMode::FusedFloor,
            RangeMode::Bits,
            RunMode::FullProof,
            TOTAL_ITERS,
            NUM_STEPS,
            ITERS_PER_STEP,
        ),
        run_conv_case::<16, 4, 4, 2, 3, 3, 4, 4>(
            "overall_conv_16",
            NAME,
            DenoiseUpdateMode::FusedFloor,
            RangeMode::Bits,
            RunMode::FullProof,
            TOTAL_ITERS,
            NUM_STEPS,
            ITERS_PER_STEP,
        ),
        run_mlp_case::<64, 4, 68, 64>(
            "overall_mlp_64",
            NAME,
            DenoiseUpdateMode::FusedFloor,
            RangeMode::Bits,
            RunMode::FullProof,
            TOTAL_ITERS,
            NUM_STEPS,
            ITERS_PER_STEP,
        ),
        run_conv_case::<64, 8, 8, 2, 3, 3, 8, 8>(
            "overall_conv_64",
            NAME,
            DenoiseUpdateMode::FusedFloor,
            RangeMode::Bits,
            RunMode::FullProof,
            TOTAL_ITERS,
            NUM_STEPS,
            ITERS_PER_STEP,
        ),
        // run_mlp_case::<128, 4, 132, 128>(
        //     "overall_mlp_128",
        //     NAME,
        //     DenoiseUpdateMode::FusedFloor,
        //     RangeMode::Bits,
        //     RunMode::FullProof,
        //     TOTAL_ITERS,
        //     NUM_STEPS,
        //     ITERS_PER_STEP,
        // ),
        run_conv_case::<128, 8, 16, 2, 3, 3, 8, 16>(
            "overall_conv_128",
            NAME,
            DenoiseUpdateMode::FusedFloor,
            RangeMode::Bits,
            RunMode::FullProof,
            TOTAL_ITERS,
            NUM_STEPS,
            ITERS_PER_STEP,
        ),
        run_mlp_case::<256, 4, 260, 256>(
            "overall_mlp_256_build",
            NAME,
            DenoiseUpdateMode::FusedFloor,
            RangeMode::Bits,
            RunMode::BuildOnly,
            TOTAL_ITERS,
            NUM_STEPS,
            ITERS_PER_STEP,
        ),
        run_conv_case::<256, 16, 16, 2, 3, 3, 16, 16>(
            "overall_conv_256_build",
            NAME,
            DenoiseUpdateMode::FusedFloor,
            RangeMode::Bits,
            RunMode::BuildOnly,
            TOTAL_ITERS,
            NUM_STEPS,
            ITERS_PER_STEP,
        ),
        run_conv_case::<1024, 32, 32, 2, 3, 3, 32, 32>(
            "overall_conv_1024_build",
            NAME,
            DenoiseUpdateMode::FusedFloor,
            RangeMode::Bits,
            RunMode::BuildOnly,
            TOTAL_ITERS,
            NUM_STEPS,
            ITERS_PER_STEP,
        ),
    ]
}
