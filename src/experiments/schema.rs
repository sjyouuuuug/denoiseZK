use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum RangeMode {
    OneHot,
    Bits,
}

impl RangeMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::OneHot => "OneHot",
            Self::Bits => "Bits",
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum RunMode {
    FullProof,
    RecursiveOnly,
    BuildOnly,
}

impl RunMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::FullProof => "FullProof",
            Self::RecursiveOnly => "RecursiveOnly",
            Self::BuildOnly => "BuildOnly",
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum ExperimentStatus {
    Ok,
    Overflow,
    BuildOk,
    Failed,
    SkippedTooSlow,
}

impl ExperimentStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Ok => "OK",
            Self::Overflow => "OVERFLOW",
            Self::BuildOk => "BUILD_OK",
            Self::Failed => "FAILED",
            Self::SkippedTooSlow => "SKIPPED_TOO_SLOW",
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExperimentResult {
    pub case: String,
    pub group: String,
    pub backend: String,
    pub update_mode: String,
    pub n: usize,
    pub hidden: Option<usize>,
    pub image_h: Option<usize>,
    pub image_w: Option<usize>,
    pub kernel_h: Option<usize>,
    pub kernel_w: Option<usize>,
    pub total_iters: usize,
    pub num_steps: usize,
    pub num_iters_per_step: usize,
    pub scale: i64,
    pub range_mode: String,
    pub run_mode: String,
    pub status: String,
    pub primary_constraints: Option<usize>,
    pub secondary_constraints: Option<usize>,
    pub primary_variables: Option<usize>,
    pub secondary_variables: Option<usize>,
    pub proof_size_bytes: Option<usize>,
    pub recursive_prove_ms: Option<f64>,
    pub compressed_prove_ms: Option<f64>,
    pub compressed_verify_ms: Option<f64>,
    pub witness_gen_ms: Option<f64>,
    pub setup_ms: Option<f64>,
    pub error: Option<String>,
}

impl ExperimentResult {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        case: &str,
        group: &str,
        backend: &str,
        update_mode: &str,
        n: usize,
        hidden: Option<usize>,
        image_h: Option<usize>,
        image_w: Option<usize>,
        kernel_h: Option<usize>,
        kernel_w: Option<usize>,
        total_iters: usize,
        num_steps: usize,
        num_iters_per_step: usize,
        scale: i64,
        range_mode: RangeMode,
        run_mode: RunMode,
    ) -> Self {
        assert!(
            n >= 16,
            "official denoise experiment cases must use N >= 16"
        );
        Self {
            case: case.to_string(),
            group: group.to_string(),
            backend: backend.to_string(),
            update_mode: update_mode.to_string(),
            n,
            hidden,
            image_h,
            image_w,
            kernel_h,
            kernel_w,
            total_iters,
            num_steps,
            num_iters_per_step,
            scale,
            range_mode: range_mode.as_str().to_string(),
            run_mode: run_mode.as_str().to_string(),
            status: "FAILED".to_string(),
            primary_constraints: None,
            secondary_constraints: None,
            primary_variables: None,
            secondary_variables: None,
            proof_size_bytes: None,
            recursive_prove_ms: None,
            compressed_prove_ms: None,
            compressed_verify_ms: None,
            witness_gen_ms: None,
            setup_ms: None,
            error: None,
        }
    }

    pub fn set_status(&mut self, status: ExperimentStatus) {
        self.status = status.as_str().to_string();
    }
}
