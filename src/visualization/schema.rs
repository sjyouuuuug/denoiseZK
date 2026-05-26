use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VisualDemo {
    pub metadata: DemoMetadata,
    pub config: DemoConfig,
    pub trajectory: Vec<VisualStep>,
    pub nova_steps: Vec<NovaStepView>,
    pub proof: ProofSummary,
    pub comparisons: Vec<ComparisonSeries>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DemoMetadata {
    pub title: String,
    pub backend: String,
    pub update_mode: String,
    pub description: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DemoConfig {
    pub scale: i64,
    pub total_iters: usize,
    pub num_steps: usize,
    pub num_iters_per_step: usize,
    pub state_height: usize,
    pub state_width: usize,
    pub time_embedding_dim: usize,
    pub lookup_range: String,
    pub clip_max: i64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VisualStep {
    pub t: usize,
    pub x: Vec<Vec<i64>>,
    pub epsilon: Vec<Vec<i64>>,
    pub x_next: Vec<Vec<i64>>,
    pub time_embedding: Vec<i64>,
    pub alpha: i64,
    pub beta: i64,
    pub predictor: PredictorSummary,
    pub update: UpdateSummary,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PredictorSummary {
    pub backend: String,
    pub hidden: Option<Vec<i64>>,
    pub conv_raw: Option<Vec<Vec<i64>>>,
    pub conv_pre_activation: Option<Vec<Vec<i64>>>,
    pub relu_output: Option<Vec<Vec<i64>>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct UpdateSummary {
    pub mode: String,
    pub alpha_term: Vec<Vec<i64>>,
    pub beta_term: Vec<Vec<i64>>,
    pub fused_raw: Option<Vec<Vec<i64>>>,
    pub output: Vec<Vec<i64>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NovaStepView {
    pub nova_step_index: usize,
    pub iter_start: usize,
    pub iter_end: usize,
    pub label: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProofSummary {
    pub recursive_verified: bool,
    pub compressed_verified: bool,
    pub primary_constraints: usize,
    pub secondary_constraints: usize,
    pub primary_variables: usize,
    pub secondary_variables: usize,
    pub recursive_prove_ms: f64,
    pub recursive_verify_ms: f64,
    pub compressed_prove_ms: f64,
    pub compressed_verify_ms: f64,
    pub proof_size_bytes: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ComparisonSeries {
    pub name: String,
    pub metrics: Vec<ComparisonMetric>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ComparisonMetric {
    pub label: String,
    pub value: f64,
    pub unit: String,
}
