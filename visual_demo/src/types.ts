export interface VisualDemo {
  metadata: DemoMetadata;
  config: DemoConfig;
  trajectory: VisualStep[];
  nova_steps: NovaStepView[];
  proof: ProofSummary;
  comparisons: ComparisonSeries[];
}

export interface DemoMetadata {
  title: string;
  backend: string;
  update_mode: string;
  description: string;
}

export interface DemoConfig {
  scale: number;
  total_iters: number;
  num_steps: number;
  num_iters_per_step: number;
  state_height: number;
  state_width: number;
  time_embedding_dim: number;
  lookup_range: string;
  clip_max: number;
}

export interface VisualStep {
  t: number;
  x: number[][];
  epsilon: number[][];
  x_next: number[][];
  time_embedding: number[];
  alpha: number;
  beta: number;
  predictor: PredictorSummary;
  update: UpdateSummary;
}

export interface PredictorSummary {
  backend: string;
  hidden?: number[] | null;
  conv_raw?: number[][] | null;
  conv_pre_activation?: number[][] | null;
  relu_output?: number[][] | null;
}

export interface UpdateSummary {
  mode: string;
  alpha_term: number[][];
  beta_term: number[][];
  fused_raw?: number[][] | null;
  output: number[][];
}

export interface NovaStepView {
  nova_step_index: number;
  iter_start: number;
  iter_end: number;
  label: string;
}

export interface ProofSummary {
  recursive_verified: boolean;
  compressed_verified: boolean;
  primary_constraints: number;
  secondary_constraints: number;
  primary_variables: number;
  secondary_variables: number;
  recursive_prove_ms: number;
  recursive_verify_ms: number;
  compressed_prove_ms: number;
  compressed_verify_ms: number;
  proof_size_bytes: number;
}

export interface ComparisonSeries {
  name: string;
  metrics: ComparisonMetric[];
}

export interface ComparisonMetric {
  label: string;
  value: number;
  unit: string;
}
