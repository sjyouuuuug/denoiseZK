use ff::Field;

use crate::{
    clipped_relu::field_from_i64,
    commitment::{toy_hash_block_prefixes_i64_from_field, TOY_HASH_BASE_U64},
    denoise_fixed_point_time_embedding::{
        generate_fixed_point_denoise_time_embedding_trace, FixedDenoiseTimeEmbeddingIteration,
        FixedDenoiseTimeEmbeddingPublicParams, FixedDenoiseTimeEmbeddingStepParams,
        PublicFixedPointDenoiseTimeEmbeddingCircuit,
    },
    fixed_point::FixedPointConfig,
    models::denoise_update::{DenoiseUpdateMode, DenoiseUpdateWitness},
    nova_ivc::{F1, G1},
    public_state::CommittedDenoiseStateLayout,
};

pub use crate::denoise_fixed_point_time_embedding::{
    pad_denoise_time_embedding_step_params, pad_time_table_vec,
};

#[derive(Clone, Copy, Debug)]
pub struct PaddedDenoiseShape {
    pub n_real: usize,
    pub te_real: usize,
    pub in_real: usize,
    pub h_real: usize,
}

impl PaddedDenoiseShape {
    pub fn new(
        n_real: usize,
        te_real: usize,
        in_real: usize,
        h_real: usize,
        n_max: usize,
        te_max: usize,
        in_max: usize,
        h_max: usize,
    ) -> Self {
        assert_eq!(
            in_real,
            n_real + te_real,
            "in_real must equal n_real + te_real"
        );
        assert_eq!(in_max, n_max + te_max, "in_max must equal n_max + te_max");
        assert!(n_real <= n_max, "n_real must be <= n_max");
        assert!(te_real <= te_max, "te_real must be <= te_max");
        assert!(in_real <= in_max, "in_real must be <= in_max");
        assert!(h_real <= h_max, "h_real must be <= h_max");
        Self {
            n_real,
            te_real,
            in_real,
            h_real,
        }
    }
}

pub fn build_padded_denoise_time_embedding_placeholder_circuit<
    const N_MAX: usize,
    const TE_MAX: usize,
    const IN_MAX: usize,
    const H_MAX: usize,
>(
    total_iters: usize,
    num_iters_per_step: usize,
    config: FixedPointConfig,
    time_table_values: Vec<[i64; TE_MAX]>,
    shape: PaddedDenoiseShape,
) -> PublicFixedPointDenoiseTimeEmbeddingCircuit<G1, N_MAX, TE_MAX, IN_MAX, H_MAX> {
    PublicFixedPointDenoiseTimeEmbeddingCircuit {
        num_iters_per_step,
        total_iters,
        clipped_relu_table: config.clipped_relu_table(),
        config,
        time_table_values,
        n_real: shape.n_real,
        te_real: shape.te_real,
        in_real: shape.in_real,
        h_real: shape.h_real,
        bind_public_output: false,
        commit_params: false,
        update_mode: DenoiseUpdateMode::DoubleFloor,
        param_hash_witnesses: Vec::new(),
        seq: vec![
            FixedDenoiseTimeEmbeddingIteration {
                t_int: 0,
                time_emb_int: [0; TE_MAX],
                x_i: [F1::ZERO; N_MAX],
                x_i_int: [0; N_MAX],
                mlp_input: [F1::ZERO; IN_MAX],
                mlp_input_int: [0; IN_MAX],
                hidden_raw: [F1::ZERO; H_MAX],
                hidden_quotient: [F1::ZERO; H_MAX],
                hidden_remainder: [F1::ZERO; H_MAX],
                hidden_affine: [F1::ZERO; H_MAX],
                hidden_act: [F1::ZERO; H_MAX],
                epsilon: [F1::ZERO; N_MAX],
                hidden_raw_int: [0; H_MAX],
                hidden_quotient_int: [0; H_MAX],
                hidden_remainder_int: [0; H_MAX],
                hidden_affine_int: [0; H_MAX],
                hidden_act_int: [0; H_MAX],
                epsilon_int: [0; N_MAX],
                output_raw: [F1::ZERO; N_MAX],
                output_quotient: [F1::ZERO; N_MAX],
                output_remainder: [F1::ZERO; N_MAX],
                output_raw_int: [0; N_MAX],
                output_quotient_int: [0; N_MAX],
                output_remainder_int: [0; N_MAX],
                alpha_x: [F1::ZERO; N_MAX],
                beta_epsilon: [F1::ZERO; N_MAX],
                x_i_plus_1: [F1::ZERO; N_MAX],
                alpha_x_int: [0; N_MAX],
                beta_epsilon_int: [0; N_MAX],
                x_i_plus_1_int: [0; N_MAX],
                alpha_mul_raw_int: [0; N_MAX],
                alpha_remainder_int: [0; N_MAX],
                beta_mul_raw_int: [0; N_MAX],
                beta_remainder_int: [0; N_MAX],
                update_witness: DenoiseUpdateWitness::zero_double_floor(),
            };
            num_iters_per_step
        ],
    }
}

pub fn build_padded_denoise_time_embedding_step_circuits<
    const N_MAX: usize,
    const TE_MAX: usize,
    const IN_MAX: usize,
    const H_MAX: usize,
>(
    trace: &[FixedDenoiseTimeEmbeddingIteration<F1, N_MAX, TE_MAX, IN_MAX, H_MAX>],
    num_steps: usize,
    num_iters_per_step: usize,
    total_iters: usize,
    config: FixedPointConfig,
    time_table_values: Vec<[i64; TE_MAX]>,
    shape: PaddedDenoiseShape,
) -> Vec<PublicFixedPointDenoiseTimeEmbeddingCircuit<G1, N_MAX, TE_MAX, IN_MAX, H_MAX>> {
    assert_eq!(trace.len(), num_steps * num_iters_per_step);
    let table = config.clipped_relu_table();
    (0..num_steps)
        .map(|i| PublicFixedPointDenoiseTimeEmbeddingCircuit {
            num_iters_per_step,
            total_iters,
            clipped_relu_table: table.clone(),
            config: config.clone(),
            time_table_values: time_table_values.clone(),
            n_real: shape.n_real,
            te_real: shape.te_real,
            in_real: shape.in_real,
            h_real: shape.h_real,
            bind_public_output: false,
            commit_params: false,
            update_mode: DenoiseUpdateMode::DoubleFloor,
            param_hash_witnesses: Vec::new(),
            seq: (0..num_iters_per_step)
                .map(|j| trace[i * num_iters_per_step + j].clone())
                .collect(),
        })
        .collect()
}

pub fn build_z0_with_public_output<
    F: ff::PrimeField,
    const N_MAX: usize,
    const TE_MAX: usize,
    const IN_MAX: usize,
    const H_MAX: usize,
>(
    public_params: &FixedDenoiseTimeEmbeddingPublicParams<N_MAX, TE_MAX, IN_MAX, H_MAX>,
    x0: [i64; N_MAX],
    expected_output: [i64; N_MAX],
    shape: PaddedDenoiseShape,
) -> Vec<F> {
    for j in shape.n_real..N_MAX {
        assert!(
            expected_output[j] == 0,
            "expected output padding entry at index {j} must be zero, got {}",
            expected_output[j]
        );
    }

    let mut z0 = Vec::new();
    z0.extend(x0.into_iter().map(field_from_i64::<F>));
    z0.extend(expected_output.into_iter().map(field_from_i64::<F>));
    z0.push(F::ZERO);
    for step in &public_params.params_seq {
        z0.extend(step.flatten_field::<F>());
    }
    for row in &public_params.time_table {
        for &value in row {
            z0.push(field_from_i64::<F>(value));
        }
    }
    z0
}

pub fn generate_padded_denoise_trace_with_expected_output<
    F: ff::PrimeField + Copy,
    const N_MAX: usize,
    const TE_MAX: usize,
    const IN_MAX: usize,
    const H_MAX: usize,
>(
    public_params: &FixedDenoiseTimeEmbeddingPublicParams<N_MAX, TE_MAX, IN_MAX, H_MAX>,
    x0: [i64; N_MAX],
    expected_output: [i64; N_MAX],
    shape: PaddedDenoiseShape,
) -> (
    Vec<F>,
    Vec<FixedDenoiseTimeEmbeddingIteration<F, N_MAX, TE_MAX, IN_MAX, H_MAX>>,
) {
    let (_old_z0, trace) =
        generate_fixed_point_denoise_time_embedding_trace::<F, N_MAX, TE_MAX, IN_MAX, H_MAX>(
            public_params,
            x0,
        );
    let z0 = build_z0_with_public_output(public_params, x0, expected_output, shape);
    (z0, trace)
}

pub fn generate_padded_denoise_trace_with_computed_output<
    F: ff::PrimeField + Copy,
    const N_MAX: usize,
    const TE_MAX: usize,
    const IN_MAX: usize,
    const H_MAX: usize,
>(
    public_params: &FixedDenoiseTimeEmbeddingPublicParams<N_MAX, TE_MAX, IN_MAX, H_MAX>,
    x0: [i64; N_MAX],
    shape: PaddedDenoiseShape,
) -> (
    Vec<F>,
    Vec<FixedDenoiseTimeEmbeddingIteration<F, N_MAX, TE_MAX, IN_MAX, H_MAX>>,
    [i64; N_MAX],
) {
    let (_old_z0, trace) =
        generate_fixed_point_denoise_time_embedding_trace::<F, N_MAX, TE_MAX, IN_MAX, H_MAX>(
            public_params,
            x0,
        );
    let expected_output = trace.last().map(|it| it.x_i_plus_1_int).unwrap_or(x0);
    let z0 = build_z0_with_public_output(public_params, x0, expected_output, shape);
    (z0, trace, expected_output)
}

pub fn build_padded_output_denoise_time_embedding_placeholder_circuit<
    const N_MAX: usize,
    const TE_MAX: usize,
    const IN_MAX: usize,
    const H_MAX: usize,
>(
    total_iters: usize,
    num_iters_per_step: usize,
    config: FixedPointConfig,
    time_table_values: Vec<[i64; TE_MAX]>,
    shape: PaddedDenoiseShape,
) -> PublicFixedPointDenoiseTimeEmbeddingCircuit<G1, N_MAX, TE_MAX, IN_MAX, H_MAX> {
    let mut circuit = build_padded_denoise_time_embedding_placeholder_circuit(
        total_iters,
        num_iters_per_step,
        config,
        time_table_values,
        shape,
    );
    circuit.bind_public_output = true;
    circuit
}

pub fn build_padded_output_denoise_time_embedding_step_circuits<
    const N_MAX: usize,
    const TE_MAX: usize,
    const IN_MAX: usize,
    const H_MAX: usize,
>(
    trace: &[FixedDenoiseTimeEmbeddingIteration<F1, N_MAX, TE_MAX, IN_MAX, H_MAX>],
    num_steps: usize,
    num_iters_per_step: usize,
    total_iters: usize,
    config: FixedPointConfig,
    time_table_values: Vec<[i64; TE_MAX]>,
    shape: PaddedDenoiseShape,
) -> Vec<PublicFixedPointDenoiseTimeEmbeddingCircuit<G1, N_MAX, TE_MAX, IN_MAX, H_MAX>> {
    let mut circuits = build_padded_denoise_time_embedding_step_circuits(
        trace,
        num_steps,
        num_iters_per_step,
        total_iters,
        config,
        time_table_values,
        shape,
    );
    for circuit in &mut circuits {
        circuit.bind_public_output = true;
    }
    circuits
}

pub fn compute_padded_denoise_time_embedding_param_hash_witnesses<
    F: ff::PrimeField,
    const N_MAX: usize,
    const TE_MAX: usize,
    const IN_MAX: usize,
    const H_MAX: usize,
>(
    public_params: &FixedDenoiseTimeEmbeddingPublicParams<N_MAX, TE_MAX, IN_MAX, H_MAX>,
) -> (Vec<Vec<F>>, F) {
    let mut h = F::ZERO;
    let mut all = Vec::with_capacity(public_params.total_iters());
    for step in &public_params.params_seq {
        let (prefixes, h_next) =
            toy_hash_block_prefixes_i64_from_field(&step.flatten_i64(), TOY_HASH_BASE_U64, h);
        all.push(prefixes);
        h = h_next;
    }
    (all, h)
}

pub fn build_z0_with_commitment<
    F: ff::PrimeField,
    const N_MAX: usize,
    const TE_MAX: usize,
    const IN_MAX: usize,
    const H_MAX: usize,
>(
    public_params: &FixedDenoiseTimeEmbeddingPublicParams<N_MAX, TE_MAX, IN_MAX, H_MAX>,
    x0: [i64; N_MAX],
    expected_output: [i64; N_MAX],
    final_commitment: F,
    shape: PaddedDenoiseShape,
) -> Vec<F> {
    for j in shape.n_real..N_MAX {
        assert!(
            expected_output[j] == 0,
            "expected output padding entry at index {j} must be zero, got {}",
            expected_output[j]
        );
    }
    let layout = CommittedDenoiseStateLayout::new(
        N_MAX,
        true,
        public_params.total_iters(),
        public_params.block_len(),
        TE_MAX,
    );
    let mut z0 = Vec::with_capacity(layout.state_len());
    z0.extend(x0.into_iter().map(field_from_i64::<F>));
    z0.extend(expected_output.into_iter().map(field_from_i64::<F>));
    z0.push(F::ZERO);
    z0.push(final_commitment);
    z0.push(F::ZERO);
    for step in &public_params.params_seq {
        z0.extend(step.flatten_field::<F>());
    }
    for row in &public_params.time_table {
        for &value in row {
            z0.push(field_from_i64::<F>(value));
        }
    }
    layout.assert_state_len(z0.len());
    z0
}

pub fn generate_padded_denoise_trace_with_commitment<
    F: ff::PrimeField + Copy,
    const N_MAX: usize,
    const TE_MAX: usize,
    const IN_MAX: usize,
    const H_MAX: usize,
>(
    public_params: &FixedDenoiseTimeEmbeddingPublicParams<N_MAX, TE_MAX, IN_MAX, H_MAX>,
    x0: [i64; N_MAX],
    shape: PaddedDenoiseShape,
) -> (
    Vec<F>,
    Vec<FixedDenoiseTimeEmbeddingIteration<F, N_MAX, TE_MAX, IN_MAX, H_MAX>>,
    Vec<Vec<F>>,
    [i64; N_MAX],
    F,
) {
    let (_old_z0, trace) =
        generate_fixed_point_denoise_time_embedding_trace::<F, N_MAX, TE_MAX, IN_MAX, H_MAX>(
            public_params,
            x0,
        );
    let expected_output = trace.last().map(|it| it.x_i_plus_1_int).unwrap_or(x0);
    let (hash_witnesses, commitment) =
        compute_padded_denoise_time_embedding_param_hash_witnesses(public_params);
    let z0 = build_z0_with_commitment(public_params, x0, expected_output, commitment, shape);
    (z0, trace, hash_witnesses, expected_output, commitment)
}

pub fn build_padded_commitment_denoise_time_embedding_placeholder_circuit<
    const N_MAX: usize,
    const TE_MAX: usize,
    const IN_MAX: usize,
    const H_MAX: usize,
>(
    total_iters: usize,
    num_iters_per_step: usize,
    config: FixedPointConfig,
    time_table_values: Vec<[i64; TE_MAX]>,
    shape: PaddedDenoiseShape,
) -> PublicFixedPointDenoiseTimeEmbeddingCircuit<G1, N_MAX, TE_MAX, IN_MAX, H_MAX> {
    let mut circuit = build_padded_output_denoise_time_embedding_placeholder_circuit(
        total_iters,
        num_iters_per_step,
        config,
        time_table_values,
        shape,
    );
    circuit.commit_params = true;
    circuit.param_hash_witnesses =
        vec![
            vec![
                F1::ZERO;
                FixedDenoiseTimeEmbeddingStepParams::<N_MAX, TE_MAX, IN_MAX, H_MAX>::block_len()
            ];
            num_iters_per_step
        ];
    circuit
}

pub fn build_padded_commitment_denoise_time_embedding_step_circuits<
    const N_MAX: usize,
    const TE_MAX: usize,
    const IN_MAX: usize,
    const H_MAX: usize,
>(
    trace: &[FixedDenoiseTimeEmbeddingIteration<F1, N_MAX, TE_MAX, IN_MAX, H_MAX>],
    hash_witnesses: &[Vec<F1>],
    num_steps: usize,
    num_iters_per_step: usize,
    total_iters: usize,
    config: FixedPointConfig,
    time_table_values: Vec<[i64; TE_MAX]>,
    shape: PaddedDenoiseShape,
) -> Vec<PublicFixedPointDenoiseTimeEmbeddingCircuit<G1, N_MAX, TE_MAX, IN_MAX, H_MAX>> {
    assert_eq!(hash_witnesses.len(), num_steps * num_iters_per_step);
    let mut circuits = build_padded_output_denoise_time_embedding_step_circuits(
        trace,
        num_steps,
        num_iters_per_step,
        total_iters,
        config,
        time_table_values,
        shape,
    );
    for (i, circuit) in circuits.iter_mut().enumerate() {
        circuit.commit_params = true;
        circuit.param_hash_witnesses = (0..num_iters_per_step)
            .map(|j| hash_witnesses[i * num_iters_per_step + j].clone())
            .collect();
    }
    circuits
}
