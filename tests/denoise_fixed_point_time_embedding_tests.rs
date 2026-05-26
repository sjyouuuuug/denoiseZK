use denoise::{
    denoise_fixed_point_time_embedding::{
        build_fixed_point_denoise_time_embedding_placeholder_circuit,
        build_fixed_point_denoise_time_embedding_step_circuits,
        generate_fixed_point_denoise_time_embedding_trace, generate_simple_time_table,
        run_fixed_point_denoise_time_embedding_recursive,
        setup_fixed_point_denoise_time_embedding_public_params,
        verify_fixed_point_denoise_time_embedding_recursive, FixedDenoiseTimeEmbeddingPublicParams,
        FixedDenoiseTimeEmbeddingStepParams,
    },
    fixed_point::FixedPointConfig,
    nova_ivc::E1,
};
use nova_snark::traits::Engine;

type F = <E1 as Engine>::Scalar;

#[test]
fn time_table_generation_has_expected_shape() {
    let table = generate_simple_time_table::<2>(4, 16);
    assert_eq!(table.len(), 4);
    assert_eq!(table[0], [0, 16]);
    assert_eq!(table[3], [16, 0]);
}

#[test]
fn time_embedding_param_block_len_and_flatten_layout() {
    const N: usize = 1;
    const TE: usize = 1;
    const IN: usize = 2;
    const H: usize = 1;
    let params =
        FixedDenoiseTimeEmbeddingStepParams::<N, TE, IN, H>::new([[1, 2]], [3], [[4]], [5], 6, 7);
    assert_eq!(
        FixedDenoiseTimeEmbeddingStepParams::<N, TE, IN, H>::block_len(),
        H * IN + H + N * H + N + 2
    );
    assert_eq!(params.flatten_i64(), vec![1, 2, 3, 4, 5, 6, 7]);
}

#[test]
fn time_embedding_lookup_trace_uses_correct_row() {
    const N: usize = 1;
    const TE: usize = 2;
    const IN: usize = 3;
    const H: usize = 1;
    let cfg = FixedPointConfig::from_real_bounds(4, -8, 8, 4);
    let table = vec![[4, 0], [2, 2], [0, 4]];
    let step = FixedDenoiseTimeEmbeddingStepParams::<N, TE, IN, H>::new(
        [[0, 0, 0]],
        [0],
        [[0]],
        [0],
        4,
        0,
    );
    let public_params = FixedDenoiseTimeEmbeddingPublicParams::new(
        vec![step.clone(), step.clone(), step],
        table.clone(),
        cfg,
    );
    let (_, trace) =
        generate_fixed_point_denoise_time_embedding_trace::<F, N, TE, IN, H>(&public_params, [4]);
    assert_eq!(trace[0].time_emb_int, table[0]);
    assert_eq!(trace[1].time_emb_int, table[1]);
    assert_eq!(trace[2].time_emb_int, table[2]);
}

#[test]
fn denoise_time_embedding_trace_matches_hand_computation() {
    const N: usize = 1;
    const TE: usize = 1;
    const IN: usize = 2;
    const H: usize = 1;
    let cfg = FixedPointConfig::from_real_bounds(4, -8, 8, 4);
    let table = vec![[2]];
    let step =
        FixedDenoiseTimeEmbeddingStepParams::<N, TE, IN, H>::new([[4, 4]], [0], [[4]], [0], 2, 2);
    let public_params = FixedDenoiseTimeEmbeddingPublicParams::new(vec![step], table, cfg);
    let (_, trace) =
        generate_fixed_point_denoise_time_embedding_trace::<F, N, TE, IN, H>(&public_params, [4]);
    let it = &trace[0];

    assert_eq!(it.t_int, 0);
    assert_eq!(it.time_emb_int, [2]);
    assert_eq!(it.mlp_input_int, [4, 2]);
    assert_eq!(it.hidden_affine_int, [6]);
    assert_eq!(it.hidden_act_int, [6]);
    assert_eq!(it.epsilon_int, [6]);
    assert_eq!(it.alpha_mul_raw_int, [8]);
    assert_eq!(it.alpha_x_int, [2]);
    assert_eq!(it.alpha_remainder_int, [0]);
    assert_eq!(it.beta_mul_raw_int, [12]);
    assert_eq!(it.beta_epsilon_int, [3]);
    assert_eq!(it.beta_remainder_int, [0]);
    assert_eq!(it.x_i_plus_1_int, [5]);
}

#[test]
fn time_index_increments_across_iterations() {
    const N: usize = 1;
    const TE: usize = 1;
    const IN: usize = 2;
    const H: usize = 1;
    let cfg = FixedPointConfig::from_real_bounds(4, -8, 8, 4);
    let table = generate_simple_time_table::<TE>(3, 4);
    let step =
        FixedDenoiseTimeEmbeddingStepParams::<N, TE, IN, H>::new([[0, 0]], [0], [[0]], [0], 4, 0);
    let public_params = FixedDenoiseTimeEmbeddingPublicParams::new(
        vec![step.clone(), step.clone(), step],
        table,
        cfg,
    );
    let (_, trace) =
        generate_fixed_point_denoise_time_embedding_trace::<F, N, TE, IN, H>(&public_params, [4]);
    assert_eq!(trace[0].t_int, 0);
    assert_eq!(trace[1].t_int, 1);
    assert_eq!(trace[2].t_int, 2);
}

#[test]
#[ignore]
fn small_recursive_denoise_time_embedding_proof_verifies() {
    const N: usize = 1;
    const TE: usize = 1;
    const IN: usize = 2;
    const H: usize = 1;
    let cfg = FixedPointConfig::from_real_bounds(4, -8, 8, 4);
    let total_iters = 1;
    let num_steps = 1;
    let num_iters_per_step = 1;
    let table = vec![[2]];
    let step =
        FixedDenoiseTimeEmbeddingStepParams::<N, TE, IN, H>::new([[4, 4]], [0], [[4]], [0], 2, 2);
    let public_params =
        FixedDenoiseTimeEmbeddingPublicParams::new(vec![step], table.clone(), cfg.clone());
    let (z0, trace) =
        generate_fixed_point_denoise_time_embedding_trace::<F, N, TE, IN, H>(&public_params, [4]);
    let placeholder = build_fixed_point_denoise_time_embedding_placeholder_circuit::<N, TE, IN, H>(
        total_iters,
        num_iters_per_step,
        cfg.clone(),
        table.clone(),
    );
    let pp = setup_fixed_point_denoise_time_embedding_public_params(&placeholder);
    let circuits = build_fixed_point_denoise_time_embedding_step_circuits(
        &trace,
        num_steps,
        num_iters_per_step,
        total_iters,
        cfg,
        table,
    );
    let recursive_snark = run_fixed_point_denoise_time_embedding_recursive(&pp, &circuits, &z0);
    verify_fixed_point_denoise_time_embedding_recursive(&recursive_snark, &pp, num_steps, &z0);
}
