use denoise::{
    clipped_relu::field_from_i64,
    denoise_fixed_point::{
        build_fixed_point_denoise_placeholder_circuit, build_fixed_point_denoise_step_circuits,
        generate_fixed_point_denoise_trace, run_fixed_point_denoise_recursive,
        setup_fixed_point_denoise_public_params, verify_fixed_point_denoise_recursive,
        FixedDenoisePublicParams, FixedDenoiseStepParams,
    },
    fixed_point::{floor_div, FixedPointConfig},
    mlp_fixed_point_clipped_relu_lookup::FixedMlpClippedReluStepParams,
    nova_ivc::E1,
};
use nova_snark::traits::Engine;

type F = <E1 as Engine>::Scalar;

#[test]
fn denoise_step_param_block_len_and_flatten_layout_are_correct() {
    const N: usize = 2;
    const H: usize = 2;
    let mlp = FixedMlpClippedReluStepParams::<N, H>::new(
        [[1, 2], [3, 4]],
        [5, 6],
        [[7, 8], [9, 10]],
        [11, 12],
    );
    let params = FixedDenoiseStepParams::new(mlp, 13, 14);

    assert_eq!(
        FixedDenoiseStepParams::<N, H>::block_len(),
        H * N + H + N * H + N + 2
    );
    assert_eq!(
        params.flatten_i64(),
        vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    );
}

#[test]
fn generate_denoise_trace_matches_hand_computation() {
    const N: usize = 1;
    const H: usize = 1;
    let cfg = FixedPointConfig::from_real_bounds(4, -8, 8, 4);
    let step = FixedDenoiseStepParams::<N, H>::new(
        FixedMlpClippedReluStepParams::new([[4]], [1], [[4]], [0]),
        2,
        2,
    );
    let public_params = FixedDenoisePublicParams::new(vec![step], cfg);
    let (z0, trace) = generate_fixed_point_denoise_trace::<F, N, H>(&public_params, [4]);
    let it = &trace[0];

    assert_eq!(z0[0], field_from_i64::<F>(4));
    assert_eq!(it.hidden_affine_int, [5]);
    assert_eq!(it.hidden_act_int, [5]);
    assert_eq!(it.epsilon_int, [5]);
    assert_eq!(it.alpha_mul_raw_int, [8]);
    assert_eq!(it.alpha_x_int, [2]);
    assert_eq!(it.alpha_remainder_int, [0]);
    assert_eq!(it.beta_mul_raw_int, [10]);
    assert_eq!(it.beta_epsilon_int, [2]);
    assert_eq!(it.beta_remainder_int, [2]);
    assert_eq!(it.x_i_plus_1_int, [4]);
}

#[test]
fn floor_schedule_update_handles_negative_values() {
    assert_eq!(floor_div(-7, 4), -2);

    const N: usize = 1;
    const H: usize = 1;
    let cfg = FixedPointConfig::from_real_bounds(4, -8, 8, 4);
    let step = FixedDenoiseStepParams::<N, H>::new(
        FixedMlpClippedReluStepParams::new([[0]], [0], [[0]], [0]),
        1,
        0,
    );
    let public_params = FixedDenoisePublicParams::new(vec![step], cfg);
    let (_, trace) = generate_fixed_point_denoise_trace::<F, N, H>(&public_params, [-7]);
    let it = &trace[0];
    assert_eq!(it.alpha_mul_raw_int, [-7]);
    assert_eq!(it.alpha_x_int, [-2]);
    assert_eq!(it.alpha_remainder_int, [1]);
}

#[test]
#[should_panic(expected = "outside clipped ReLU table range")]
fn denoise_trace_rejects_relu_lookup_overflow() {
    const N: usize = 1;
    const H: usize = 1;
    let cfg = FixedPointConfig::from_real_bounds(4, -1, 1, 1);
    let step = FixedDenoiseStepParams::<N, H>::new(
        FixedMlpClippedReluStepParams::new([[4]], [8], [[0]], [0]),
        4,
        0,
    );
    let public_params = FixedDenoisePublicParams::new(vec![step], cfg);
    let _ = generate_fixed_point_denoise_trace::<F, N, H>(&public_params, [4]);
}

#[test]
#[should_panic(expected = "alpha quotient")]
fn denoise_trace_rejects_schedule_range_overflow() {
    const N: usize = 1;
    const H: usize = 1;
    let cfg = FixedPointConfig::from_real_bounds(4, -1, 1, 1).with_integer_ranges(-8, 7, -128, 127);
    let step = FixedDenoiseStepParams::<N, H>::new(
        FixedMlpClippedReluStepParams::new([[0]], [0], [[0]], [0]),
        8,
        0,
    );
    let public_params = FixedDenoisePublicParams::new(vec![step], cfg);
    let _ = generate_fixed_point_denoise_trace::<F, N, H>(&public_params, [8]);
}

#[test]
fn build_denoise_step_circuits_chunks_trace() {
    const N: usize = 2;
    const H: usize = 2;
    let cfg = FixedPointConfig::default_scale16();
    let step = FixedDenoiseStepParams::<N, H>::from_f64(
        [[0.5, 0.0], [0.0, 0.5]],
        [0.0, 0.0],
        [[0.5, 0.0], [0.0, 0.5]],
        [0.0, 0.0],
        0.875,
        0.125,
        cfg.scale,
    );
    let public_params = FixedDenoisePublicParams::new(
        vec![step.clone(), step.clone(), step.clone(), step],
        cfg.clone(),
    );
    let (_z0, trace) = generate_fixed_point_denoise_trace::<F, N, H>(&public_params, [16, 16]);
    let circuits = build_fixed_point_denoise_step_circuits(&trace, 2, 2, 4, cfg.clone());
    assert_eq!(circuits.len(), 2);
    assert_eq!(circuits[0].seq.len(), 2);
    assert_eq!(circuits[0].total_iters, 4);
    assert_eq!(circuits[0].config.scale, cfg.scale);
}

#[test]
#[ignore]
fn small_recursive_denoise_fixed_point_proof_verifies() {
    const N: usize = 1;
    const H: usize = 1;
    let cfg = FixedPointConfig::from_real_bounds(4, -8, 8, 4);
    let num_steps = 1;
    let num_iters_per_step = 1;
    let total_iters = 1;
    let step = FixedDenoiseStepParams::<N, H>::new(
        FixedMlpClippedReluStepParams::new([[4]], [0], [[4]], [0]),
        3,
        1,
    );
    let public_params = FixedDenoisePublicParams::new(vec![step], cfg.clone());
    let (z0, trace) = generate_fixed_point_denoise_trace::<F, N, H>(&public_params, [4]);
    let placeholder =
        build_fixed_point_denoise_placeholder_circuit(total_iters, num_iters_per_step, cfg.clone());
    let pp = setup_fixed_point_denoise_public_params(&placeholder);
    let circuits = build_fixed_point_denoise_step_circuits(
        &trace,
        num_steps,
        num_iters_per_step,
        total_iters,
        cfg,
    );
    let recursive_snark = run_fixed_point_denoise_recursive(&pp, &circuits, &z0);
    verify_fixed_point_denoise_recursive(&recursive_snark, &pp, num_steps, &z0);
}
