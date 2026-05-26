use denoise::{
    affine::fixed_point::{apply_affine_fixed_point, apply_affine_fixed_point_with_witness},
    clipped_relu::{field_from_i64, fixed_point::fixed_point_clipped_relu},
    fixed_point::{
        decode_i64_to_f64, encode_f64_round, floor_div, rescale_with_remainder, FixedPointConfig,
    },
    mlp_fixed_point_clipped_relu_lookup::{
        build_fixed_point_mlp_placeholder_circuit, build_fixed_point_mlp_step_circuits,
        generate_fixed_point_mlp_trace, run_fixed_point_mlp_recursive,
        setup_fixed_point_mlp_public_params, verify_fixed_point_mlp_recursive,
        FixedMlpClippedReluPublicParams, FixedMlpClippedReluStepParams,
    },
    nova_ivc::E1,
};
use nova_snark::traits::Engine;

type F = <E1 as Engine>::Scalar;

#[test]
fn encode_round_and_decode_work() {
    let scale = 16;
    assert_eq!(encode_f64_round(1.25, scale), 20);
    assert_eq!(encode_f64_round(-0.5, scale), -8);
    assert_eq!(encode_f64_round(0.125, scale), 2);
    assert!((decode_i64_to_f64(20, scale) - 1.25).abs() < 1e-9);
}

#[test]
fn floor_division_is_mathematical_floor() {
    assert_eq!(floor_div(7, 4), 1);
    assert_eq!(floor_div(8, 4), 2);
    assert_eq!(floor_div(-7, 4), -2);
    assert_eq!(floor_div(-8, 4), -2);
    assert_eq!(floor_div(-1, 16), -1);
}

#[test]
fn rescale_returns_canonical_remainder() {
    assert_eq!(rescale_with_remainder(35, 16), (2, 3));
    assert_eq!(rescale_with_remainder(-7, 4), (-2, 1));
    assert_eq!(rescale_with_remainder(-33, 16), (-3, 15));
}

#[test]
fn fixed_point_config_scales_relu_bounds_and_clip() {
    let cfg = FixedPointConfig::from_real_bounds(16, -4, 4, 2);
    assert_eq!(cfg.scale, 16);
    assert_eq!(cfg.relu_min, -64);
    assert_eq!(cfg.relu_max, 64);
    assert_eq!(cfg.clip_max, 32);
    assert_eq!(cfg.quotient_min, -256);
    assert_eq!(cfg.quotient_max, 255);
    assert_eq!(cfg.value_min, -128);
    assert_eq!(cfg.value_max, 127);
    let table = cfg.clipped_relu_table();
    assert_eq!(table.min, -64);
    assert_eq!(table.max, 64);
    assert_eq!(table.clip_max, 32);
}

#[test]
fn fixed_point_clipped_relu_uses_scaled_clip_bound() {
    let cfg = FixedPointConfig::from_real_bounds(16, -4, 4, 2);
    assert_eq!(fixed_point_clipped_relu(-8, &cfg), 0);
    assert_eq!(fixed_point_clipped_relu(0, &cfg), 0);
    assert_eq!(fixed_point_clipped_relu(16, &cfg), 16);
    assert_eq!(fixed_point_clipped_relu(48, &cfg), 32);
}

#[test]
fn fixed_point_affine_uses_floor_rescale_then_bias() {
    const OUT: usize = 2;
    const IN: usize = 2;
    let scale = 16;
    // Represents W = [[0.5, -0.25], [0.25, 0.5]], b = [0.25, -0.125]
    let w = [[8, -4], [4, 8]];
    let b = [4, -2];
    let x = [16, -8]; // [1.0, -0.5]
    let y = apply_affine_fixed_point::<OUT, IN>(&w, &b, &x, scale);
    // row0 raw = 8*16 + (-4)*(-8) = 160, floor(160/16)=10, +4 = 14
    // row1 raw = 4*16 + 8*(-8) = 0, floor(0/16)=0, -2 = -2
    assert_eq!(y, [14, -2]);

    let (raw, q, rem, y2) = apply_affine_fixed_point_with_witness::<OUT, IN>(&w, &b, &x, scale);
    assert_eq!(raw, [160, 0]);
    assert_eq!(q, [10, 0]);
    assert_eq!(rem, [0, 0]);
    assert_eq!(y2, y);
}

#[test]
fn fixed_point_step_param_from_f64_and_flatten_layout_are_correct() {
    const N: usize = 2;
    const H: usize = 2;
    let scale = 16;
    let params = FixedMlpClippedReluStepParams::<N, H>::from_f64(
        [[0.5, -0.25], [1.0, 0.0]],
        [0.25, -0.5],
        [[0.5, 0.25], [-0.25, 1.0]],
        [0.0, 0.125],
        scale,
    );
    assert_eq!(params.w1, [[8, -4], [16, 0]]);
    assert_eq!(params.b1, [4, -8]);
    assert_eq!(params.w2, [[8, 4], [-4, 16]]);
    assert_eq!(params.b2, [0, 2]);
    assert_eq!(
        FixedMlpClippedReluStepParams::<N, H>::block_len(),
        H * N + H + N * H + N
    );
    assert_eq!(
        params.flatten_i64(),
        vec![8, -4, 16, 0, 4, -8, 8, 4, -4, 16, 0, 2]
    );
}

#[test]
fn fixed_point_mlp_trace_matches_hand_computation() {
    const N: usize = 2;
    const H: usize = 2;
    let cfg = FixedPointConfig::from_real_bounds(16, -4, 4, 2);
    let scale = cfg.scale;
    let step = FixedMlpClippedReluStepParams::<N, H>::from_f64(
        [[0.5, -0.25], [0.25, 0.5]],
        [0.25, 0.0],
        [[0.5, 0.25], [-0.25, 0.5]],
        [0.0, 0.125],
        scale,
    );
    let public_params = FixedMlpClippedReluPublicParams::new(vec![step], cfg.clone());
    let x0 = [16, -8]; // [1.0, -0.5]
    let (z0, trace) = generate_fixed_point_mlp_trace::<F, N, H>(&public_params, x0);

    assert_eq!(z0[0], field_from_i64::<F>(16));
    assert_eq!(z0[1], field_from_i64::<F>(-8));
    assert_eq!(trace.len(), 1);
    let it = &trace[0];

    // hidden raw:
    // row0 = 8*16 + (-4)*(-8) = 160; q=10, rem=0; + b=4 => 14; relu=14
    // row1 = 4*16 + 8*(-8) = 0; q=0, rem=0; + b=0 => 0; relu=0
    assert_eq!(it.hidden_raw_int, [160, 0]);
    assert_eq!(it.hidden_quotient_int, [10, 0]);
    assert_eq!(it.hidden_remainder_int, [0, 0]);
    assert_eq!(it.hidden_affine_int, [14, 0]);
    assert_eq!(it.hidden_act_int, [14, 0]);

    // output raw:
    // row0 = 8*14 + 4*0 = 112; q=7, rem=0; + b=0 => 7
    // row1 = -4*14 + 8*0 = -56; floor(-56/16)=-4, rem=8; + b=2 => -2
    assert_eq!(it.output_raw_int, [112, -56]);
    assert_eq!(it.output_quotient_int, [7, -4]);
    assert_eq!(it.output_remainder_int, [0, 8]);
    assert_eq!(it.x_i_plus_1_int, [7, -2]);
}

#[test]
#[should_panic(expected = "outside clipped ReLU table range")]
fn fixed_point_trace_rejects_hidden_affine_outside_lookup_range() {
    const N: usize = 2;
    const H: usize = 1;
    let cfg = FixedPointConfig::from_real_bounds(16, -1, 1, 1);
    let step = FixedMlpClippedReluStepParams::<N, H>::new([[16, 0]], [32], [[16], [0]], [0, 0]);
    let public_params = FixedMlpClippedReluPublicParams::new(vec![step], cfg);
    let _ = generate_fixed_point_mlp_trace::<F, N, H>(&public_params, [16, 0]);
}

#[test]
fn fixed_point_build_step_circuits_chunks_trace() {
    const N: usize = 2;
    const H: usize = 2;
    let cfg = FixedPointConfig::default_scale16();
    let step = FixedMlpClippedReluStepParams::<N, H>::from_f64(
        [[0.5, 0.0], [0.0, 0.5]],
        [0.0, 0.0],
        [[0.5, 0.0], [0.0, 0.5]],
        [0.0, 0.0],
        cfg.scale,
    );
    let public_params = FixedMlpClippedReluPublicParams::new(
        vec![step.clone(), step.clone(), step.clone(), step],
        cfg.clone(),
    );
    let (_z0, trace) = generate_fixed_point_mlp_trace::<F, N, H>(&public_params, [16, 16]);
    let circuits = build_fixed_point_mlp_step_circuits(&trace, 2, 2, 4, cfg.clone());
    assert_eq!(circuits.len(), 2);
    assert_eq!(circuits[0].seq.len(), 2);
    assert_eq!(circuits[0].total_iters, 4);
    assert_eq!(circuits[0].config.scale, cfg.scale);
}

// Heavier end-to-end proof test. Run explicitly with:
// cargo test --release fixed_point_small_recursive_mlp_proof_verifies -- --ignored --nocapture
#[test]
#[ignore]
fn fixed_point_small_recursive_mlp_proof_verifies() {
    const N: usize = 2;
    const H: usize = 2;
    let cfg = FixedPointConfig::from_real_bounds(16, -4, 4, 2);
    let num_steps = 1;
    let num_iters_per_step = 1;
    let total_iters = 1;
    let step = FixedMlpClippedReluStepParams::<N, H>::from_f64(
        [[0.5, -0.25], [0.25, 0.5]],
        [0.0, 0.0],
        [[0.5, 0.25], [0.0, 0.5]],
        [0.0, 0.0],
        cfg.scale,
    );
    let public_params = FixedMlpClippedReluPublicParams::new(vec![step], cfg.clone());
    let (z0, trace) = generate_fixed_point_mlp_trace::<F, N, H>(&public_params, [16, -8]);

    let placeholder =
        build_fixed_point_mlp_placeholder_circuit(total_iters, num_iters_per_step, cfg.clone());
    let pp = setup_fixed_point_mlp_public_params(&placeholder);
    let circuits = build_fixed_point_mlp_step_circuits(
        &trace,
        num_steps,
        num_iters_per_step,
        total_iters,
        cfg,
    );
    let recursive_snark = run_fixed_point_mlp_recursive(&pp, &circuits, &z0);
    verify_fixed_point_mlp_recursive(&recursive_snark, &pp, num_steps, &z0);
}
