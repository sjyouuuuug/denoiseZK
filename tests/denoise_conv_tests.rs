use denoise::{
    denoise_fixed_point_conv::{
        build_fixed_point_denoise_conv_placeholder_circuit,
        build_fixed_point_denoise_conv_placeholder_circuit_with_shape,
        build_fixed_point_denoise_conv_step_circuits,
        build_fixed_point_denoise_conv_step_circuits_with_shape,
        generate_fixed_point_denoise_conv_trace,
        generate_fixed_point_denoise_conv_trace_with_computed_output,
        generate_fixed_point_denoise_conv_trace_with_expected_output,
        run_fixed_point_denoise_conv_recursive, setup_fixed_point_denoise_conv_public_params,
        verify_fixed_point_denoise_conv_recursive, FixedDenoiseConvPublicParams,
        FixedDenoiseConvStepParams,
    },
    denoise_fixed_point_time_embedding::generate_simple_time_table,
    fixed_point::FixedPointConfig,
    layers::conv2d::{Conv2dPadding, Conv2dRealShape},
    nova_ivc::E1,
    public_state::PublicStateLayout,
};
use nova_snark::{traits::circuit::StepCircuit, traits::Engine};

type F = <E1 as Engine>::Scalar;

#[test]
fn denoise_conv_tests() {}

#[test]
fn denoise_conv_param_block_len_and_flatten_layout() {
    const TE: usize = 2;
    const KH: usize = 2;
    const KW: usize = 2;
    let params =
        FixedDenoiseConvStepParams::<TE, KH, KW>::new([[1, 2], [3, 4]], 5, [6, 7], 8, 9, 10);
    assert_eq!(
        FixedDenoiseConvStepParams::<TE, KH, KW>::block_len(),
        KH * KW + 1 + TE + 1 + 2
    );
    assert_eq!(params.flatten_i64(), vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
}

#[test]
fn conv_real_shape_n_real_is_not_flat_prefix() {
    let shape = Conv2dRealShape::new(2, 2, 1, 1, 2, 2);
    let real_indices: Vec<usize> = (0..4)
        .flat_map(|row| (0..4).map(move |col| (row, col)))
        .filter(|&(row, col)| row < shape.ih_real && col < shape.iw_real)
        .map(|(row, col)| row * 4 + col)
        .collect();
    assert_eq!(shape.n_real(), 4);
    assert_eq!(real_indices, vec![0, 1, 4, 5]);
    assert_ne!(real_indices, vec![0, 1, 2, 3]);
}

#[test]
fn time_bias_matches_hand_computation() {
    const N: usize = 1;
    const IH: usize = 1;
    const IW: usize = 1;
    const TE: usize = 2;
    const KH: usize = 1;
    const KW: usize = 1;
    const OH: usize = 1;
    const OW: usize = 1;
    let cfg = FixedPointConfig::from_real_bounds(4, -8, 8, 4);
    let step = FixedDenoiseConvStepParams::<TE, KH, KW>::new([[0]], 0, [4, 2], 1, 4, 0);
    let public_params = FixedDenoiseConvPublicParams::new(
        vec![step],
        vec![[2, 6]],
        cfg,
        Conv2dPadding::default(),
        Conv2dRealShape::full::<IH, IW, KH, KW, OH, OW>(),
        TE,
    );

    let (_, trace) = generate_fixed_point_denoise_conv_trace::<F, N, IH, IW, TE, KH, KW, OH, OW>(
        &public_params,
        [4],
    );
    let it = &trace[0];
    assert_eq!(it.time_raw_int, 20);
    assert_eq!(it.time_quotient_int, 5);
    assert_eq!(it.time_remainder_int, 0);
    assert_eq!(it.time_bias_int, 6);
}

#[test]
fn denoise_conv_trace_matches_hand_computation_for_2x2_image() {
    const N: usize = 4;
    const IH: usize = 2;
    const IW: usize = 2;
    const TE: usize = 1;
    const KH: usize = 1;
    const KW: usize = 1;
    const OH: usize = 2;
    const OW: usize = 2;
    let cfg = FixedPointConfig::from_real_bounds(4, -8, 8, 4);
    let step = FixedDenoiseConvStepParams::<TE, KH, KW>::new([[4]], 0, [0], 0, 2, 2);
    let public_params = FixedDenoiseConvPublicParams::new(
        vec![step],
        vec![[0]],
        cfg,
        Conv2dPadding::default(),
        Conv2dRealShape::full::<IH, IW, KH, KW, OH, OW>(),
        TE,
    );

    let (_, trace) = generate_fixed_point_denoise_conv_trace::<F, N, IH, IW, TE, KH, KW, OH, OW>(
        &public_params,
        [4, -4, 8, 0],
    );
    let it = &trace[0];
    assert_eq!(it.conv_witness.quotient, [[4, -4], [8, 0]]);
    assert_eq!(it.epsilon_int, [4, 0, 8, 0]);
    assert_eq!(it.alpha_x_int, [2, -2, 4, 0]);
    assert_eq!(it.beta_epsilon_int, [2, 0, 4, 0]);
    assert_eq!(it.x_i_plus_1_int, [4, -2, 8, 0]);
}

#[test]
fn denoise_conv_same_padding_output_shape_equals_input_shape() {
    const N: usize = 9;
    const IH: usize = 3;
    const IW: usize = 3;
    const TE: usize = 1;
    const KH: usize = 3;
    const KW: usize = 3;
    const OH: usize = 3;
    const OW: usize = 3;
    let cfg = FixedPointConfig::from_real_bounds(4, -8, 8, 4);
    let step = FixedDenoiseConvStepParams::<TE, KH, KW>::new(
        [[0, 0, 0], [0, 4, 0], [0, 0, 0]],
        0,
        [0],
        0,
        4,
        0,
    );
    let public_params = FixedDenoiseConvPublicParams::new(
        vec![step],
        vec![[0]],
        cfg,
        Conv2dPadding {
            top: 1,
            bottom: 1,
            left: 1,
            right: 1,
        },
        Conv2dRealShape::full::<IH, IW, KH, KW, OH, OW>(),
        TE,
    );

    let (_, trace) = generate_fixed_point_denoise_conv_trace::<F, N, IH, IW, TE, KH, KW, OH, OW>(
        &public_params,
        [1, 2, 3, 4, 5, 6, 7, 8, 9],
    );
    assert_eq!(trace[0].epsilon_int.len(), N);
}

#[test]
fn denoise_conv_larger_4x4_trace_runs_multiple_iterations() {
    const N: usize = 16;
    const IH: usize = 4;
    const IW: usize = 4;
    const TE: usize = 3;
    const KH: usize = 3;
    const KW: usize = 3;
    const OH: usize = 4;
    const OW: usize = 4;
    let cfg = FixedPointConfig::from_real_bounds(16, -8, 8, 4);
    let total_iters = 6;
    let time_table = generate_simple_time_table::<TE>(total_iters, cfg.scale);
    let mut params = Vec::with_capacity(total_iters);
    for t in 0..total_iters {
        params.push(FixedDenoiseConvStepParams::<TE, KH, KW>::new(
            [[0, 1, 0], [1, 2 + t as i64, 1], [0, 1, 0]],
            0,
            [1, -1, 1],
            0,
            14,
            2,
        ));
    }
    let public_params = FixedDenoiseConvPublicParams::new(
        params,
        time_table,
        cfg.clone(),
        Conv2dPadding::same_3x3(),
        Conv2dRealShape::full::<IH, IW, KH, KW, OH, OW>(),
        TE,
    );
    let x0 = [16, -8, 8, 0, -4, 12, -8, 4, 8, 0, 16, -4, 0, 4, -12, 8];

    let (z0, trace) = generate_fixed_point_denoise_conv_trace::<F, N, IH, IW, TE, KH, KW, OH, OW>(
        &public_params,
        x0,
    );

    assert_eq!(trace.len(), total_iters);
    assert_eq!(
        z0.len(),
        N + 1
            + total_iters * FixedDenoiseConvStepParams::<TE, KH, KW>::block_len()
            + total_iters * TE
    );
    for (t, it) in trace.iter().enumerate() {
        assert_eq!(it.t_int, t as i64);
        for &value in &it.epsilon_int {
            assert!(
                cfg.relu_min <= value && value <= cfg.clip_max,
                "epsilon {value} should be clipped into [{}, {}]",
                cfg.relu_min,
                cfg.clip_max
            );
        }
        for &value in &it.x_i_plus_1_int {
            assert!(
                cfg.value_min <= value && value <= cfg.value_max,
                "state {value} should remain in configured value range"
            );
        }
    }
    assert_ne!(
        trace.last().unwrap().x_i_plus_1_int,
        x0,
        "larger trace should actually update the state"
    );
}

#[test]
fn denoise_conv_dimension_padding_keeps_padded_state_zero() {
    const N: usize = 4;
    const IH: usize = 2;
    const IW: usize = 2;
    const TE: usize = 2;
    const KH: usize = 1;
    const KW: usize = 1;
    const OH: usize = 2;
    const OW: usize = 2;
    let cfg = FixedPointConfig::from_real_bounds(4, -8, 8, 4);
    let step = FixedDenoiseConvStepParams::<TE, KH, KW>::new([[4]], 0, [0, 0], 0, 4, 0);
    let public_params = FixedDenoiseConvPublicParams::new(
        vec![step],
        vec![[0, 0]],
        cfg,
        Conv2dPadding::default(),
        Conv2dRealShape::new(1, 1, KH, KW, 1, 1),
        1,
    );

    let (_, trace) = generate_fixed_point_denoise_conv_trace::<F, N, IH, IW, TE, KH, KW, OH, OW>(
        &public_params,
        [4, 0, 0, 0],
    );
    assert_eq!(trace[0].x_i_plus_1_int[1..], [0, 0, 0]);
}

#[test]
fn denoise_conv_trace_keeps_spatial_padding_zero() {
    const N: usize = 16;
    const IH: usize = 4;
    const IW: usize = 4;
    const TE: usize = 2;
    const KH: usize = 3;
    const KW: usize = 3;
    const OH: usize = 4;
    const OW: usize = 4;
    let cfg = FixedPointConfig::from_real_bounds(4, -8, 8, 4);
    let real_shape = Conv2dRealShape::new(2, 2, KH, KW, 2, 2);
    let step = FixedDenoiseConvStepParams::<TE, KH, KW>::new(
        [[0, 0, 0], [0, 4, 0], [0, 0, 0]],
        0,
        [0, 0],
        0,
        4,
        0,
    );
    let public_params = FixedDenoiseConvPublicParams::new(
        vec![step.clone(), step],
        vec![[0, 0], [0, 0]],
        cfg,
        Conv2dPadding::same_3x3(),
        real_shape.clone(),
        TE,
    );
    let x0 = [4, 8, 0, 0, -4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    let (_, trace) = generate_fixed_point_denoise_conv_trace::<F, N, IH, IW, TE, KH, KW, OH, OW>(
        &public_params,
        x0,
    );
    for it in trace {
        for row in 0..IH {
            for col in 0..IW {
                if row >= real_shape.ih_real || col >= real_shape.iw_real {
                    let idx = row * IW + col;
                    assert_eq!(it.x_i_int[idx], 0);
                    assert_eq!(it.x_i_plus_1_int[idx], 0);
                }
            }
        }
        for row in 0..OH {
            for col in 0..OW {
                if row >= real_shape.oh_real || col >= real_shape.ow_real {
                    assert_eq!(it.epsilon_int[row * OW + col], 0);
                }
            }
        }
    }
}

#[test]
fn denoise_conv_public_output_layout_has_y() {
    const N: usize = 4;
    const TE: usize = 1;
    const KH: usize = 1;
    const KW: usize = 1;
    let layout = PublicStateLayout::new(
        N,
        true,
        2,
        FixedDenoiseConvStepParams::<TE, KH, KW>::block_len(),
        TE,
    );
    assert_eq!(layout.x_range(), 0..4);
    assert_eq!(layout.y_range().unwrap(), 4..8);
    assert_eq!(layout.t_index(), 8);
}

#[test]
#[should_panic(expected = "expected_output spatial padding")]
fn denoise_conv_expected_output_padding_must_be_zero() {
    const N: usize = 4;
    const IH: usize = 2;
    const IW: usize = 2;
    const TE: usize = 1;
    const KH: usize = 1;
    const KW: usize = 1;
    const OH: usize = 2;
    const OW: usize = 2;
    let cfg = FixedPointConfig::from_real_bounds(4, -8, 8, 4);
    let public_params = FixedDenoiseConvPublicParams::new(
        vec![FixedDenoiseConvStepParams::<TE, KH, KW>::new(
            [[4]],
            0,
            [0],
            0,
            4,
            0,
        )],
        vec![[0]],
        cfg,
        Conv2dPadding::default(),
        Conv2dRealShape::new(1, 1, KH, KW, 1, 1),
        TE,
    );
    let _ = generate_fixed_point_denoise_conv_trace_with_expected_output::<
        F,
        N,
        IH,
        IW,
        TE,
        KH,
        KW,
        OH,
        OW,
    >(&public_params, [4, 0, 0, 0], [4, 9, 0, 0]);
}

#[test]
fn denoise_conv_output_binding_z0_layout_includes_y() {
    const N: usize = 4;
    const IH: usize = 2;
    const IW: usize = 2;
    const TE: usize = 1;
    const KH: usize = 1;
    const KW: usize = 1;
    const OH: usize = 2;
    const OW: usize = 2;
    let cfg = FixedPointConfig::from_real_bounds(4, -8, 8, 4);
    let public_params = FixedDenoiseConvPublicParams::new(
        vec![FixedDenoiseConvStepParams::<TE, KH, KW>::new(
            [[4]],
            0,
            [0],
            0,
            4,
            0,
        )],
        vec![[0]],
        cfg,
        Conv2dPadding::default(),
        Conv2dRealShape::full::<IH, IW, KH, KW, OH, OW>(),
        TE,
    );
    let (z0, trace, expected_output) = generate_fixed_point_denoise_conv_trace_with_computed_output::<
        F,
        N,
        IH,
        IW,
        TE,
        KH,
        KW,
        OH,
        OW,
    >(&public_params, [4, -4, 8, 0]);
    assert_eq!(expected_output, trace.last().unwrap().x_i_plus_1_int);
    assert_eq!(z0.len(), 2 * N + 1 + public_params.block_len() + TE);
    assert_eq!(z0[2 * N], <F as ff::Field>::ZERO);
}

#[test]
#[should_panic(expected = "step=0 conv_pre[0,0]")]
fn denoise_conv_trace_preflight_reports_relu_overflow() {
    const N: usize = 1;
    const IH: usize = 1;
    const IW: usize = 1;
    const TE: usize = 1;
    const KH: usize = 1;
    const KW: usize = 1;
    const OH: usize = 1;
    const OW: usize = 1;
    let cfg =
        FixedPointConfig::from_real_bounds(4, -1, 1, 1).with_integer_ranges(-100, 100, -100, 100);
    let public_params = FixedDenoiseConvPublicParams::new(
        vec![FixedDenoiseConvStepParams::<TE, KH, KW>::new(
            [[16]],
            0,
            [0],
            0,
            4,
            0,
        )],
        vec![[0]],
        cfg,
        Conv2dPadding::default(),
        Conv2dRealShape::full::<IH, IW, KH, KW, OH, OW>(),
        TE,
    );
    let _ = generate_fixed_point_denoise_conv_trace::<F, N, IH, IW, TE, KH, KW, OH, OW>(
        &public_params,
        [4],
    );
}

#[test]
fn denoise_conv_circuit_arity_uses_public_state_layout() {
    const N: usize = 4;
    const IH: usize = 2;
    const IW: usize = 2;
    const TE: usize = 1;
    const KH: usize = 1;
    const KW: usize = 1;
    const OH: usize = 2;
    const OW: usize = 2;
    let cfg = FixedPointConfig::from_real_bounds(4, -8, 8, 4);
    let circuit = build_fixed_point_denoise_conv_placeholder_circuit::<N, IH, IW, TE, KH, KW, OH, OW>(
        3,
        1,
        cfg,
        Conv2dPadding::default(),
        generate_simple_time_table::<TE>(3, 4),
    );
    assert_eq!(
        circuit.arity(),
        N + 1 + 3 * FixedDenoiseConvStepParams::<TE, KH, KW>::block_len() + 3 * TE
    );
}

#[test]
#[ignore]
fn small_denoise_conv_proof_verifies() {
    const N: usize = 16;
    const IH: usize = 4;
    const IW: usize = 4;
    const TE: usize = 2;
    const KH: usize = 3;
    const KW: usize = 3;
    const OH: usize = 4;
    const OW: usize = 4;
    let cfg = FixedPointConfig::from_real_bounds(16, -8, 8, 4);
    let padding = Conv2dPadding::same_3x3();
    let num_steps = 2;
    let num_iters_per_step = 2;
    let total_iters = num_steps * num_iters_per_step;
    let time_table = generate_simple_time_table::<TE>(total_iters, cfg.scale);
    let mut params = Vec::with_capacity(total_iters);
    for t in 0..total_iters {
        params.push(FixedDenoiseConvStepParams::<TE, KH, KW>::new(
            [[0, 2, 0], [2, 4 + t as i64, 2], [0, 2, 0]],
            0,
            [2, -1],
            0,
            14,
            2,
        ));
    }
    let public_params = FixedDenoiseConvPublicParams::new(
        params,
        time_table.clone(),
        cfg.clone(),
        padding.clone(),
        Conv2dRealShape::full::<IH, IW, KH, KW, OH, OW>(),
        TE,
    );
    let x0 = [16, -8, 8, 0, -4, 12, -8, 4, 8, 0, 16, -4, 0, 4, -12, 8];
    let (z0, trace) = generate_fixed_point_denoise_conv_trace::<F, N, IH, IW, TE, KH, KW, OH, OW>(
        &public_params,
        x0,
    );
    let placeholder =
        build_fixed_point_denoise_conv_placeholder_circuit::<N, IH, IW, TE, KH, KW, OH, OW>(
            total_iters,
            num_iters_per_step,
            cfg.clone(),
            padding.clone(),
            time_table.clone(),
        );
    let pp = setup_fixed_point_denoise_conv_public_params(&placeholder);
    let circuits = build_fixed_point_denoise_conv_step_circuits(
        &trace,
        num_steps,
        num_iters_per_step,
        total_iters,
        cfg,
        padding,
        time_table,
    );
    let recursive_snark = run_fixed_point_denoise_conv_recursive(&pp, &circuits, &z0);
    verify_fixed_point_denoise_conv_recursive(&recursive_snark, &pp, num_steps, &z0);
}

#[test]
#[ignore]
fn small_denoise_conv_output_binding_proof_verifies() {
    const N: usize = 4;
    const IH: usize = 2;
    const IW: usize = 2;
    const TE: usize = 1;
    const KH: usize = 1;
    const KW: usize = 1;
    const OH: usize = 2;
    const OW: usize = 2;
    let cfg = FixedPointConfig::from_real_bounds(4, -8, 8, 4);
    let padding = Conv2dPadding::default();
    let real_shape = Conv2dRealShape::full::<IH, IW, KH, KW, OH, OW>();
    let num_steps = 2;
    let num_iters_per_step = 1;
    let total_iters = num_steps * num_iters_per_step;
    let time_table = generate_simple_time_table::<TE>(total_iters, cfg.scale);
    let step = FixedDenoiseConvStepParams::<TE, KH, KW>::new([[4]], 0, [0], 0, 2, 2);
    let public_params = FixedDenoiseConvPublicParams::new(
        vec![step.clone(), step],
        time_table.clone(),
        cfg.clone(),
        padding.clone(),
        real_shape.clone(),
        TE,
    );
    let (z0, trace, expected_output) = generate_fixed_point_denoise_conv_trace_with_computed_output::<
        F,
        N,
        IH,
        IW,
        TE,
        KH,
        KW,
        OH,
        OW,
    >(&public_params, [4, -4, 8, 0]);
    assert_eq!(expected_output, trace.last().unwrap().x_i_plus_1_int);
    let placeholder = build_fixed_point_denoise_conv_placeholder_circuit_with_shape::<
        N,
        IH,
        IW,
        TE,
        KH,
        KW,
        OH,
        OW,
    >(
        total_iters,
        num_iters_per_step,
        cfg.clone(),
        padding.clone(),
        time_table.clone(),
        real_shape.clone(),
        TE,
        true,
    );
    let pp = setup_fixed_point_denoise_conv_public_params(&placeholder);
    let circuits = build_fixed_point_denoise_conv_step_circuits_with_shape(
        &trace,
        num_steps,
        num_iters_per_step,
        total_iters,
        cfg,
        padding,
        time_table,
        real_shape,
        TE,
        true,
    );
    let recursive_snark = run_fixed_point_denoise_conv_recursive(&pp, &circuits, &z0);
    verify_fixed_point_denoise_conv_recursive(&recursive_snark, &pp, num_steps, &z0);
}
