use denoise::{
    denoise_fixed_point_time_embedding::{
        generate_fixed_point_denoise_time_embedding_trace, generate_simple_time_table,
        pad_denoise_time_embedding_step_params, pad_time_table_vec,
        run_fixed_point_denoise_time_embedding_recursive,
        setup_fixed_point_denoise_time_embedding_public_params,
        verify_fixed_point_denoise_time_embedding_recursive, FixedDenoiseTimeEmbeddingPublicParams,
        FixedDenoiseTimeEmbeddingStepParams,
    },
    denoise_fixed_point_time_embedding_padded::{
        build_padded_denoise_time_embedding_placeholder_circuit,
        build_padded_denoise_time_embedding_step_circuits, PaddedDenoiseShape,
    },
    fixed_point::FixedPointConfig,
    nova_ivc::E1,
    padding::pad_vector_i64,
};
use nova_snark::traits::Engine;

type F = <E1 as Engine>::Scalar;

const N_REAL: usize = 2;
const TE_REAL: usize = 2;
const IN_REAL: usize = 4;
const H_REAL: usize = 3;
const N_MAX: usize = 4;
const TE_MAX: usize = 4;
const IN_MAX: usize = 8;
const H_MAX: usize = 4;

fn real_step(scale: i64) -> FixedDenoiseTimeEmbeddingStepParams<N_REAL, TE_REAL, IN_REAL, H_REAL> {
    FixedDenoiseTimeEmbeddingStepParams::from_f64(
        [
            [0.50, 0.00, 0.125, 0.00],
            [0.00, 0.50, 0.00, 0.125],
            [-0.25, 0.25, 0.125, -0.125],
        ],
        [0.00, 0.125, 0.00],
        [[0.50, 0.00, 0.00], [0.00, 0.50, 0.00]],
        [0.00, 0.00],
        0.875,
        0.125,
        scale,
    )
}

#[test]
fn padded_param_block_len_matches_max_shape() {
    assert_eq!(
        FixedDenoiseTimeEmbeddingStepParams::<N_MAX, TE_MAX, IN_MAX, H_MAX>::block_len(),
        H_MAX * IN_MAX + H_MAX + N_MAX * H_MAX + N_MAX + 2
    );
}

#[test]
fn pad_denoise_params_preserves_real_entries() {
    let padded = pad_denoise_time_embedding_step_params::<
        N_REAL,
        TE_REAL,
        IN_REAL,
        H_REAL,
        N_MAX,
        TE_MAX,
        IN_MAX,
        H_MAX,
    >(real_step(16));
    assert_eq!(padded.w1[0][0], 8);
    assert_eq!(padded.w1[2][N_MAX + 1], -2);
    assert_eq!(padded.b1[1], 2);
    assert_eq!(padded.w2[1][1], 8);
}

#[test]
fn pad_denoise_params_zeroes_padding_entries() {
    let padded = pad_denoise_time_embedding_step_params::<
        N_REAL,
        TE_REAL,
        IN_REAL,
        H_REAL,
        N_MAX,
        TE_MAX,
        IN_MAX,
        H_MAX,
    >(real_step(16));
    for r in H_REAL..H_MAX {
        assert_eq!(padded.b1[r], 0);
    }
    for r in 0..H_MAX {
        for c in 0..IN_MAX {
            let is_real_x_weight = r < H_REAL && c < N_REAL;
            let is_real_embedding_weight = r < H_REAL && c >= N_MAX && c < N_MAX + TE_REAL;
            if !is_real_x_weight && !is_real_embedding_weight {
                assert_eq!(padded.w1[r][c], 0);
            }
        }
    }
    for r in 0..N_MAX {
        for c in 0..H_MAX {
            if r >= N_REAL || c >= H_REAL {
                assert_eq!(padded.w2[r][c], 0);
            }
        }
    }
    for r in N_REAL..N_MAX {
        assert_eq!(padded.b2[r], 0);
    }
}

#[test]
fn padded_time_table_zeroes_extra_te_entries() {
    let table = pad_time_table_vec::<TE_REAL, TE_MAX>(vec![[16, 0], [8, 8]]);
    assert_eq!(table[0], [16, 0, 0, 0]);
    assert_eq!(table[1], [8, 8, 0, 0]);
}

#[test]
fn padded_trace_keeps_state_padding_zero() {
    let cfg = FixedPointConfig::from_real_bounds(16, -4, 4, 2);
    let step = pad_denoise_time_embedding_step_params::<
        N_REAL,
        TE_REAL,
        IN_REAL,
        H_REAL,
        N_MAX,
        TE_MAX,
        IN_MAX,
        H_MAX,
    >(real_step(cfg.scale));
    let table =
        pad_time_table_vec::<TE_REAL, TE_MAX>(generate_simple_time_table::<TE_REAL>(2, cfg.scale));
    let params = FixedDenoiseTimeEmbeddingPublicParams::new(vec![step.clone(), step], table, cfg);
    let x0 = pad_vector_i64::<N_REAL, N_MAX>([16, -8]);
    let (_, trace) =
        generate_fixed_point_denoise_time_embedding_trace::<F, N_MAX, TE_MAX, IN_MAX, H_MAX>(
            &params, x0,
        );
    for it in trace {
        for j in N_REAL..N_MAX {
            assert_eq!(it.x_i_int[j], 0);
            assert_eq!(it.x_i_plus_1_int[j], 0);
        }
    }
}

#[test]
fn padded_trace_real_output_matches_unpadded_output() {
    let cfg = FixedPointConfig::from_real_bounds(16, -4, 4, 2);
    let real_params = vec![real_step(cfg.scale), real_step(cfg.scale)];
    let real_table = generate_simple_time_table::<TE_REAL>(2, cfg.scale);
    let unpadded = FixedDenoiseTimeEmbeddingPublicParams::new(
        real_params.clone(),
        real_table.clone(),
        cfg.clone(),
    );
    let (_, unpadded_trace) =
        generate_fixed_point_denoise_time_embedding_trace::<F, N_REAL, TE_REAL, IN_REAL, H_REAL>(
            &unpadded,
            [16, -8],
        );

    let padded_params = real_params
        .into_iter()
        .map(|step| {
            pad_denoise_time_embedding_step_params::<
                N_REAL,
                TE_REAL,
                IN_REAL,
                H_REAL,
                N_MAX,
                TE_MAX,
                IN_MAX,
                H_MAX,
            >(step)
        })
        .collect();
    let padded = FixedDenoiseTimeEmbeddingPublicParams::new(
        padded_params,
        pad_time_table_vec::<TE_REAL, TE_MAX>(real_table),
        cfg,
    );
    let (_, padded_trace) =
        generate_fixed_point_denoise_time_embedding_trace::<F, N_MAX, TE_MAX, IN_MAX, H_MAX>(
            &padded,
            pad_vector_i64::<N_REAL, N_MAX>([16, -8]),
        );

    assert_eq!(
        &padded_trace.last().unwrap().x_i_plus_1_int[..N_REAL],
        &unpadded_trace.last().unwrap().x_i_plus_1_int[..]
    );
}

#[test]
#[ignore]
fn small_padded_denoise_time_embedding_proof_verifies() {
    let cfg = FixedPointConfig::from_real_bounds(16, -4, 4, 2);
    let total_iters = 1;
    let num_steps = 1;
    let num_iters_per_step = 1;
    let shape = PaddedDenoiseShape::new(
        N_REAL, TE_REAL, IN_REAL, H_REAL, N_MAX, TE_MAX, IN_MAX, H_MAX,
    );
    let step = pad_denoise_time_embedding_step_params::<
        N_REAL,
        TE_REAL,
        IN_REAL,
        H_REAL,
        N_MAX,
        TE_MAX,
        IN_MAX,
        H_MAX,
    >(real_step(cfg.scale));
    let table = pad_time_table_vec::<TE_REAL, TE_MAX>(generate_simple_time_table::<TE_REAL>(
        total_iters,
        cfg.scale,
    ));
    let params = FixedDenoiseTimeEmbeddingPublicParams::new(vec![step], table.clone(), cfg.clone());
    let x0 = pad_vector_i64::<N_REAL, N_MAX>([16, -8]);
    let (z0, trace) =
        generate_fixed_point_denoise_time_embedding_trace::<F, N_MAX, TE_MAX, IN_MAX, H_MAX>(
            &params, x0,
        );
    let placeholder =
        build_padded_denoise_time_embedding_placeholder_circuit::<N_MAX, TE_MAX, IN_MAX, H_MAX>(
            total_iters,
            num_iters_per_step,
            cfg.clone(),
            table.clone(),
            shape,
        );
    let pp = setup_fixed_point_denoise_time_embedding_public_params(&placeholder);
    let circuits = build_padded_denoise_time_embedding_step_circuits(
        &trace,
        num_steps,
        num_iters_per_step,
        total_iters,
        cfg,
        table,
        shape,
    );
    let recursive_snark = run_fixed_point_denoise_time_embedding_recursive(&pp, &circuits, &z0);
    verify_fixed_point_denoise_time_embedding_recursive(&recursive_snark, &pp, num_steps, &z0);
}
