use denoise::{
    denoise_fixed_point_time_embedding::{
        generate_simple_time_table, pad_denoise_time_embedding_step_params, pad_time_table_vec,
        run_fixed_point_denoise_time_embedding_recursive,
        setup_fixed_point_denoise_time_embedding_public_params,
        verify_fixed_point_denoise_time_embedding_recursive, FixedDenoiseTimeEmbeddingPublicParams,
        FixedDenoiseTimeEmbeddingStepParams,
    },
    denoise_fixed_point_time_embedding_padded::{
        build_padded_output_denoise_time_embedding_placeholder_circuit,
        build_padded_output_denoise_time_embedding_step_circuits, build_z0_with_public_output,
        generate_padded_denoise_trace_with_computed_output,
        generate_padded_denoise_trace_with_expected_output, PaddedDenoiseShape,
    },
    fixed_point::FixedPointConfig,
    nova_ivc::E1,
    padding::pad_vector_i64,
};
use ff::Field;
use nova_snark::traits::{circuit::StepCircuit, Engine};

type F = <E1 as Engine>::Scalar;

const N_REAL: usize = 2;
const TE_REAL: usize = 2;
const IN_REAL: usize = 4;
const H_REAL: usize = 3;
const N_MAX: usize = 4;
const TE_MAX: usize = 4;
const IN_MAX: usize = 8;
const H_MAX: usize = 4;

fn shape() -> PaddedDenoiseShape {
    PaddedDenoiseShape::new(
        N_REAL, TE_REAL, IN_REAL, H_REAL, N_MAX, TE_MAX, IN_MAX, H_MAX,
    )
}

fn params() -> FixedDenoiseTimeEmbeddingPublicParams<N_MAX, TE_MAX, IN_MAX, H_MAX> {
    let cfg = FixedPointConfig::from_real_bounds(16, -4, 4, 2);
    let real = FixedDenoiseTimeEmbeddingStepParams::<N_REAL, TE_REAL, IN_REAL, H_REAL>::from_f64(
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
        cfg.scale,
    );
    let padded = pad_denoise_time_embedding_step_params::<
        N_REAL,
        TE_REAL,
        IN_REAL,
        H_REAL,
        N_MAX,
        TE_MAX,
        IN_MAX,
        H_MAX,
    >(real);
    FixedDenoiseTimeEmbeddingPublicParams::new(
        vec![padded],
        pad_time_table_vec::<TE_REAL, TE_MAX>(generate_simple_time_table::<TE_REAL>(1, cfg.scale)),
        cfg,
    )
}

#[test]
fn z0_layout_includes_expected_output() {
    let params = params();
    let x0 = pad_vector_i64::<N_REAL, N_MAX>([16, -8]);
    let expected = [7, -5, 0, 0];
    let z0 = build_z0_with_public_output::<F, N_MAX, TE_MAX, IN_MAX, H_MAX>(
        &params,
        x0,
        expected,
        shape(),
    );
    assert_eq!(
        z0[0..N_MAX],
        x0.map(denoise::clipped_relu::field_from_i64::<F>)
    );
    assert_eq!(
        z0[N_MAX..2 * N_MAX],
        expected.map(denoise::clipped_relu::field_from_i64::<F>)
    );
    assert_eq!(z0[2 * N_MAX], F::ZERO);
}

#[test]
#[should_panic(expected = "expected output padding entry")]
fn expected_output_padding_must_be_zero() {
    let params = params();
    let x0 = pad_vector_i64::<N_REAL, N_MAX>([16, -8]);
    let _ = build_z0_with_public_output::<F, N_MAX, TE_MAX, IN_MAX, H_MAX>(
        &params,
        x0,
        [7, -5, 1, 0],
        shape(),
    );
}

#[test]
fn computed_expected_output_matches_trace_final_state() {
    let params = params();
    let x0 = pad_vector_i64::<N_REAL, N_MAX>([16, -8]);
    let (_z0, trace, expected) =
        generate_padded_denoise_trace_with_computed_output::<F, N_MAX, TE_MAX, IN_MAX, H_MAX>(
            &params,
            x0,
            shape(),
        );
    assert_eq!(expected, trace.last().unwrap().x_i_plus_1_int);
}

#[test]
fn circuit_arity_includes_expected_output() {
    let params = params();
    let placeholder = build_padded_output_denoise_time_embedding_placeholder_circuit::<
        N_MAX,
        TE_MAX,
        IN_MAX,
        H_MAX,
    >(
        1,
        1,
        params.config.clone(),
        params.time_table.clone(),
        shape(),
    );
    let block_len =
        denoise::denoise_fixed_point_time_embedding::FixedDenoiseTimeEmbeddingStepParams::<
            N_MAX,
            TE_MAX,
            IN_MAX,
            H_MAX,
        >::block_len();
    assert_eq!(placeholder.arity(), 2 * N_MAX + 1 + block_len + TE_MAX);
}

#[test]
fn public_output_is_carried_across_step_circuits() {
    let params = params();
    let x0 = pad_vector_i64::<N_REAL, N_MAX>([16, -8]);
    let expected = [7, -5, 0, 0];
    let (z0, trace) = generate_padded_denoise_trace_with_expected_output::<
        F,
        N_MAX,
        TE_MAX,
        IN_MAX,
        H_MAX,
    >(&params, x0, expected, shape());
    assert_eq!(
        z0[N_MAX..2 * N_MAX],
        expected.map(denoise::clipped_relu::field_from_i64::<F>)
    );
    assert_eq!(trace.len(), 1);
}

#[test]
#[ignore]
fn padded_denoise_public_output_proof_verifies() {
    let params = params();
    let x0 = pad_vector_i64::<N_REAL, N_MAX>([16, -8]);
    let (z0, trace, expected) =
        generate_padded_denoise_trace_with_computed_output::<F, N_MAX, TE_MAX, IN_MAX, H_MAX>(
            &params,
            x0,
            shape(),
        );
    assert_eq!(expected, trace.last().unwrap().x_i_plus_1_int);
    let placeholder = build_padded_output_denoise_time_embedding_placeholder_circuit::<
        N_MAX,
        TE_MAX,
        IN_MAX,
        H_MAX,
    >(
        1,
        1,
        params.config.clone(),
        params.time_table.clone(),
        shape(),
    );
    let pp = setup_fixed_point_denoise_time_embedding_public_params(&placeholder);
    let circuits = build_padded_output_denoise_time_embedding_step_circuits(
        &trace,
        1,
        1,
        1,
        params.config.clone(),
        params.time_table.clone(),
        shape(),
    );
    let recursive_snark = run_fixed_point_denoise_time_embedding_recursive(&pp, &circuits, &z0);
    verify_fixed_point_denoise_time_embedding_recursive(&recursive_snark, &pp, 1, &z0);
}
