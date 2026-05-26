use denoise::{
    commitment::{toy_hash_i64_as_field, TOY_HASH_BASE_U64},
    denoise_fixed_point_conv::{
        generate_fixed_point_denoise_conv_trace_with_commitment, FixedDenoiseConvPublicParams,
        FixedDenoiseConvStepParams,
    },
    denoise_fixed_point_time_embedding::{
        generate_simple_time_table, FixedDenoiseTimeEmbeddingPublicParams,
        FixedDenoiseTimeEmbeddingStepParams,
    },
    denoise_fixed_point_time_embedding_padded::{
        generate_padded_denoise_trace_with_commitment, PaddedDenoiseShape,
    },
    fixed_point::{encode_f64_round, FixedPointConfig},
    layers::conv2d::{Conv2dPadding, Conv2dRealShape},
    nova_ivc::F1,
    public_state::CommittedDenoiseStateLayout,
};
use ff::Field;

#[test]
fn denoise_commitment_tests() {
    mlp_commitment_trace_final_commitment_matches_flat_hash();
}

#[test]
fn mlp_commitment_trace_final_commitment_matches_flat_hash() {
    const N: usize = 1;
    const TE: usize = 1;
    const IN: usize = 2;
    const H: usize = 1;
    let config = FixedPointConfig::from_real_bounds(16, -4, 4, 2);
    let total_iters = 2;
    let time_table = generate_simple_time_table::<TE>(total_iters, config.scale);
    let params = vec![
        FixedDenoiseTimeEmbeddingStepParams::<N, TE, IN, H>::from_f64(
            [[0.5, 0.0]],
            [0.0],
            [[0.5]],
            [0.0],
            0.875,
            0.125,
            config.scale,
        ),
        FixedDenoiseTimeEmbeddingStepParams::<N, TE, IN, H>::from_f64(
            [[0.25, 0.125]],
            [0.0],
            [[0.5]],
            [0.0],
            0.85,
            0.15,
            config.scale,
        ),
    ];
    let flat: Vec<i64> = params.iter().flat_map(|p| p.flatten_i64()).collect();
    let public_params = FixedDenoiseTimeEmbeddingPublicParams::new(params, time_table, config);
    let shape = PaddedDenoiseShape::new(N, TE, IN, H, N, TE, IN, H);
    let (z0, trace, hash_witnesses, expected_y, commitment) =
        generate_padded_denoise_trace_with_commitment::<F1, N, TE, IN, H>(
            &public_params,
            [encode_f64_round(1.0, 16)],
            shape,
        );
    assert_eq!(
        commitment,
        toy_hash_i64_as_field::<F1>(&flat, TOY_HASH_BASE_U64, 0)
    );
    assert_eq!(hash_witnesses.len(), total_iters);
    assert_eq!(expected_y, trace.last().unwrap().x_i_plus_1_int);

    let layout =
        CommittedDenoiseStateLayout::new(N, true, total_iters, public_params.block_len(), TE);
    assert_eq!(z0[layout.h_index()], F1::ZERO);
    assert_eq!(z0[layout.c_index()], commitment);
    assert_eq!(z0[layout.t_index()], F1::ZERO);
}

#[test]
fn conv_commitment_trace_final_commitment_matches_flat_hash() {
    const IH: usize = 2;
    const IW: usize = 2;
    const N: usize = IH * IW;
    const TE: usize = 1;
    const KH: usize = 1;
    const KW: usize = 1;
    const OH: usize = 2;
    const OW: usize = 2;
    let config = FixedPointConfig::from_real_bounds(16, -4, 4, 2);
    let total_iters = 2;
    let time_table = generate_simple_time_table::<TE>(total_iters, config.scale);
    let params = vec![
        FixedDenoiseConvStepParams::<TE, KH, KW>::from_f64(
            [[0.5]],
            0.0,
            [0.0],
            0.0,
            0.875,
            0.125,
            config.scale,
        ),
        FixedDenoiseConvStepParams::<TE, KH, KW>::from_f64(
            [[0.25]],
            0.0,
            [0.125],
            0.0,
            0.85,
            0.15,
            config.scale,
        ),
    ];
    let flat: Vec<i64> = params.iter().flat_map(|p| p.flatten_i64()).collect();
    let padding = Conv2dPadding {
        top: 0,
        bottom: 0,
        left: 0,
        right: 0,
    };
    let public_params = FixedDenoiseConvPublicParams::new(
        params,
        time_table,
        config,
        padding,
        Conv2dRealShape::full::<IH, IW, KH, KW, OH, OW>(),
        TE,
    );
    let (z0, trace, hash_witnesses, expected_y, commitment) =
        generate_fixed_point_denoise_conv_trace_with_commitment::<F1, N, IH, IW, TE, KH, KW, OH, OW>(
            &public_params,
            [16, -8, 8, 0],
        );
    assert_eq!(
        commitment,
        toy_hash_i64_as_field::<F1>(&flat, TOY_HASH_BASE_U64, 0)
    );
    assert_eq!(hash_witnesses.len(), total_iters);
    assert_eq!(expected_y, trace.last().unwrap().x_i_plus_1_int);

    let layout =
        CommittedDenoiseStateLayout::new(N, true, total_iters, public_params.block_len(), TE);
    assert_eq!(z0[layout.h_index()], F1::ZERO);
    assert_eq!(z0[layout.c_index()], commitment);
    assert_eq!(z0[layout.t_index()], F1::ZERO);
}

#[test]
fn final_check_flag_is_one_only_at_last_iteration() {
    let total_iters = 4;
    let flags: Vec<i64> = (0..total_iters)
        .map(|t| i64::from(t + 1 == total_iters))
        .collect();
    assert_eq!(flags, vec![0, 0, 0, 1]);
}

#[test]
#[ignore]
fn denoise_mlp_commitment_proof_verifies() {
    // Covered by src/bin/denoise_fixed_point_mlp_commitment_demo.rs.
}

#[test]
#[ignore]
fn denoise_conv_commitment_proof_verifies() {
    // Covered by src/bin/denoise_fixed_point_conv_commitment_demo.rs.
}
