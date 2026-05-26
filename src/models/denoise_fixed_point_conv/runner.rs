use ff::Field;
use flate2::{write::ZlibEncoder, Compression};
use nova_snark::{
    nova::{CompressedSNARK, PublicParams, RecursiveSNARK},
    traits::snark::RelaxedR1CSSNARKTrait,
};
use std::time::Instant;

use crate::{
    fixed_point::FixedPointConfig,
    layers::conv2d::{Conv2dFixedPointWitness, Conv2dPadding, Conv2dRealShape},
    models::denoise_update::{DenoiseUpdateMode, DenoiseUpdateWitness},
    nova_ivc::{E1, E2, F1, G1, S1, S2},
};

use super::{
    FixedDenoiseConvIteration, FixedDenoiseConvStepParams, PublicFixedPointDenoiseConvCircuit,
};

pub fn build_fixed_point_denoise_conv_placeholder_circuit<
    const N: usize,
    const IH: usize,
    const IW: usize,
    const TE: usize,
    const KH: usize,
    const KW: usize,
    const OH: usize,
    const OW: usize,
>(
    total_iters: usize,
    num_iters_per_step: usize,
    config: FixedPointConfig,
    padding: Conv2dPadding,
    time_table_values: Vec<[i64; TE]>,
) -> PublicFixedPointDenoiseConvCircuit<G1, N, IH, IW, TE, KH, KW, OH, OW> {
    build_fixed_point_denoise_conv_placeholder_circuit_with_shape::<N, IH, IW, TE, KH, KW, OH, OW>(
        total_iters,
        num_iters_per_step,
        config,
        padding,
        time_table_values,
        Conv2dRealShape::full::<IH, IW, KH, KW, OH, OW>(),
        TE,
        false,
    )
}

pub fn build_fixed_point_denoise_conv_placeholder_circuit_with_shape<
    const N: usize,
    const IH: usize,
    const IW: usize,
    const TE: usize,
    const KH: usize,
    const KW: usize,
    const OH: usize,
    const OW: usize,
>(
    total_iters: usize,
    num_iters_per_step: usize,
    config: FixedPointConfig,
    padding: Conv2dPadding,
    time_table_values: Vec<[i64; TE]>,
    real_shape: Conv2dRealShape,
    te_real: usize,
    bind_output: bool,
) -> PublicFixedPointDenoiseConvCircuit<G1, N, IH, IW, TE, KH, KW, OH, OW> {
    real_shape.assert_fits::<IH, IW, KH, KW, OH, OW>();
    assert!(te_real <= TE, "te_real must be <= TE");
    PublicFixedPointDenoiseConvCircuit {
        num_iters_per_step,
        total_iters,
        clipped_relu_table: config.clipped_relu_table(),
        config,
        padding,
        time_table_values,
        real_shape,
        te_real,
        bind_output,
        commit_params: false,
        update_mode: DenoiseUpdateMode::DoubleFloor,
        param_hash_witnesses: Vec::new(),
        seq: vec![
            FixedDenoiseConvIteration {
                t_int: 0,
                time_emb_int: [0; TE],
                x_i_int: [0; N],
                time_raw_int: 0,
                time_quotient_int: 0,
                time_remainder_int: 0,
                time_bias_int: 0,
                conv_witness: Conv2dFixedPointWitness {
                    raw: [[0; OW]; OH],
                    quotient: [[0; OW]; OH],
                    remainder: [[0; OW]; OH],
                    output: [[0; OW]; OH],
                },
                epsilon_int: [0; N],
                alpha_x_int: [0; N],
                alpha_remainder_int: [0; N],
                beta_epsilon_int: [0; N],
                beta_remainder_int: [0; N],
                update_witness: DenoiseUpdateWitness::zero_double_floor(),
                x_i_plus_1_int: [0; N],
                epsilon: [F1::ZERO; N],
                x_i_plus_1: [F1::ZERO; N],
            };
            num_iters_per_step
        ],
    }
}

pub fn build_fixed_point_denoise_conv_step_circuits<
    const N: usize,
    const IH: usize,
    const IW: usize,
    const TE: usize,
    const KH: usize,
    const KW: usize,
    const OH: usize,
    const OW: usize,
>(
    trace: &[FixedDenoiseConvIteration<F1, N, IH, IW, TE, KH, KW, OH, OW>],
    num_steps: usize,
    num_iters_per_step: usize,
    total_iters: usize,
    config: FixedPointConfig,
    padding: Conv2dPadding,
    time_table_values: Vec<[i64; TE]>,
) -> Vec<PublicFixedPointDenoiseConvCircuit<G1, N, IH, IW, TE, KH, KW, OH, OW>> {
    build_fixed_point_denoise_conv_step_circuits_with_shape::<N, IH, IW, TE, KH, KW, OH, OW>(
        trace,
        num_steps,
        num_iters_per_step,
        total_iters,
        config,
        padding,
        time_table_values,
        Conv2dRealShape::full::<IH, IW, KH, KW, OH, OW>(),
        TE,
        false,
    )
}

pub fn build_fixed_point_denoise_conv_step_circuits_with_shape<
    const N: usize,
    const IH: usize,
    const IW: usize,
    const TE: usize,
    const KH: usize,
    const KW: usize,
    const OH: usize,
    const OW: usize,
>(
    trace: &[FixedDenoiseConvIteration<F1, N, IH, IW, TE, KH, KW, OH, OW>],
    num_steps: usize,
    num_iters_per_step: usize,
    total_iters: usize,
    config: FixedPointConfig,
    padding: Conv2dPadding,
    time_table_values: Vec<[i64; TE]>,
    real_shape: Conv2dRealShape,
    te_real: usize,
    bind_output: bool,
) -> Vec<PublicFixedPointDenoiseConvCircuit<G1, N, IH, IW, TE, KH, KW, OH, OW>> {
    assert_eq!(
        trace.len(),
        num_steps * num_iters_per_step,
        "trace length must equal num_steps * num_iters_per_step"
    );
    real_shape.assert_fits::<IH, IW, KH, KW, OH, OW>();
    assert!(te_real <= TE, "te_real must be <= TE");
    let table = config.clipped_relu_table();
    (0..num_steps)
        .map(|i| PublicFixedPointDenoiseConvCircuit {
            num_iters_per_step,
            total_iters,
            clipped_relu_table: table.clone(),
            config: config.clone(),
            padding: padding.clone(),
            time_table_values: time_table_values.clone(),
            real_shape: real_shape.clone(),
            te_real,
            bind_output,
            commit_params: false,
            update_mode: DenoiseUpdateMode::DoubleFloor,
            param_hash_witnesses: Vec::new(),
            seq: (0..num_iters_per_step)
                .map(|j| trace[i * num_iters_per_step + j].clone())
                .collect(),
        })
        .collect()
}

pub fn build_fixed_point_denoise_conv_commitment_placeholder_circuit<
    const N: usize,
    const IH: usize,
    const IW: usize,
    const TE: usize,
    const KH: usize,
    const KW: usize,
    const OH: usize,
    const OW: usize,
>(
    total_iters: usize,
    num_iters_per_step: usize,
    config: FixedPointConfig,
    padding: Conv2dPadding,
    time_table_values: Vec<[i64; TE]>,
    real_shape: Conv2dRealShape,
    te_real: usize,
) -> PublicFixedPointDenoiseConvCircuit<G1, N, IH, IW, TE, KH, KW, OH, OW> {
    let mut circuit = build_fixed_point_denoise_conv_placeholder_circuit_with_shape(
        total_iters,
        num_iters_per_step,
        config,
        padding,
        time_table_values,
        real_shape,
        te_real,
        true,
    );
    circuit.commit_params = true;
    circuit.param_hash_witnesses =
        vec![
            vec![F1::ZERO; FixedDenoiseConvStepParams::<TE, KH, KW>::block_len()];
            num_iters_per_step
        ];
    circuit
}

pub fn build_fixed_point_denoise_conv_commitment_step_circuits<
    const N: usize,
    const IH: usize,
    const IW: usize,
    const TE: usize,
    const KH: usize,
    const KW: usize,
    const OH: usize,
    const OW: usize,
>(
    trace: &[FixedDenoiseConvIteration<F1, N, IH, IW, TE, KH, KW, OH, OW>],
    hash_witnesses: &[Vec<F1>],
    num_steps: usize,
    num_iters_per_step: usize,
    total_iters: usize,
    config: FixedPointConfig,
    padding: Conv2dPadding,
    time_table_values: Vec<[i64; TE]>,
    real_shape: Conv2dRealShape,
    te_real: usize,
) -> Vec<PublicFixedPointDenoiseConvCircuit<G1, N, IH, IW, TE, KH, KW, OH, OW>> {
    assert_eq!(hash_witnesses.len(), num_steps * num_iters_per_step);
    let mut circuits = build_fixed_point_denoise_conv_step_circuits_with_shape(
        trace,
        num_steps,
        num_iters_per_step,
        total_iters,
        config,
        padding,
        time_table_values,
        real_shape,
        te_real,
        true,
    );
    for (i, circuit) in circuits.iter_mut().enumerate() {
        circuit.commit_params = true;
        circuit.param_hash_witnesses = (0..num_iters_per_step)
            .map(|j| hash_witnesses[i * num_iters_per_step + j].clone())
            .collect();
    }
    circuits
}

pub fn setup_fixed_point_denoise_conv_public_params<
    const N: usize,
    const IH: usize,
    const IW: usize,
    const TE: usize,
    const KH: usize,
    const KW: usize,
    const OH: usize,
    const OW: usize,
>(
    circuit: &PublicFixedPointDenoiseConvCircuit<G1, N, IH, IW, TE, KH, KW, OH, OW>,
) -> PublicParams<E1, E2, PublicFixedPointDenoiseConvCircuit<G1, N, IH, IW, TE, KH, KW, OH, OW>> {
    PublicParams::<
        E1,
        E2,
        PublicFixedPointDenoiseConvCircuit<G1, N, IH, IW, TE, KH, KW, OH, OW>,
    >::setup(circuit, &*S1::ck_floor(), &*S2::ck_floor())
    .expect("failed to setup public parameters")
}

pub fn run_fixed_point_denoise_conv_recursive<
    const N: usize,
    const IH: usize,
    const IW: usize,
    const TE: usize,
    const KH: usize,
    const KW: usize,
    const OH: usize,
    const OW: usize,
>(
    pp: &PublicParams<
        E1,
        E2,
        PublicFixedPointDenoiseConvCircuit<G1, N, IH, IW, TE, KH, KW, OH, OW>,
    >,
    circuits: &[PublicFixedPointDenoiseConvCircuit<G1, N, IH, IW, TE, KH, KW, OH, OW>],
    z0: &[F1],
) -> RecursiveSNARK<E1, E2, PublicFixedPointDenoiseConvCircuit<G1, N, IH, IW, TE, KH, KW, OH, OW>> {
    assert!(!circuits.is_empty(), "circuits must not be empty");
    let mut recursive_snark = RecursiveSNARK::<
        E1,
        E2,
        PublicFixedPointDenoiseConvCircuit<G1, N, IH, IW, TE, KH, KW, OH, OW>,
    >::new(pp, &circuits[0], z0)
    .expect("failed to initialize recursive SNARK");

    for (i, circuit) in circuits.iter().enumerate() {
        let start = Instant::now();
        let res = recursive_snark.prove_step(pp, circuit);
        assert!(res.is_ok(), "prove_step failed at step {i}");
        println!("RecursiveSNARK::prove_step {i}: took {:?}", start.elapsed());
    }
    recursive_snark
}

pub fn verify_fixed_point_denoise_conv_recursive<
    const N: usize,
    const IH: usize,
    const IW: usize,
    const TE: usize,
    const KH: usize,
    const KW: usize,
    const OH: usize,
    const OW: usize,
>(
    recursive_snark: &RecursiveSNARK<
        E1,
        E2,
        PublicFixedPointDenoiseConvCircuit<G1, N, IH, IW, TE, KH, KW, OH, OW>,
    >,
    pp: &PublicParams<
        E1,
        E2,
        PublicFixedPointDenoiseConvCircuit<G1, N, IH, IW, TE, KH, KW, OH, OW>,
    >,
    num_steps: usize,
    z0: &[F1],
) {
    let start = Instant::now();
    let res = recursive_snark.verify(pp, num_steps, z0);
    println!(
        "RecursiveSNARK::verify: {:?}, took {:?}",
        res.is_ok(),
        start.elapsed()
    );
    assert!(res.is_ok(), "recursive verification failed");
}

pub fn compress_fixed_point_denoise_conv_and_verify<
    const N: usize,
    const IH: usize,
    const IW: usize,
    const TE: usize,
    const KH: usize,
    const KW: usize,
    const OH: usize,
    const OW: usize,
>(
    pp: &PublicParams<
        E1,
        E2,
        PublicFixedPointDenoiseConvCircuit<G1, N, IH, IW, TE, KH, KW, OH, OW>,
    >,
    recursive_snark: &RecursiveSNARK<
        E1,
        E2,
        PublicFixedPointDenoiseConvCircuit<G1, N, IH, IW, TE, KH, KW, OH, OW>,
    >,
    num_steps: usize,
    z0: &[F1],
) -> usize {
    let (pk, vk) = CompressedSNARK::<_, _, _, S1, S2>::setup(pp)
        .expect("failed to setup compressed SNARK keys");

    let start = Instant::now();
    let compressed_snark = CompressedSNARK::<_, _, _, S1, S2>::prove(pp, &pk, recursive_snark)
        .expect("failed to produce compressed SNARK");
    println!("CompressedSNARK::prove took {:?}", start.elapsed());

    let mut encoder = ZlibEncoder::new(Vec::new(), Compression::default());
    bincode::serde::encode_into_std_write(
        &compressed_snark,
        &mut encoder,
        bincode::config::legacy(),
    )
    .expect("failed to serialize compressed SNARK");
    let compressed_snark_encoded = encoder.finish().expect("failed to finish compression");

    let start = Instant::now();
    let res = compressed_snark.verify(&vk, num_steps, z0);
    println!(
        "CompressedSNARK::verify: {:?}, took {:?}",
        res.is_ok(),
        start.elapsed()
    );
    assert!(res.is_ok(), "compressed verification failed");
    compressed_snark_encoded.len()
}
