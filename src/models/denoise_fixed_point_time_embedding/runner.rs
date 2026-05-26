use ff::Field;
use flate2::{write::ZlibEncoder, Compression};
use nova_snark::{
    nova::{CompressedSNARK, PublicParams, RecursiveSNARK},
    traits::snark::RelaxedR1CSSNARKTrait,
};
use std::time::Instant;

use crate::{
    denoise_fixed_point_time_embedding::{
        FixedDenoiseTimeEmbeddingIteration, PublicFixedPointDenoiseTimeEmbeddingCircuit,
    },
    fixed_point::FixedPointConfig,
    models::denoise_update::{DenoiseUpdateMode, DenoiseUpdateWitness},
    nova_ivc::{E1, E2, F1, G1, S1, S2},
};

pub fn build_fixed_point_denoise_time_embedding_placeholder_circuit<
    const N: usize,
    const TE: usize,
    const IN: usize,
    const H: usize,
>(
    total_iters: usize,
    num_iters_per_step: usize,
    config: FixedPointConfig,
    time_table_values: Vec<[i64; TE]>,
) -> PublicFixedPointDenoiseTimeEmbeddingCircuit<G1, N, TE, IN, H> {
    PublicFixedPointDenoiseTimeEmbeddingCircuit {
        num_iters_per_step,
        total_iters,
        clipped_relu_table: config.clipped_relu_table(),
        config,
        time_table_values,
        n_real: N,
        te_real: TE,
        in_real: IN,
        h_real: H,
        bind_public_output: false,
        commit_params: false,
        update_mode: DenoiseUpdateMode::DoubleFloor,
        param_hash_witnesses: Vec::new(),
        seq: vec![
            FixedDenoiseTimeEmbeddingIteration {
                t_int: 0,
                time_emb_int: [0; TE],
                x_i: [F1::ZERO; N],
                x_i_int: [0; N],
                mlp_input: [F1::ZERO; IN],
                mlp_input_int: [0; IN],
                hidden_raw: [F1::ZERO; H],
                hidden_quotient: [F1::ZERO; H],
                hidden_remainder: [F1::ZERO; H],
                hidden_affine: [F1::ZERO; H],
                hidden_act: [F1::ZERO; H],
                epsilon: [F1::ZERO; N],
                hidden_raw_int: [0; H],
                hidden_quotient_int: [0; H],
                hidden_remainder_int: [0; H],
                hidden_affine_int: [0; H],
                hidden_act_int: [0; H],
                epsilon_int: [0; N],
                output_raw: [F1::ZERO; N],
                output_quotient: [F1::ZERO; N],
                output_remainder: [F1::ZERO; N],
                output_raw_int: [0; N],
                output_quotient_int: [0; N],
                output_remainder_int: [0; N],
                alpha_x: [F1::ZERO; N],
                beta_epsilon: [F1::ZERO; N],
                x_i_plus_1: [F1::ZERO; N],
                alpha_x_int: [0; N],
                beta_epsilon_int: [0; N],
                x_i_plus_1_int: [0; N],
                alpha_mul_raw_int: [0; N],
                alpha_remainder_int: [0; N],
                beta_mul_raw_int: [0; N],
                beta_remainder_int: [0; N],
                update_witness: DenoiseUpdateWitness::zero_double_floor(),
            };
            num_iters_per_step
        ],
    }
}

pub fn build_fixed_point_denoise_time_embedding_step_circuits<
    const N: usize,
    const TE: usize,
    const IN: usize,
    const H: usize,
>(
    trace: &[FixedDenoiseTimeEmbeddingIteration<F1, N, TE, IN, H>],
    num_steps: usize,
    num_iters_per_step: usize,
    total_iters: usize,
    config: FixedPointConfig,
    time_table_values: Vec<[i64; TE]>,
) -> Vec<PublicFixedPointDenoiseTimeEmbeddingCircuit<G1, N, TE, IN, H>> {
    assert_eq!(
        trace.len(),
        num_steps * num_iters_per_step,
        "trace length must equal num_steps * num_iters_per_step"
    );
    let table = config.clipped_relu_table();
    (0..num_steps)
        .map(|i| PublicFixedPointDenoiseTimeEmbeddingCircuit {
            num_iters_per_step,
            total_iters,
            clipped_relu_table: table.clone(),
            config: config.clone(),
            time_table_values: time_table_values.clone(),
            n_real: N,
            te_real: TE,
            in_real: IN,
            h_real: H,
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

pub fn setup_fixed_point_denoise_time_embedding_public_params<
    const N: usize,
    const TE: usize,
    const IN: usize,
    const H: usize,
>(
    circuit: &PublicFixedPointDenoiseTimeEmbeddingCircuit<G1, N, TE, IN, H>,
) -> PublicParams<E1, E2, PublicFixedPointDenoiseTimeEmbeddingCircuit<G1, N, TE, IN, H>> {
    PublicParams::<E1, E2, PublicFixedPointDenoiseTimeEmbeddingCircuit<G1, N, TE, IN, H>>::setup(
        circuit,
        &*S1::ck_floor(),
        &*S2::ck_floor(),
    )
    .expect("failed to setup public parameters")
}

pub fn run_fixed_point_denoise_time_embedding_recursive<
    const N: usize,
    const TE: usize,
    const IN: usize,
    const H: usize,
>(
    pp: &PublicParams<E1, E2, PublicFixedPointDenoiseTimeEmbeddingCircuit<G1, N, TE, IN, H>>,
    circuits: &[PublicFixedPointDenoiseTimeEmbeddingCircuit<G1, N, TE, IN, H>],
    z0: &[F1],
) -> RecursiveSNARK<E1, E2, PublicFixedPointDenoiseTimeEmbeddingCircuit<G1, N, TE, IN, H>> {
    assert!(!circuits.is_empty(), "circuits must not be empty");
    let mut recursive_snark = RecursiveSNARK::<
        E1,
        E2,
        PublicFixedPointDenoiseTimeEmbeddingCircuit<G1, N, TE, IN, H>,
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

pub fn verify_fixed_point_denoise_time_embedding_recursive<
    const N: usize,
    const TE: usize,
    const IN: usize,
    const H: usize,
>(
    recursive_snark: &RecursiveSNARK<
        E1,
        E2,
        PublicFixedPointDenoiseTimeEmbeddingCircuit<G1, N, TE, IN, H>,
    >,
    pp: &PublicParams<E1, E2, PublicFixedPointDenoiseTimeEmbeddingCircuit<G1, N, TE, IN, H>>,
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

pub fn compress_fixed_point_denoise_time_embedding_and_verify<
    const N: usize,
    const TE: usize,
    const IN: usize,
    const H: usize,
>(
    pp: &PublicParams<E1, E2, PublicFixedPointDenoiseTimeEmbeddingCircuit<G1, N, TE, IN, H>>,
    recursive_snark: &RecursiveSNARK<
        E1,
        E2,
        PublicFixedPointDenoiseTimeEmbeddingCircuit<G1, N, TE, IN, H>,
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
