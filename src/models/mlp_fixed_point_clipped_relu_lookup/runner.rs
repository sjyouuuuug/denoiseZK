use ff::Field;
use flate2::{write::ZlibEncoder, Compression};
use nova_snark::{
    nova::{CompressedSNARK, PublicParams, RecursiveSNARK},
    traits::snark::RelaxedR1CSSNARKTrait,
};
use std::time::Instant;

use crate::{
    fixed_point::FixedPointConfig,
    mlp_fixed_point_clipped_relu_lookup::{
        FixedPointMlpClippedReluIteration, PublicFixedPointMlpClippedReluCircuit,
    },
    nova_ivc::{E1, E2, F1, G1, S1, S2},
};

pub fn build_placeholder_circuit<const N: usize, const H: usize>(
    total_iters: usize,
    num_iters_per_step: usize,
    config: FixedPointConfig,
) -> PublicFixedPointMlpClippedReluCircuit<G1, N, H> {
    PublicFixedPointMlpClippedReluCircuit {
        num_iters_per_step,
        total_iters,
        clipped_relu_table: config.clipped_relu_table(),
        config,
        seq: vec![
            FixedPointMlpClippedReluIteration {
                x_i: [F1::ZERO; N],
                hidden_raw_int: [0; H],
                hidden_quotient_int: [0; H],
                hidden_remainder_int: [0; H],
                hidden_affine_int: [0; H],
                hidden_act_int: [0; H],
                output_raw_int: [0; N],
                output_quotient_int: [0; N],
                output_remainder_int: [0; N],
                x_i_plus_1_int: [0; N],
                hidden_raw: [F1::ZERO; H],
                hidden_quotient: [F1::ZERO; H],
                hidden_remainder: [F1::ZERO; H],
                hidden_affine: [F1::ZERO; H],
                hidden_act: [F1::ZERO; H],
                output_raw: [F1::ZERO; N],
                output_quotient: [F1::ZERO; N],
                output_remainder: [F1::ZERO; N],
                x_i_plus_1: [F1::ZERO; N],
            };
            num_iters_per_step
        ],
    }
}

pub fn build_step_circuits<const N: usize, const H: usize>(
    trace: &[FixedPointMlpClippedReluIteration<F1, N, H>],
    num_steps: usize,
    num_iters_per_step: usize,
    total_iters: usize,
    config: FixedPointConfig,
) -> Vec<PublicFixedPointMlpClippedReluCircuit<G1, N, H>> {
    assert_eq!(
        trace.len(),
        num_steps * num_iters_per_step,
        "trace length must equal num_steps * num_iters_per_step"
    );
    let table = config.clipped_relu_table();
    (0..num_steps)
        .map(|i| PublicFixedPointMlpClippedReluCircuit {
            num_iters_per_step,
            total_iters,
            clipped_relu_table: table.clone(),
            config: config.clone(),
            seq: (0..num_iters_per_step)
                .map(|j| trace[i * num_iters_per_step + j].clone())
                .collect(),
        })
        .collect()
}

pub fn setup_public_params<const N: usize, const H: usize>(
    circuit: &PublicFixedPointMlpClippedReluCircuit<G1, N, H>,
) -> PublicParams<E1, E2, PublicFixedPointMlpClippedReluCircuit<G1, N, H>> {
    PublicParams::<E1, E2, PublicFixedPointMlpClippedReluCircuit<G1, N, H>>::setup(
        circuit,
        &*S1::ck_floor(),
        &*S2::ck_floor(),
    )
    .expect("failed to setup public parameters")
}

pub fn run_recursive<const N: usize, const H: usize>(
    pp: &PublicParams<E1, E2, PublicFixedPointMlpClippedReluCircuit<G1, N, H>>,
    circuits: &[PublicFixedPointMlpClippedReluCircuit<G1, N, H>],
    z0: &[F1],
) -> RecursiveSNARK<E1, E2, PublicFixedPointMlpClippedReluCircuit<G1, N, H>> {
    assert!(!circuits.is_empty(), "circuits must not be empty");
    let mut recursive_snark = RecursiveSNARK::<
        E1,
        E2,
        PublicFixedPointMlpClippedReluCircuit<G1, N, H>,
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

pub fn verify_recursive<const N: usize, const H: usize>(
    recursive_snark: &RecursiveSNARK<E1, E2, PublicFixedPointMlpClippedReluCircuit<G1, N, H>>,
    pp: &PublicParams<E1, E2, PublicFixedPointMlpClippedReluCircuit<G1, N, H>>,
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

pub fn compress_and_verify<const N: usize, const H: usize>(
    pp: &PublicParams<E1, E2, PublicFixedPointMlpClippedReluCircuit<G1, N, H>>,
    recursive_snark: &RecursiveSNARK<E1, E2, PublicFixedPointMlpClippedReluCircuit<G1, N, H>>,
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
