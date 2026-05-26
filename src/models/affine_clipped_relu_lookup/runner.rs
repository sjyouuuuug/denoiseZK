use ff::Field;
use flate2::{write::ZlibEncoder, Compression};
use nova_snark::{
    nova::{CompressedSNARK, PublicParams, RecursiveSNARK},
    traits::snark::RelaxedR1CSSNARKTrait,
};
use std::time::Instant;

use crate::{
    affine_clipped_relu_lookup::{
        AffineClippedReluLookupCircuit, AffineClippedReluLookupIteration,
        AffineClippedReluLookupParams,
    },
    nova_ivc::{E1, E2, F1, G1, S1, S2},
};

pub fn build_placeholder_circuit<const N: usize>(
    params: &AffineClippedReluLookupParams<F1, N>,
    num_iters_per_step: usize,
) -> AffineClippedReluLookupCircuit<G1, N> {
    AffineClippedReluLookupCircuit {
        params: params.clone(),
        seq: vec![
            AffineClippedReluLookupIteration {
                x_i: [F1::ZERO; N],
                affine_out: [F1::ZERO; N],
                x_i_plus_1: [F1::ZERO; N],
                affine_out_int: [0; N],
                x_i_plus_1_int: [0; N],
            };
            num_iters_per_step
        ],
    }
}

pub fn build_step_circuits<const N: usize>(
    params: &AffineClippedReluLookupParams<F1, N>,
    trace: &[AffineClippedReluLookupIteration<F1, N>],
    num_steps: usize,
    num_iters_per_step: usize,
) -> Vec<AffineClippedReluLookupCircuit<G1, N>> {
    assert_eq!(
        trace.len(),
        num_steps * num_iters_per_step,
        "trace length must equal num_steps * num_iters_per_step"
    );

    (0..num_steps)
        .map(|i| AffineClippedReluLookupCircuit {
            params: params.clone(),
            seq: (0..num_iters_per_step)
                .map(|j| trace[i * num_iters_per_step + j].clone())
                .collect(),
        })
        .collect()
}

pub fn setup_public_params<const N: usize>(
    circuit: &AffineClippedReluLookupCircuit<G1, N>,
) -> PublicParams<E1, E2, AffineClippedReluLookupCircuit<G1, N>> {
    PublicParams::<E1, E2, AffineClippedReluLookupCircuit<G1, N>>::setup(
        circuit,
        &*S1::ck_floor(),
        &*S2::ck_floor(),
    )
    .expect("failed to setup public parameters")
}

pub fn run_recursive<const N: usize>(
    pp: &PublicParams<E1, E2, AffineClippedReluLookupCircuit<G1, N>>,
    circuits: &[AffineClippedReluLookupCircuit<G1, N>],
    z0: &[F1],
) -> RecursiveSNARK<E1, E2, AffineClippedReluLookupCircuit<G1, N>> {
    assert!(!circuits.is_empty(), "circuits must not be empty");

    let mut recursive_snark =
        RecursiveSNARK::<E1, E2, AffineClippedReluLookupCircuit<G1, N>>::new(pp, &circuits[0], z0)
            .expect("failed to initialize recursive SNARK");

    for (i, circuit) in circuits.iter().enumerate() {
        let start = Instant::now();
        let res = recursive_snark.prove_step(pp, circuit);
        assert!(res.is_ok(), "prove_step failed at step {i}");
        println!("RecursiveSNARK::prove_step {i}: took {:?}", start.elapsed());
    }

    recursive_snark
}

pub fn verify_recursive<const N: usize>(
    recursive_snark: &RecursiveSNARK<E1, E2, AffineClippedReluLookupCircuit<G1, N>>,
    pp: &PublicParams<E1, E2, AffineClippedReluLookupCircuit<G1, N>>,
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

pub fn compress_and_verify<const N: usize>(
    pp: &PublicParams<E1, E2, AffineClippedReluLookupCircuit<G1, N>>,
    recursive_snark: &RecursiveSNARK<E1, E2, AffineClippedReluLookupCircuit<G1, N>>,
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
