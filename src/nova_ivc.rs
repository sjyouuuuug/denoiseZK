use ff::Field;
use flate2::{write::ZlibEncoder, Compression};
use nova_snark::{
    nova::{CompressedSNARK, PublicParams, RecursiveSNARK},
    provider::{Bn256EngineKZG, GrumpkinEngine},
    traits::{snark::RelaxedR1CSSNARKTrait, Engine},
};
use std::time::Instant;

use crate::affine::{AffineCircuit, AffineIteration, AffineParams};

pub type E1 = Bn256EngineKZG;
pub type E2 = GrumpkinEngine;
pub type EE1 = nova_snark::provider::hyperkzg::EvaluationEngine<E1>;
pub type EE2 = nova_snark::provider::ipa_pc::EvaluationEngine<E2>;
pub type S1 = nova_snark::spartan::snark::RelaxedR1CSSNARK<E1, EE1>;
pub type S2 = nova_snark::spartan::snark::RelaxedR1CSSNARK<E2, EE2>;

pub type G1 = <E1 as Engine>::GE;
pub type F1 = <E1 as Engine>::Scalar;

pub fn build_placeholder_circuit<const N: usize>(
    params: &AffineParams<F1, N>,
    num_iters_per_step: usize,
) -> AffineCircuit<G1, N> {
    AffineCircuit {
        params: params.clone(),
        seq: vec![
            AffineIteration {
                x_i: [F1::ZERO; N],
                x_i_plus_1: [F1::ZERO; N],
            };
            num_iters_per_step
        ],
    }
}

pub fn build_step_circuits<const N: usize>(
    params: &AffineParams<F1, N>,
    trace: &[AffineIteration<F1, N>],
    num_steps: usize,
    num_iters_per_step: usize,
) -> Vec<AffineCircuit<G1, N>> {
    assert_eq!(
        trace.len(),
        num_steps * num_iters_per_step,
        "trace length must equal num_steps * num_iters_per_step"
    );

    (0..num_steps)
        .map(|i| AffineCircuit {
            params: params.clone(),
            seq: (0..num_iters_per_step)
                .map(|j| trace[i * num_iters_per_step + j].clone())
                .collect(),
        })
        .collect()
}

pub fn setup_public_params<const N: usize>(
    circuit: &AffineCircuit<G1, N>,
) -> PublicParams<E1, E2, AffineCircuit<G1, N>> {
    PublicParams::<E1, E2, AffineCircuit<G1, N>>::setup(circuit, &*S1::ck_floor(), &*S2::ck_floor())
        .expect("failed to setup public parameters")
}

pub fn run_recursive<const N: usize>(
    pp: &PublicParams<E1, E2, AffineCircuit<G1, N>>,
    circuits: &[AffineCircuit<G1, N>],
    z0: &[F1],
) -> RecursiveSNARK<E1, E2, AffineCircuit<G1, N>> {
    assert!(!circuits.is_empty(), "circuits must not be empty");

    let mut recursive_snark =
        RecursiveSNARK::<E1, E2, AffineCircuit<G1, N>>::new(pp, &circuits[0], z0)
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
    recursive_snark: &RecursiveSNARK<E1, E2, AffineCircuit<G1, N>>,
    pp: &PublicParams<E1, E2, AffineCircuit<G1, N>>,
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
    pp: &PublicParams<E1, E2, AffineCircuit<G1, N>>,
    recursive_snark: &RecursiveSNARK<E1, E2, AffineCircuit<G1, N>>,
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
