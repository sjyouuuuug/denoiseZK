use flate2::{write::ZlibEncoder, Compression};
use nova_snark::{
    nova::{CompressedSNARK, PublicParams, RecursiveSNARK},
    traits::{circuit::StepCircuit, snark::RelaxedR1CSSNARKTrait},
};
use std::time::Instant;

use crate::nova_ivc::{E1, E2, F1, S1, S2};

pub fn setup_public_params<C>(circuit: &C) -> PublicParams<E1, E2, C>
where
    C: StepCircuit<F1> + Clone + Send + Sync,
{
    PublicParams::<E1, E2, C>::setup(circuit, &*S1::ck_floor(), &*S2::ck_floor())
        .expect("failed to setup public parameters")
}

pub fn run_recursive<C>(
    pp: &PublicParams<E1, E2, C>,
    circuits: &[C],
    z0: &[F1],
) -> RecursiveSNARK<E1, E2, C>
where
    C: StepCircuit<F1> + Clone + Send + Sync,
{
    assert!(!circuits.is_empty(), "circuits must not be empty");
    let mut recursive_snark = RecursiveSNARK::<E1, E2, C>::new(pp, &circuits[0], z0)
        .expect("failed to initialize recursive SNARK");

    for (i, circuit) in circuits.iter().enumerate() {
        let start = Instant::now();
        let res = recursive_snark.prove_step(pp, circuit);
        assert!(res.is_ok(), "prove_step failed at step {i}");
        println!("RecursiveSNARK::prove_step {i}: took {:?}", start.elapsed());
    }

    recursive_snark
}

pub fn verify_recursive<C>(
    recursive_snark: &RecursiveSNARK<E1, E2, C>,
    pp: &PublicParams<E1, E2, C>,
    num_steps: usize,
    z0: &[F1],
) where
    C: StepCircuit<F1> + Clone + Send + Sync,
{
    let start = Instant::now();
    let res = recursive_snark.verify(pp, num_steps, z0);
    println!(
        "RecursiveSNARK::verify: {:?}, took {:?}",
        res.is_ok(),
        start.elapsed()
    );
    assert!(res.is_ok(), "recursive verification failed");
}

pub fn compress_and_verify<C>(
    pp: &PublicParams<E1, E2, C>,
    recursive_snark: &RecursiveSNARK<E1, E2, C>,
    num_steps: usize,
    z0: &[F1],
) -> usize
where
    C: StepCircuit<F1> + Clone + Send + Sync,
    CompressedSNARK<E1, E2, C, S1, S2>: serde::Serialize,
{
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
