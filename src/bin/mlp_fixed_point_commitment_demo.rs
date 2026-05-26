use denoise::{
    commitment::{toy_hash_i64_as_field, toy_hash_prefixes_i64_as_field, TOY_HASH_BASE_U64},
    fixed_point::{encode_f64_round, FixedPointConfig},
    mlp_fixed_point_clipped_relu_lookup::{
        generate_fixed_point_mlp_trace, FixedMlpClippedReluPublicParams,
        FixedMlpClippedReluStepParams, PublicFixedPointMlpCommitmentCircuit,
    },
    nova_ivc::{E1, E2, F1, G1, S1, S2},
    public_state::PublicStateLayout,
};
use ff::Field;
use flate2::{write::ZlibEncoder, Compression};
use nova_snark::{
    nova::{CompressedSNARK, PublicParams, RecursiveSNARK},
    traits::{snark::RelaxedR1CSSNARKTrait, Engine},
};
use std::time::Instant;

fn main() {
    const N: usize = 2;
    const H: usize = 3;

    println!("Nova fixed-point MLP + toy parameter commitment demo");
    println!("Statement: prove y = MLP_params(x0) and C = ToyHash(params)");
    println!("Toy hash is not cryptographically binding.");
    println!("=========================================================");

    let config = FixedPointConfig::from_real_bounds(16, -4, 4, 2);
    let scale = config.scale;
    let total_iters = 2;
    let num_steps = 1;
    let num_iters_per_step = 2;

    let params_seq = vec![
        FixedMlpClippedReluStepParams::<N, H>::from_f64(
            [[0.50, 0.00], [0.00, 0.50], [-0.25, 0.25]],
            [0.00, 0.125, 0.00],
            [[0.50, 0.00, 0.00], [0.00, 0.50, 0.00]],
            [0.00, 0.00],
            scale,
        ),
        FixedMlpClippedReluStepParams::<N, H>::from_f64(
            [[0.375, 0.125], [0.125, 0.375], [-0.125, 0.25]],
            [0.00, 0.00, 0.125],
            [[0.50, 0.125, 0.00], [0.00, 0.375, 0.125]],
            [0.00, 0.00],
            scale,
        ),
    ];
    let public_params = FixedMlpClippedReluPublicParams::new(params_seq.clone(), config.clone());
    let x0 = [encode_f64_round(1.0, scale), encode_f64_round(-0.5, scale)];
    let (_, trace) =
        generate_fixed_point_mlp_trace::<<E1 as Engine>::Scalar, N, H>(&public_params, x0);
    let expected_y = trace.last().unwrap().x_i_plus_1_int;

    let flat_params: Vec<i64> = params_seq.iter().flat_map(|p| p.flatten_i64()).collect();
    let commitment =
        toy_hash_i64_as_field::<<E1 as Engine>::Scalar>(&flat_params, TOY_HASH_BASE_U64, 0);
    let hash_witnesses = toy_hash_prefixes_i64_as_field::<<E1 as Engine>::Scalar>(
        &flat_params,
        TOY_HASH_BASE_U64,
        0,
    );

    let layout = PublicStateLayout::new_with_commitment(
        N,
        true,
        true,
        total_iters,
        FixedMlpClippedReluStepParams::<N, H>::block_len(),
        0,
    );
    let mut z0 = Vec::with_capacity(layout.state_len());
    z0.extend(
        x0.iter()
            .map(|&v| denoise::clipped_relu::field_from_i64::<F1>(v)),
    );
    z0.extend(
        expected_y
            .iter()
            .map(|&v| denoise::clipped_relu::field_from_i64::<F1>(v)),
    );
    z0.push(commitment);
    z0.push(F1::ZERO);
    for params in &params_seq {
        z0.extend(params.flatten_field::<F1>());
    }
    layout.assert_state_len(z0.len());

    let placeholder = PublicFixedPointMlpCommitmentCircuit::<G1, N, H> {
        num_iters_per_step,
        total_iters,
        clipped_relu_table: config.clipped_relu_table(),
        config: config.clone(),
        seq: vec![trace[0].clone(); num_iters_per_step],
        param_hash_witnesses: hash_witnesses.clone(),
    };

    println!("Preparing public parameters...");
    let start = Instant::now();
    let pp = PublicParams::<E1, E2, PublicFixedPointMlpCommitmentCircuit<G1, N, H>>::setup(
        &placeholder,
        &*S1::ck_floor(),
        &*S2::ck_floor(),
    )
    .expect("failed to setup public parameters");
    println!("PublicParams::setup took {:?}", start.elapsed());
    println!(
        "constraints (primary, secondary): {:?}",
        pp.num_constraints()
    );
    println!("variables (primary, secondary): {:?}", pp.num_variables());

    println!("Commitment C = {:?}", commitment);
    println!("Expected public output y = {:?}", expected_y);
    println!("Public state length = {}", z0.len());

    let circuit = PublicFixedPointMlpCommitmentCircuit::<G1, N, H> {
        num_iters_per_step,
        total_iters,
        clipped_relu_table: config.clipped_relu_table(),
        config,
        seq: trace,
        param_hash_witnesses: hash_witnesses,
    };
    let circuits = vec![circuit];

    println!("Generating RecursiveSNARK...");
    let mut recursive_snark = RecursiveSNARK::<
        E1,
        E2,
        PublicFixedPointMlpCommitmentCircuit<G1, N, H>,
    >::new(&pp, &circuits[0], &z0)
    .expect("failed to initialize recursive SNARK");
    for (i, circuit) in circuits.iter().enumerate() {
        let start = Instant::now();
        recursive_snark
            .prove_step(&pp, circuit)
            .expect("prove_step failed");
        println!("RecursiveSNARK::prove_step {i}: took {:?}", start.elapsed());
    }

    let start = Instant::now();
    let res = recursive_snark.verify(&pp, num_steps, &z0);
    println!(
        "RecursiveSNARK::verify: {:?}, took {:?}",
        res.is_ok(),
        start.elapsed()
    );
    assert!(res.is_ok(), "recursive verification failed");

    let (pk, vk) = CompressedSNARK::<_, _, _, S1, S2>::setup(&pp)
        .expect("failed to setup compressed SNARK keys");
    let start = Instant::now();
    let compressed_snark = CompressedSNARK::<_, _, _, S1, S2>::prove(&pp, &pk, &recursive_snark)
        .expect("failed to produce compressed SNARK");
    println!("CompressedSNARK::prove took {:?}", start.elapsed());
    let mut encoder = ZlibEncoder::new(Vec::new(), Compression::default());
    bincode::serde::encode_into_std_write(
        &compressed_snark,
        &mut encoder,
        bincode::config::legacy(),
    )
    .expect("failed to serialize compressed SNARK");
    let encoded = encoder.finish().expect("failed to finish compression");
    let start = Instant::now();
    let res = compressed_snark.verify(&vk, num_steps, &z0);
    println!(
        "CompressedSNARK::verify: {:?}, took {:?}",
        res.is_ok(),
        start.elapsed()
    );
    assert!(res.is_ok(), "compressed verification failed");
    println!("CompressedSNARK size: {} bytes", encoded.len());
    println!("=========================================================");
}
