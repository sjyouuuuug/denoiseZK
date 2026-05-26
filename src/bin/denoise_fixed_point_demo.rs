use denoise::{
    denoise_fixed_point::{
        build_fixed_point_denoise_placeholder_circuit, build_fixed_point_denoise_step_circuits,
        compress_fixed_point_denoise_and_verify, generate_fixed_point_denoise_trace,
        run_fixed_point_denoise_recursive, setup_fixed_point_denoise_public_params,
        verify_fixed_point_denoise_recursive, FixedDenoisePublicParams, FixedDenoiseStepParams,
    },
    fixed_point::{encode_f64_round, FixedPointConfig},
    nova_ivc::E1,
};
use nova_snark::traits::Engine;
use std::time::Instant;

fn main() {
    const N: usize = 2;
    const H: usize = 3;

    println!("Nova public-parameter fixed-point denoise demo");
    println!("epsilon_t = MLP_t(x_t)");
    println!("x_(t+1) = floor(alpha_t*x_t/S) + floor(beta_t*epsilon_t/S)");
    println!("=========================================================");

    let config = FixedPointConfig::from_real_bounds(16, -4, 4, 2);
    let num_steps = 2;
    let num_iters_per_step = 2;
    let total_iters = num_steps * num_iters_per_step;
    let scale = config.scale;

    let mut params_seq = Vec::with_capacity(total_iters);
    for t in 0..total_iters {
        let alpha = 0.875 - 0.025 * (t as f64);
        let beta = 0.125 + 0.025 * (t as f64);
        params_seq.push(FixedDenoiseStepParams::<N, H>::from_f64(
            [[0.50, 0.00], [0.00, 0.50], [-0.25, 0.25]],
            [0.00, 0.125, 0.00],
            [[0.50, 0.00, 0.00], [0.00, 0.50, 0.00]],
            [0.00, 0.00],
            alpha,
            beta,
            scale,
        ));
    }

    let public_params = FixedDenoisePublicParams::new(params_seq, config.clone());
    let x0 = [encode_f64_round(1.0, scale), encode_f64_round(-0.5, scale)];

    println!("Preparing public parameters...");
    let start = Instant::now();
    let placeholder = build_fixed_point_denoise_placeholder_circuit(
        total_iters,
        num_iters_per_step,
        config.clone(),
    );
    let pp = setup_fixed_point_denoise_public_params(&placeholder);
    println!("PublicParams::setup took {:?}", start.elapsed());
    println!(
        "Number of constraints per step (primary, secondary): {:?}",
        pp.num_constraints()
    );
    println!(
        "Number of variables per step (primary, secondary): {:?}",
        pp.num_variables()
    );

    println!("Generating fixed-point denoise trace...");
    let (z0, trace) =
        generate_fixed_point_denoise_trace::<<E1 as Engine>::Scalar, N, H>(&public_params, x0);
    println!("Public state length = {}", z0.len());
    if let Some(last) = trace.last() {
        println!(
            "Final fixed-point x_T witness derived by the proof = {:?}",
            last.x_i_plus_1_int
        );
    }

    let circuits = build_fixed_point_denoise_step_circuits(
        &trace,
        num_steps,
        num_iters_per_step,
        total_iters,
        config,
    );

    println!("Generating RecursiveSNARK...");
    let recursive_snark = run_fixed_point_denoise_recursive(&pp, &circuits, &z0);

    println!("Verifying RecursiveSNARK...");
    verify_fixed_point_denoise_recursive(&recursive_snark, &pp, num_steps, &z0);

    println!("Generating and verifying CompressedSNARK...");
    let proof_size = compress_fixed_point_denoise_and_verify(&pp, &recursive_snark, num_steps, &z0);
    println!("CompressedSNARK size: {} bytes", proof_size);
    println!("=========================================================");
}
