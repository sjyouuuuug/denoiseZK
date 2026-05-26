use denoise::{
    fixed_point::{encode_f64_round, FixedPointConfig},
    mlp_fixed_point_clipped_relu_lookup::{
        build_fixed_point_mlp_placeholder_circuit, build_fixed_point_mlp_step_circuits,
        compress_fixed_point_mlp_and_verify, generate_fixed_point_mlp_trace,
        run_fixed_point_mlp_recursive, setup_fixed_point_mlp_public_params,
        verify_fixed_point_mlp_recursive, FixedMlpClippedReluPublicParams,
        FixedMlpClippedReluStepParams,
    },
    nova_ivc::E1,
};
use nova_snark::traits::Engine;
use std::time::Instant;

// Proves a public-parameter time-varying fixed-point two-layer MLP recurrence:
//
//   h_i       = ClippedReLU(floor((W1_i * x_i) / S) + b1_i)
//   x_{i+1}  = floor((W2_i * h_i) / S) + b2_i
//
// All W1_i, b1_i, W2_i, b2_i are PUBLIC because they are embedded into z0.
// f64 values are encoded with round; in-circuit rescale uses mathematical floor.
fn main() {
    const N: usize = 2;
    const H: usize = 3;

    println!("Nova public-parameter fixed-point MLP+clipped-ReLU demo");
    println!("x_(i+1) = floor(W2_i * ClippedReLU(floor(W1_i*x_i/S)+b1_i)/S)+b2_i");
    println!("=========================================================");

    let config = FixedPointConfig::default_scale16();
    let num_steps = 3;
    let num_iters_per_step = 2;
    let total_iters = num_steps * num_iters_per_step;

    let scale = config.scale;
    let params_seq: Vec<FixedMlpClippedReluStepParams<N, H>> = vec![
        FixedMlpClippedReluStepParams::from_f64(
            [[0.50, 0.00], [0.00, 0.50], [-0.25, 0.25]],
            [0.00, 0.25, 0.00],
            [[0.50, 0.00, 0.00], [0.00, 0.50, 0.00]],
            [0.00, 0.00],
            scale,
        ),
        FixedMlpClippedReluStepParams::from_f64(
            [[0.25, 0.25], [0.25, -0.25], [0.00, 0.25]],
            [0.00, 0.00, 0.25],
            [[0.50, 0.00, 0.25], [0.00, 0.50, -0.25]],
            [0.25, 0.00],
            scale,
        ),
        FixedMlpClippedReluStepParams::from_f64(
            [[0.00, 0.50], [0.50, 0.00], [0.25, 0.25]],
            [0.25, 0.00, -0.25],
            [[0.50, 0.50, 0.00], [0.00, 0.50, 0.50]],
            [0.00, 0.25],
            scale,
        ),
        FixedMlpClippedReluStepParams::from_f64(
            [[0.50, 0.00], [-0.25, 0.25], [0.00, 0.50]],
            [0.00, 0.25, 0.00],
            [[0.50, 0.00, 0.00], [0.25, 0.50, 0.00]],
            [0.00, -0.25],
            scale,
        ),
        FixedMlpClippedReluStepParams::from_f64(
            [[0.25, -0.25], [0.00, 0.50], [0.50, 0.00]],
            [0.00, 0.00, 0.25],
            [[0.00, 0.50, 0.00], [0.50, 0.00, 0.50]],
            [0.25, 0.00],
            scale,
        ),
        FixedMlpClippedReluStepParams::from_f64(
            [[0.50, 0.00], [0.25, 0.25], [0.00, -0.25]],
            [0.25, 0.00, 0.25],
            [[0.50, 0.00, 0.25], [0.00, 0.50, 0.00]],
            [0.00, 0.00],
            scale,
        ),
    ];
    assert_eq!(
        params_seq.len(),
        total_iters,
        "params_seq length must match total iterations"
    );

    let public_params = FixedMlpClippedReluPublicParams::new(params_seq, config.clone());
    let x0 = [encode_f64_round(1.0, scale), encode_f64_round(-1.0, scale)];

    println!("Preparing public parameters...");
    let start = Instant::now();
    let placeholder =
        build_fixed_point_mlp_placeholder_circuit(total_iters, num_iters_per_step, config.clone());
    let pp = setup_fixed_point_mlp_public_params(&placeholder);
    println!("PublicParams::setup took {:?}", start.elapsed());
    println!(
        "Number of constraints per step (primary, secondary): {:?}",
        pp.num_constraints()
    );
    println!(
        "Number of variables per step (primary, secondary): {:?}",
        pp.num_variables()
    );

    println!("Generating public-parameter fixed-point MLP trace...");
    let (z0, trace) =
        generate_fixed_point_mlp_trace::<<E1 as Engine>::Scalar, N, H>(&public_params, x0);
    let table = config.clipped_relu_table();
    println!("Public state length = {}", z0.len());
    println!(
        "scale = {}, ReLU lookup range = [{}, {}], clip_max = {}",
        config.scale, table.min, table.max, table.clip_max
    );
    if let Some(last) = trace.last() {
        println!("Final fixed-point x_T witness = {:?}", last.x_i_plus_1_int);
    }

    let circuits = build_fixed_point_mlp_step_circuits(
        &trace,
        num_steps,
        num_iters_per_step,
        total_iters,
        config.clone(),
    );

    println!("Generating RecursiveSNARK...");
    let recursive_snark = run_fixed_point_mlp_recursive(&pp, &circuits, &z0);

    println!("Verifying RecursiveSNARK...");
    verify_fixed_point_mlp_recursive(&recursive_snark, &pp, num_steps, &z0);

    println!("Generating and verifying CompressedSNARK...");
    let proof_size = compress_fixed_point_mlp_and_verify(&pp, &recursive_snark, num_steps, &z0);
    println!("CompressedSNARK size: {} bytes", proof_size);
    println!("=========================================================");
}
