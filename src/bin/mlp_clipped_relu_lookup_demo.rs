use denoise::{
    clipped_relu::ClippedReluLookupTable,
    mlp_clipped_relu_lookup::{
        build_mlp_clipped_relu_placeholder_circuit, build_mlp_clipped_relu_step_circuits,
        compress_mlp_clipped_relu_and_verify, generate_mlp_clipped_relu_trace,
        run_mlp_clipped_relu_recursive, setup_mlp_clipped_relu_public_params,
        verify_mlp_clipped_relu_recursive, IntMlpClippedReluStepParams, MlpClippedReluPublicParams,
    },
    nova_ivc::E1,
};
use nova_snark::traits::Engine;
use std::time::Instant;

// Proves a public-parameter time-varying two-layer MLP recurrence:
//
//   h_i       = ClippedReLU(W1_i x_i + b1_i)
//   x_{i+1}  = W2_i h_i + b2_i
//
// All W1_i, b1_i, W2_i, b2_i are PUBLIC because they are embedded into z0.
// The private witness contains the intermediate hidden pre-activation, hidden activation,
// and next-state values.

fn main() {
    const N: usize = 2; // state dimension
    const H: usize = 3; // hidden dimension

    println!("Nova public-parameter MLP+clipped-ReLU demo");
    println!("x_(i+1) = W2_i * ClippedReLU(W1_i * x_i + b1_i) + b2_i");
    println!("=========================================================");

    let num_steps = 3;
    let num_iters_per_step = 2;
    let total_iters = num_steps * num_iters_per_step;

    let table = ClippedReluLookupTable::new(-64, 64, 8);

    let params_seq: Vec<IntMlpClippedReluStepParams<N, H>> = vec![
        IntMlpClippedReluStepParams::new(
            [[1, 0], [0, 1], [-1, 1]],
            [0, 1, 0],
            [[1, 0, 0], [0, 1, 0]],
            [0, 0],
        ),
        IntMlpClippedReluStepParams::new(
            [[1, 1], [1, -1], [0, 1]],
            [0, 0, 1],
            [[1, 0, 1], [0, 1, -1]],
            [1, 0],
        ),
        IntMlpClippedReluStepParams::new(
            [[0, 1], [1, 0], [1, 1]],
            [1, 0, -1],
            [[1, 1, 0], [0, 1, 1]],
            [0, 1],
        ),
        IntMlpClippedReluStepParams::new(
            [[1, 0], [-1, 1], [0, 1]],
            [0, 2, 0],
            [[1, 0, 0], [1, 1, 0]],
            [0, -1],
        ),
        IntMlpClippedReluStepParams::new(
            [[1, -1], [0, 1], [1, 0]],
            [0, 0, 1],
            [[0, 1, 0], [1, 0, 1]],
            [1, 0],
        ),
        IntMlpClippedReluStepParams::new(
            [[1, 0], [1, 1], [0, -1]],
            [1, 0, 2],
            [[1, 0, 1], [0, 1, 0]],
            [0, 0],
        ),
    ];
    assert_eq!(
        params_seq.len(),
        total_iters,
        "params_seq length must match total iterations"
    );

    let public_params = MlpClippedReluPublicParams::new(params_seq, table.clone());
    let x0 = [1i64, -1i64];

    println!("Preparing public parameters...");
    let start = Instant::now();
    let placeholder =
        build_mlp_clipped_relu_placeholder_circuit(total_iters, num_iters_per_step, table.clone());
    let pp = setup_mlp_clipped_relu_public_params(&placeholder);
    println!("PublicParams::setup took {:?}", start.elapsed());
    println!(
        "Number of constraints per step (primary, secondary): {:?}",
        pp.num_constraints()
    );
    println!(
        "Number of variables per step (primary, secondary): {:?}",
        pp.num_variables()
    );

    println!("Generating public-parameter MLP trace...");
    let (z0, trace) =
        generate_mlp_clipped_relu_trace::<<E1 as Engine>::Scalar, N, H>(&public_params, x0);
    println!("Public state length = {}", z0.len());
    println!(
        "Using clipped-ReLU lookup table over signed integer domain [{}, {}] ({} entries), clip_max = {}",
        table.min,
        table.max,
        table.size(),
        table.clip_max,
    );
    if let Some(last) = trace.last() {
        println!("Final x_T witness = {:?}", last.x_i_plus_1_int);
    }

    let circuits = build_mlp_clipped_relu_step_circuits(
        &trace,
        num_steps,
        num_iters_per_step,
        total_iters,
        table.clone(),
    );

    println!("Generating RecursiveSNARK...");
    let recursive_snark = run_mlp_clipped_relu_recursive(&pp, &circuits, &z0);

    println!("Verifying RecursiveSNARK...");
    verify_mlp_clipped_relu_recursive(&recursive_snark, &pp, num_steps, &z0);

    println!("Generating and verifying CompressedSNARK...");
    let proof_size = compress_mlp_clipped_relu_and_verify(&pp, &recursive_snark, num_steps, &z0);
    println!("CompressedSNARK size: {} bytes", proof_size);
    println!("=========================================================");
}
