use denoise::{
    affine_clipped_relu_lookup::{
        build_affine_clipped_relu_lookup_placeholder_circuit,
        build_affine_clipped_relu_lookup_step_circuits,
        compress_affine_clipped_relu_lookup_and_verify, generate_affine_clipped_relu_lookup_trace,
        run_affine_clipped_relu_lookup_recursive, setup_affine_clipped_relu_lookup_public_params,
        verify_affine_clipped_relu_lookup_recursive, IntAffineClippedReluLookupParams,
    },
    clipped_relu::ClippedReluLookupTable,
    nova_ivc::E1,
};
use nova_snark::traits::Engine;
use std::time::Instant;

// Proves:
// x_(i+1) = ClippedReLU(A x_i + b)
// where ClippedReLU(x) = min(max(0, x), clip_max)
// and the activation is implemented by a lookup-table-style one-hot selection gadget.

fn main() {
    const N: usize = 2;

    println!("Nova affine+clipped-ReLU (lookup-table) demo: x_(i+1) = ClippedReLU(A x_i + b)");
    println!("=========================================================");

    let num_steps = 4;
    let num_iters_per_step = 2;
    let total_iters = num_steps * num_iters_per_step;

    // The table domain must contain every affine output.
    // clip_max is the clipping threshold after ReLU.
    let int_params = IntAffineClippedReluLookupParams::<N>::new(
        [[2, -1], [1, 2]],
        [1, -2],
        ClippedReluLookupTable::new(-32, 32, 8),
    );
    let field_params = int_params.to_field::<<E1 as Engine>::Scalar>();

    let x0 = [1i64, 2i64];

    println!("Preparing public parameters...");
    let start = Instant::now();
    let placeholder =
        build_affine_clipped_relu_lookup_placeholder_circuit(&field_params, num_iters_per_step);
    let pp = setup_affine_clipped_relu_lookup_public_params(&placeholder);
    println!("PublicParams::setup took {:?}", start.elapsed());
    println!(
        "Number of constraints per step (primary, secondary): {:?}",
        pp.num_constraints()
    );
    println!(
        "Number of variables per step (primary, secondary): {:?}",
        pp.num_variables()
    );

    println!("Generating affine+clipped-ReLU trace...");
    let (z0, trace) = generate_affine_clipped_relu_lookup_trace::<<E1 as Engine>::Scalar, N>(
        &int_params,
        x0,
        total_iters,
    );
    let circuits = build_affine_clipped_relu_lookup_step_circuits(
        &field_params,
        &trace,
        num_steps,
        num_iters_per_step,
    );

    println!(
        "Using clipped-ReLU lookup table over signed integer domain [{}, {}] ({} entries), clip_max = {}",
        field_params.clipped_relu_table.min,
        field_params.clipped_relu_table.max,
        field_params.clipped_relu_table.size(),
        field_params.clipped_relu_table.clip_max,
    );

    println!("Generating RecursiveSNARK...");
    let recursive_snark = run_affine_clipped_relu_lookup_recursive(&pp, &circuits, &z0);

    println!("Verifying RecursiveSNARK...");
    verify_affine_clipped_relu_lookup_recursive(&recursive_snark, &pp, num_steps, &z0);

    println!("Generating and verifying CompressedSNARK...");
    let proof_size =
        compress_affine_clipped_relu_lookup_and_verify(&pp, &recursive_snark, num_steps, &z0);
    println!("CompressedSNARK size: {} bytes", proof_size);
    println!("=========================================================");
}
