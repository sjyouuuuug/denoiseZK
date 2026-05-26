use denoise::{
    affine::{generate_affine_trace, AffineParams},
    nova_ivc::{
        build_placeholder_circuit, build_step_circuits, compress_and_verify, run_recursive,
        setup_public_params, verify_recursive, E1,
    },
};
use ff::Field;
use nova_snark::traits::Engine;
use std::time::Instant;

// proving the following affine relation for N=2:
// x_(i+1) = A x_i + b

fn main() {
    const N: usize = 2;

    println!("Nova affine demo: x_(i+1) = A x_i + b");
    println!("=========================================================");

    let num_steps = 5;
    let num_iters_per_step = 4;

    let two = <E1 as Engine>::Scalar::from(2u64);
    let three = <E1 as Engine>::Scalar::from(3u64);
    let four = <E1 as Engine>::Scalar::from(4u64);
    let five = <E1 as Engine>::Scalar::from(5u64);
    let seven = <E1 as Engine>::Scalar::from(7u64);
    let one = <E1 as Engine>::Scalar::ONE;

    let params = AffineParams::new([[two, three], [one, four]], [five, seven]);
    let x0 = [
        <E1 as Engine>::Scalar::from(1u64),
        <E1 as Engine>::Scalar::from(2u64),
    ];

    println!("Preparing public parameters...");
    let start = Instant::now();
    let placeholder = build_placeholder_circuit(&params, num_iters_per_step);
    let pp = setup_public_params(&placeholder);
    println!("PublicParams::setup took {:?}", start.elapsed());
    println!(
        "Number of constraints per step (primary, secondary): {:?}",
        pp.num_constraints()
    );
    println!(
        "Number of variables per step (primary, secondary): {:?}",
        pp.num_variables()
    );

    println!("Generating affine trace...");
    let total_iters = num_steps * num_iters_per_step;
    let (z0, trace) = generate_affine_trace(&params, x0, total_iters);
    let circuits = build_step_circuits(&params, &trace, num_steps, num_iters_per_step);

    println!("Generating RecursiveSNARK...");
    let recursive_snark = run_recursive(&pp, &circuits, &z0);

    println!("Verifying RecursiveSNARK...");
    verify_recursive(&recursive_snark, &pp, num_steps, &z0);

    println!("Generating and verifying CompressedSNARK...");
    let proof_size = compress_and_verify(&pp, &recursive_snark, num_steps, &z0);
    println!("CompressedSNARK size: {} bytes", proof_size);
    println!("=========================================================");
}
