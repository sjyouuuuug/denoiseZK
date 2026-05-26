use denoise::{
    denoise_fixed_point_conv::{
        build_fixed_point_denoise_conv_placeholder_circuit_with_shape,
        build_fixed_point_denoise_conv_step_circuits_with_shape,
        compress_fixed_point_denoise_conv_and_verify,
        generate_fixed_point_denoise_conv_trace_with_computed_output,
        run_fixed_point_denoise_conv_recursive, setup_fixed_point_denoise_conv_public_params,
        verify_fixed_point_denoise_conv_recursive, FixedDenoiseConvPublicParams,
        FixedDenoiseConvStepParams,
    },
    denoise_fixed_point_time_embedding::generate_simple_time_table,
    fixed_point::{encode_f64_round, FixedPointConfig},
    layers::conv2d::{Conv2dPadding, Conv2dRealShape},
    nova_ivc::E1,
};
use nova_snark::traits::Engine;
use std::time::Instant;

fn main() {
    const IH: usize = 4;
    const IW: usize = 4;
    const N: usize = IH * IW;
    const TE: usize = 2;
    const KH: usize = 3;
    const KW: usize = 3;
    const OH: usize = 4;
    const OW: usize = 4;

    println!("Nova fixed-point denoise + conv backend + public output binding demo");
    println!("z = [x | y | t | params_queue | time_table]");
    println!("=========================================================");

    let config = FixedPointConfig::from_real_bounds(16, -4, 4, 2);
    let num_steps = 2;
    let num_iters_per_step = 2;
    let total_iters = num_steps * num_iters_per_step;
    let scale = config.scale;
    let padding = Conv2dPadding::same_3x3();
    let real_shape = Conv2dRealShape::full::<IH, IW, KH, KW, OH, OW>();
    let time_table = generate_simple_time_table::<TE>(total_iters, scale);

    let mut params_seq = Vec::with_capacity(total_iters);
    for t in 0..total_iters {
        let alpha = 0.875 - 0.025 * (t as f64);
        let beta = 0.125 + 0.025 * (t as f64);
        params_seq.push(FixedDenoiseConvStepParams::<TE, KH, KW>::from_f64(
            [[0.0, 0.125, 0.0], [0.125, 0.25, 0.125], [0.0, 0.125, 0.0]],
            0.0,
            [0.125, -0.125],
            0.0,
            alpha,
            beta,
            scale,
        ));
    }

    let public_params = FixedDenoiseConvPublicParams::new(
        params_seq,
        time_table.clone(),
        config.clone(),
        padding.clone(),
        real_shape.clone(),
        TE,
    );
    let x0 = [
        encode_f64_round(1.0, scale),
        encode_f64_round(-0.5, scale),
        encode_f64_round(0.5, scale),
        encode_f64_round(0.0, scale),
        encode_f64_round(-0.25, scale),
        encode_f64_round(0.75, scale),
        encode_f64_round(-0.5, scale),
        encode_f64_round(0.25, scale),
        encode_f64_round(0.5, scale),
        encode_f64_round(0.0, scale),
        encode_f64_round(1.0, scale),
        encode_f64_round(-0.25, scale),
        encode_f64_round(0.0, scale),
        encode_f64_round(0.25, scale),
        encode_f64_round(-0.75, scale),
        encode_f64_round(0.5, scale),
    ];

    println!("Preparing public parameters...");
    let start = Instant::now();
    let placeholder = build_fixed_point_denoise_conv_placeholder_circuit_with_shape::<
        N,
        IH,
        IW,
        TE,
        KH,
        KW,
        OH,
        OW,
    >(
        total_iters,
        num_iters_per_step,
        config.clone(),
        padding.clone(),
        time_table.clone(),
        real_shape.clone(),
        TE,
        true,
    );
    let pp = setup_fixed_point_denoise_conv_public_params(&placeholder);
    println!("PublicParams::setup took {:?}", start.elapsed());
    println!(
        "Number of constraints per step (primary, secondary): {:?}",
        pp.num_constraints()
    );
    println!(
        "Number of variables per step (primary, secondary): {:?}",
        pp.num_variables()
    );

    println!("Generating fixed-point denoise conv trace with public output...");
    let (z0, trace, expected_output) = generate_fixed_point_denoise_conv_trace_with_computed_output::<
        <E1 as Engine>::Scalar,
        N,
        IH,
        IW,
        TE,
        KH,
        KW,
        OH,
        OW,
    >(&public_params, x0);
    let final_witness = trace.last().unwrap().x_i_plus_1_int;
    assert_eq!(expected_output, final_witness);
    println!("Public state length = {}", z0.len());
    println!("Expected public output y = {:?}", expected_output);
    println!("Final fixed-point x_T witness = {:?}", final_witness);

    let circuits = build_fixed_point_denoise_conv_step_circuits_with_shape(
        &trace,
        num_steps,
        num_iters_per_step,
        total_iters,
        config,
        padding,
        time_table,
        real_shape,
        TE,
        true,
    );

    println!("Generating RecursiveSNARK...");
    let recursive_snark = run_fixed_point_denoise_conv_recursive(&pp, &circuits, &z0);

    println!("Verifying RecursiveSNARK...");
    verify_fixed_point_denoise_conv_recursive(&recursive_snark, &pp, num_steps, &z0);

    println!("Generating and verifying CompressedSNARK...");
    let proof_size =
        compress_fixed_point_denoise_conv_and_verify(&pp, &recursive_snark, num_steps, &z0);
    println!("CompressedSNARK size: {} bytes", proof_size);
    println!("=========================================================");
}
