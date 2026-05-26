use denoise::{
    denoise_fixed_point_conv::{
        build_fixed_point_denoise_conv_commitment_placeholder_circuit,
        build_fixed_point_denoise_conv_commitment_step_circuits,
        compress_fixed_point_denoise_conv_and_verify,
        generate_fixed_point_denoise_conv_trace_with_commitment,
        run_fixed_point_denoise_conv_recursive, setup_fixed_point_denoise_conv_public_params,
        verify_fixed_point_denoise_conv_recursive, FixedDenoiseConvPublicParams,
        FixedDenoiseConvStepParams,
    },
    denoise_fixed_point_time_embedding::generate_simple_time_table,
    fixed_point::{encode_f64_round, FixedPointConfig},
    layers::conv2d::{Conv2dPadding, Conv2dRealShape},
    models::denoise_update::{DenoiseUpdateMode, DenoiseUpdateWitness},
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

    println!("Nova fixed-point denoise Conv fused-update commitment demo");
    println!("Update mode: {:?}", DenoiseUpdateMode::FusedFloor);
    println!("public state: [x | y | h | C | t | params_queue | time_table]");
    println!("=========================================================");

    let config = FixedPointConfig::from_real_bounds(16, -4, 4, 2);
    let num_steps = 2;
    let num_iters_per_step = 2;
    let total_iters = num_steps * num_iters_per_step;
    let scale = config.scale;
    let padding = Conv2dPadding {
        top: 1,
        bottom: 1,
        left: 1,
        right: 1,
    };
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

    let real_shape = Conv2dRealShape::full::<IH, IW, KH, KW, OH, OW>();
    let public_params = FixedDenoiseConvPublicParams::new(
        params_seq,
        time_table.clone(),
        config.clone(),
        padding.clone(),
        real_shape.clone(),
        TE,
    )
    .with_update_mode(DenoiseUpdateMode::FusedFloor);
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

    let (z0, trace, hash_witnesses, expected_y, commitment) =
        generate_fixed_point_denoise_conv_trace_with_commitment::<
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
    println!("Commitment C = {:?}", commitment);
    println!("Expected public output y = {:?}", expected_y);
    if let Some(last) = trace.last() {
        println!("Final witness x_T = {:?}", last.x_i_plus_1_int);
    }
    println!("Public state length = {}", z0.len());

    let start = Instant::now();
    let mut placeholder = build_fixed_point_denoise_conv_commitment_placeholder_circuit::<
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
    );
    placeholder.update_mode = DenoiseUpdateMode::FusedFloor;
    for it in &mut placeholder.seq {
        it.update_witness = DenoiseUpdateWitness::zero_fused_floor();
    }
    let pp = setup_fixed_point_denoise_conv_public_params(&placeholder);
    println!("PublicParams::setup took {:?}", start.elapsed());
    println!("constraints: {:?}", pp.num_constraints());
    println!("variables: {:?}", pp.num_variables());

    let mut circuits = build_fixed_point_denoise_conv_commitment_step_circuits(
        &trace,
        &hash_witnesses,
        num_steps,
        num_iters_per_step,
        total_iters,
        config,
        padding,
        time_table,
        real_shape,
        TE,
    );
    for circuit in &mut circuits {
        circuit.update_mode = DenoiseUpdateMode::FusedFloor;
    }

    let recursive_snark = run_fixed_point_denoise_conv_recursive(&pp, &circuits, &z0);
    verify_fixed_point_denoise_conv_recursive(&recursive_snark, &pp, num_steps, &z0);
    let proof_size =
        compress_fixed_point_denoise_conv_and_verify(&pp, &recursive_snark, num_steps, &z0);
    println!("CompressedSNARK size: {} bytes", proof_size);
    println!("=========================================================");
}
