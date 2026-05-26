use denoise::{
    denoise_fixed_point_time_embedding::{
        compress_fixed_point_denoise_time_embedding_and_verify,
        generate_fixed_point_denoise_time_embedding_trace, generate_simple_time_table,
        pad_denoise_time_embedding_step_params, pad_time_table_vec,
        run_fixed_point_denoise_time_embedding_recursive,
        setup_fixed_point_denoise_time_embedding_public_params,
        verify_fixed_point_denoise_time_embedding_recursive, FixedDenoiseTimeEmbeddingPublicParams,
        FixedDenoiseTimeEmbeddingStepParams,
    },
    denoise_fixed_point_time_embedding_padded::{
        build_padded_denoise_time_embedding_placeholder_circuit,
        build_padded_denoise_time_embedding_step_circuits, PaddedDenoiseShape,
    },
    fixed_point::{encode_f64_round, FixedPointConfig},
    nova_ivc::E1,
    padding::{pad_vector_i64, slice_real_vector},
};
use nova_snark::traits::Engine;
use std::time::Instant;

fn main() {
    const N_REAL: usize = 2;
    const TE_REAL: usize = 2;
    const IN_REAL: usize = 4;
    const H_REAL: usize = 3;
    const N_MAX: usize = 4;
    const TE_MAX: usize = 4;
    const IN_MAX: usize = 8;
    const H_MAX: usize = 4;

    println!("Nova fixed-point denoise + time embedding padded demo");
    println!("real dims: N={N_REAL}, TE={TE_REAL}, IN={IN_REAL}, H={H_REAL}");
    println!("max dims:  N={N_MAX}, TE={TE_MAX}, IN={IN_MAX}, H={H_MAX}");
    println!("=========================================================");

    let config = FixedPointConfig::from_real_bounds(16, -4, 4, 2);
    let num_steps = 2;
    let num_iters_per_step = 2;
    let total_iters = num_steps * num_iters_per_step;
    let scale = config.scale;
    let shape = PaddedDenoiseShape::new(
        N_REAL, TE_REAL, IN_REAL, H_REAL, N_MAX, TE_MAX, IN_MAX, H_MAX,
    );

    let real_time_table = generate_simple_time_table::<TE_REAL>(total_iters, scale);
    let padded_time_table = pad_time_table_vec::<TE_REAL, TE_MAX>(real_time_table);

    let mut params_seq = Vec::with_capacity(total_iters);
    for t in 0..total_iters {
        let alpha = 0.875 - 0.025 * (t as f64);
        let beta = 0.125 + 0.025 * (t as f64);
        let real =
            FixedDenoiseTimeEmbeddingStepParams::<N_REAL, TE_REAL, IN_REAL, H_REAL>::from_f64(
                [
                    [0.50, 0.00, 0.125, 0.00],
                    [0.00, 0.50, 0.00, 0.125],
                    [-0.25, 0.25, 0.125, -0.125],
                ],
                [0.00, 0.125, 0.00],
                [[0.50, 0.00, 0.00], [0.00, 0.50, 0.00]],
                [0.00, 0.00],
                alpha,
                beta,
                scale,
            );
        params_seq.push(pad_denoise_time_embedding_step_params::<
            N_REAL,
            TE_REAL,
            IN_REAL,
            H_REAL,
            N_MAX,
            TE_MAX,
            IN_MAX,
            H_MAX,
        >(real));
    }

    let x0_real = [encode_f64_round(1.0, scale), encode_f64_round(-0.5, scale)];
    let x0_padded = pad_vector_i64::<N_REAL, N_MAX>(x0_real);
    let public_params = FixedDenoiseTimeEmbeddingPublicParams::new(
        params_seq,
        padded_time_table.clone(),
        config.clone(),
    );

    let start = Instant::now();
    let placeholder =
        build_padded_denoise_time_embedding_placeholder_circuit::<N_MAX, TE_MAX, IN_MAX, H_MAX>(
            total_iters,
            num_iters_per_step,
            config.clone(),
            padded_time_table.clone(),
            shape,
        );
    let pp = setup_fixed_point_denoise_time_embedding_public_params(&placeholder);
    println!("PublicParams::setup took {:?}", start.elapsed());
    println!("constraints: {:?}", pp.num_constraints());
    println!("variables: {:?}", pp.num_variables());

    let (z0, trace) = generate_fixed_point_denoise_time_embedding_trace::<
        <E1 as Engine>::Scalar,
        N_MAX,
        TE_MAX,
        IN_MAX,
        H_MAX,
    >(&public_params, x0_padded);
    println!("Public state length = {}", z0.len());
    if let Some(last) = trace.last() {
        println!("Final padded x_T = {:?}", last.x_i_plus_1_int);
        println!(
            "Final real x_T = {:?}",
            slice_real_vector::<N_REAL, N_MAX>(&last.x_i_plus_1_int)
        );
    }

    let circuits = build_padded_denoise_time_embedding_step_circuits(
        &trace,
        num_steps,
        num_iters_per_step,
        total_iters,
        config,
        padded_time_table,
        shape,
    );
    let recursive_snark = run_fixed_point_denoise_time_embedding_recursive(&pp, &circuits, &z0);
    verify_fixed_point_denoise_time_embedding_recursive(&recursive_snark, &pp, num_steps, &z0);
    let proof_size = compress_fixed_point_denoise_time_embedding_and_verify(
        &pp,
        &recursive_snark,
        num_steps,
        &z0,
    );
    println!("CompressedSNARK size: {} bytes", proof_size);
    println!("=========================================================");
}
