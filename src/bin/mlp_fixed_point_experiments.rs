use denoise::{
    affine::fixed_point::apply_affine_fixed_point_with_witness,
    fixed_point::{encode_f64_round, FixedPointConfig},
    mlp_fixed_point_clipped_relu_lookup::{
        build_fixed_point_mlp_placeholder_circuit, build_fixed_point_mlp_step_circuits,
        generate_fixed_point_mlp_trace, FixedMlpClippedReluPublicParams,
        FixedMlpClippedReluStepParams, PublicFixedPointMlpClippedReluCircuit,
    },
    nova_ivc::{E1, E2, F1, S1, S2},
};
use flate2::{write::ZlibEncoder, Compression};
use nova_snark::{
    nova::{CompressedSNARK, PublicParams, RecursiveSNARK},
    traits::snark::RelaxedR1CSSNARKTrait,
};
use std::{panic, time::Instant};

fn make_params<const N: usize, const H: usize>(
    total_iters: usize,
    scale: i64,
    weight: f64,
    bias: f64,
) -> Vec<FixedMlpClippedReluStepParams<N, H>> {
    (0..total_iters)
        .map(|step_idx| {
            let mut w1 = [[0.0; N]; H];
            let mut b1 = [0.0; H];
            let mut w2 = [[0.0; H]; N];
            let mut b2 = [0.0; N];

            for r in 0..H {
                b1[r] = if (step_idx + r) % 3 == 0 { bias } else { 0.0 };
                for c in 0..N {
                    let sign = if (step_idx + r + c) % 2 == 0 {
                        1.0
                    } else {
                        -1.0
                    };
                    w1[r][c] = sign * weight / (N as f64);
                }
            }

            for r in 0..N {
                b2[r] = if (step_idx + r) % 2 == 0 { bias } else { -bias };
                for c in 0..H {
                    let sign = if (step_idx + r + c) % 2 == 0 {
                        1.0
                    } else {
                        0.5
                    };
                    w2[r][c] = sign * weight / (H as f64);
                }
            }

            FixedMlpClippedReluStepParams::from_f64(w1, b1, w2, b2, scale)
        })
        .collect()
}

fn make_x0<const N: usize>(scale: i64) -> [i64; N] {
    let mut x0 = [0i64; N];
    for (i, x) in x0.iter_mut().enumerate() {
        let value = if i % 2 == 0 { 1.0 } else { -0.5 };
        *x = encode_f64_round(value, scale);
    }
    x0
}

fn find_lookup_overflow<const N: usize, const H: usize>(
    public_params: &FixedMlpClippedReluPublicParams<N, H>,
    x0: [i64; N],
) -> Option<String> {
    let table = public_params.config.clipped_relu_table();
    let mut x_i = x0;

    for (step_idx, step) in public_params.params_seq.iter().enumerate() {
        let (_, _, _, hidden_affine) = apply_affine_fixed_point_with_witness(
            &step.w1,
            &step.b1,
            &x_i,
            public_params.config.scale,
        );

        for (hidden_idx, value) in hidden_affine.iter().enumerate() {
            if !table.contains(*value) {
                return Some(format!(
                    "step={step_idx}, hidden={hidden_idx}, value={value}, range=[{},{}]",
                    table.min, table.max
                ));
            }
        }

        let mut hidden_act = [0i64; H];
        for r in 0..H {
            hidden_act[r] = table.clipped_relu(hidden_affine[r]);
        }
        let (_, _, _, x_next) = apply_affine_fixed_point_with_witness(
            &step.w2,
            &step.b2,
            &hidden_act,
            public_params.config.scale,
        );
        x_i = x_next;
    }

    None
}

fn compressed_proof_size<C>(compressed_snark: &CompressedSNARK<E1, E2, C, S1, S2>) -> usize
where
    C: nova_snark::traits::circuit::StepCircuit<F1> + Clone + Send + Sync,
    CompressedSNARK<E1, E2, C, S1, S2>: serde::Serialize,
{
    let mut encoder = ZlibEncoder::new(Vec::new(), Compression::default());
    bincode::serde::encode_into_std_write(
        compressed_snark,
        &mut encoder,
        bincode::config::legacy(),
    )
    .expect("failed to serialize compressed SNARK");
    encoder
        .finish()
        .expect("failed to finish compression")
        .len()
}

fn run_case<const N: usize, const H: usize>(
    name: &str,
    num_steps: usize,
    num_iters_per_step: usize,
    config: FixedPointConfig,
    weight: f64,
    bias: f64,
) {
    let total_iters = num_steps * num_iters_per_step;
    let params_seq = make_params::<N, H>(total_iters, config.scale, weight, bias);
    let public_params = FixedMlpClippedReluPublicParams::new(params_seq, config.clone());
    let x0 = make_x0::<N>(config.scale);

    if let Some(detail) = find_lookup_overflow(&public_params, x0) {
        println!(
            "{name},{N},{H},{num_steps},{num_iters_per_step},{total_iters},{},{},{},OVERFLOW,{detail}",
            config.scale, config.relu_min, config.relu_max
        );
        return;
    }

    let result = panic::catch_unwind(|| {
        let placeholder = build_fixed_point_mlp_placeholder_circuit::<N, H>(
            total_iters,
            num_iters_per_step,
            config.clone(),
        );
        let setup_start = Instant::now();
        let pp = PublicParams::<E1, E2, PublicFixedPointMlpClippedReluCircuit<_, N, H>>::setup(
            &placeholder,
            &*S1::ck_floor(),
            &*S2::ck_floor(),
        )
        .expect("failed to setup public parameters");
        let setup_ms = setup_start.elapsed().as_secs_f64() * 1000.0;
        let constraints = pp.num_constraints();
        let variables = pp.num_variables();

        let (z0, trace) = generate_fixed_point_mlp_trace::<F1, N, H>(&public_params, x0);
        let circuits = build_fixed_point_mlp_step_circuits(
            &trace,
            num_steps,
            num_iters_per_step,
            total_iters,
            config.clone(),
        );

        let recursive_start = Instant::now();
        let mut recursive_snark = RecursiveSNARK::<
            E1,
            E2,
            PublicFixedPointMlpClippedReluCircuit<_, N, H>,
        >::new(&pp, &circuits[0], &z0)
        .expect("failed to initialize recursive SNARK");
        for circuit in &circuits {
            recursive_snark
                .prove_step(&pp, circuit)
                .expect("failed to prove recursive step");
        }
        let recursive_prove_ms = recursive_start.elapsed().as_secs_f64() * 1000.0;
        recursive_snark
            .verify(&pp, num_steps, &z0)
            .expect("recursive verification failed");

        let (pk, vk) = CompressedSNARK::<_, _, _, S1, S2>::setup(&pp)
            .expect("failed to setup compressed SNARK keys");
        let prove_start = Instant::now();
        let compressed_snark =
            CompressedSNARK::<_, _, _, S1, S2>::prove(&pp, &pk, &recursive_snark)
                .expect("failed to produce compressed SNARK");
        let compressed_prove_ms = prove_start.elapsed().as_secs_f64() * 1000.0;
        let proof_size = compressed_proof_size(&compressed_snark);

        let verify_start = Instant::now();
        compressed_snark
            .verify(&vk, num_steps, &z0)
            .expect("compressed verification failed");
        let compressed_verify_ms = verify_start.elapsed().as_secs_f64() * 1000.0;

        println!(
            "{name},{N},{H},{num_steps},{num_iters_per_step},{total_iters},{},{},{},OK,{},{},{},{},{:.3},{:.3},{:.3},{:.3}",
            config.scale,
            config.relu_min,
            config.relu_max,
            constraints.0,
            constraints.1,
            variables.0,
            variables.1,
            proof_size,
            recursive_prove_ms,
            compressed_prove_ms,
            compressed_verify_ms
        );

        setup_ms
    });

    if let Err(payload) = result {
        let message = payload
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| payload.downcast_ref::<&str>().copied())
            .unwrap_or("unknown panic");
        println!(
            "{name},{N},{H},{num_steps},{num_iters_per_step},{total_iters},{},{},{},PANIC,{message}",
            config.scale, config.relu_min, config.relu_max
        );
    }
}

fn main() {
    println!("case,N,H,num_steps,num_iters_per_step,total_iters,scale,relu_min,relu_max,status,primary_constraints,secondary_constraints,primary_variables,secondary_variables,proof_size_bytes,recursive_prove_ms,compressed_prove_ms,compressed_verify_ms");

    run_case::<1, 1>(
        "tiny_1x1_t1",
        1,
        1,
        FixedPointConfig::from_real_bounds(16, -4, 4, 2),
        0.50,
        0.00,
    );
    run_case::<2, 2>(
        "small_2x2_t2",
        1,
        2,
        FixedPointConfig::from_real_bounds(16, -4, 4, 2),
        0.50,
        0.125,
    );
    run_case::<2, 3>(
        "demo_2x3_t6",
        3,
        2,
        FixedPointConfig::from_real_bounds(16, -4, 4, 2),
        0.50,
        0.125,
    );
    run_case::<3, 3>(
        "wide_3x3_t4",
        2,
        2,
        FixedPointConfig::from_real_bounds(16, -4, 4, 2),
        0.45,
        0.125,
    );
    run_case::<4, 4>(
        "wide_4x4_t2",
        1,
        2,
        FixedPointConfig::from_real_bounds(16, -4, 4, 2),
        0.40,
        0.125,
    );
    run_case::<2, 2>(
        "overflow_narrow_range",
        1,
        1,
        FixedPointConfig::from_real_bounds(16, -1, 1, 1),
        2.00,
        1.00,
    );
}
