use denoise::{
    denoise_fixed_point_conv::{
        build_fixed_point_denoise_conv_commitment_placeholder_circuit,
        build_fixed_point_denoise_conv_commitment_step_circuits,
        generate_fixed_point_denoise_conv_trace_with_commitment, FixedDenoiseConvPublicParams,
        FixedDenoiseConvStepParams,
    },
    denoise_fixed_point_time_embedding::generate_simple_time_table,
    fixed_point::{encode_f64_round, FixedPointConfig},
    layers::conv2d::{Conv2dPadding, Conv2dRealShape},
    models::denoise_update::{DenoiseUpdateMode, DenoiseUpdateWitness},
    nova_ivc::{E1, E2, G1, S1, S2},
    visualization::{
        build_nova_step_views, reshape_flat_to_matrix, ComparisonMetric, ComparisonSeries,
        DemoConfig, DemoMetadata, PredictorSummary, ProofSummary, UpdateSummary, VisualDemo,
        VisualStep,
    },
};
use flate2::{write::ZlibEncoder, Compression};
use nova_snark::{
    nova::{CompressedSNARK, PublicParams, RecursiveSNARK},
    traits::{snark::RelaxedR1CSSNARKTrait, Engine},
};
use std::{fs, time::Instant};

type Circuit = denoise::denoise_fixed_point_conv::PublicFixedPointDenoiseConvCircuit<
    G1,
    16,
    4,
    4,
    2,
    3,
    3,
    4,
    4,
>;

fn elapsed_ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    const IH: usize = 4;
    const IW: usize = 4;
    const N: usize = IH * IW;
    const TE: usize = 2;
    const KH: usize = 3;
    const KW: usize = 3;
    const OH: usize = 4;
    const OW: usize = 4;

    let update_mode = DenoiseUpdateMode::FusedFloor;
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
        params_seq.clone(),
        time_table.clone(),
        config.clone(),
        padding.clone(),
        real_shape.clone(),
        TE,
    )
    .with_update_mode(update_mode);
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
    placeholder.update_mode = update_mode;
    for it in &mut placeholder.seq {
        it.update_witness = DenoiseUpdateWitness::zero_fused_floor();
    }

    let pp =
        PublicParams::<E1, E2, Circuit>::setup(&placeholder, &*S1::ck_floor(), &*S2::ck_floor())?;
    let constraints = pp.num_constraints();
    let variables = pp.num_variables();

    let mut circuits = build_fixed_point_denoise_conv_commitment_step_circuits(
        &trace,
        &hash_witnesses,
        num_steps,
        num_iters_per_step,
        total_iters,
        config.clone(),
        padding,
        time_table,
        real_shape,
        TE,
    );
    for circuit in &mut circuits {
        circuit.update_mode = update_mode;
    }

    let recursive_start = Instant::now();
    let mut recursive_snark = RecursiveSNARK::<E1, E2, Circuit>::new(&pp, &circuits[0], &z0)?;
    for circuit in &circuits {
        recursive_snark.prove_step(&pp, circuit)?;
    }
    let recursive_prove_ms = elapsed_ms(recursive_start);

    let verify_start = Instant::now();
    let recursive_verified = recursive_snark.verify(&pp, num_steps, &z0).is_ok();
    let recursive_verify_ms = elapsed_ms(verify_start);

    let (pk, vk) = CompressedSNARK::<_, _, _, S1, S2>::setup(&pp)?;
    let compressed_start = Instant::now();
    let compressed_snark = CompressedSNARK::<_, _, _, S1, S2>::prove(&pp, &pk, &recursive_snark)?;
    let compressed_prove_ms = elapsed_ms(compressed_start);

    let mut encoder = ZlibEncoder::new(Vec::new(), Compression::default());
    bincode::serde::encode_into_std_write(
        &compressed_snark,
        &mut encoder,
        bincode::config::legacy(),
    )?;
    let proof_size_bytes = encoder.finish()?.len();

    let compressed_verify_start = Instant::now();
    let compressed_verified = compressed_snark.verify(&vk, num_steps, &z0).is_ok();
    let compressed_verify_ms = elapsed_ms(compressed_verify_start);

    let mut trajectory = Vec::with_capacity(trace.len());
    for (t, it) in trace.iter().enumerate() {
        let (alpha_term, beta_term, fused_raw) = match &it.update_witness {
            DenoiseUpdateWitness::DoubleFloor {
                alpha_q, beta_q, ..
            } => (
                reshape_flat_to_matrix(alpha_q, IH, IW),
                reshape_flat_to_matrix(beta_q, IH, IW),
                None,
            ),
            DenoiseUpdateWitness::FusedFloor { fused_raw, .. } => {
                let alpha_raw: Vec<i64> = it
                    .x_i_int
                    .iter()
                    .map(|&x| params_seq[t].alpha * x)
                    .collect();
                let beta_raw: Vec<i64> = it
                    .epsilon_int
                    .iter()
                    .map(|&epsilon| params_seq[t].beta * epsilon)
                    .collect();
                (
                    reshape_flat_to_matrix(&alpha_raw, IH, IW),
                    reshape_flat_to_matrix(&beta_raw, IH, IW),
                    Some(reshape_flat_to_matrix(fused_raw, IH, IW)),
                )
            }
        };
        trajectory.push(VisualStep {
            t,
            x: reshape_flat_to_matrix(&it.x_i_int, IH, IW),
            epsilon: reshape_flat_to_matrix(&it.epsilon_int, IH, IW),
            x_next: reshape_flat_to_matrix(&it.x_i_plus_1_int, IH, IW),
            time_embedding: it.time_emb_int.to_vec(),
            alpha: params_seq[t].alpha,
            beta: params_seq[t].beta,
            predictor: PredictorSummary {
                backend: "Conv".to_string(),
                hidden: None,
                conv_raw: Some(reshape_flat_to_matrix(
                    &it.conv_witness
                        .raw
                        .iter()
                        .flat_map(|row| row.iter().copied())
                        .collect::<Vec<_>>(),
                    OH,
                    OW,
                )),
                conv_pre_activation: Some(reshape_flat_to_matrix(
                    &it.conv_witness
                        .output
                        .iter()
                        .flat_map(|row| row.iter().copied())
                        .collect::<Vec<_>>(),
                    OH,
                    OW,
                )),
                relu_output: Some(reshape_flat_to_matrix(&it.epsilon_int, IH, IW)),
            },
            update: UpdateSummary {
                mode: format!("{update_mode:?}"),
                alpha_term,
                beta_term,
                fused_raw,
                output: reshape_flat_to_matrix(&it.x_i_plus_1_int, IH, IW),
            },
        });
    }

    let dense_mul = (IH * IW * OH * OW) as f64;
    let sparse_conv_mul = (OH * OW * KH * KW) as f64;
    let double_rescales = (2 * N * total_iters) as f64;
    let fused_rescales = (N * total_iters) as f64;
    let demo = VisualDemo {
        metadata: DemoMetadata {
            title: "Nova-based Verifiable Denoising Demo".to_string(),
            backend: "Conv".to_string(),
            update_mode: format!("{update_mode:?}"),
            description: format!(
                "A 4x4 fixed-point Conv denoise trajectory with time embedding, toy model commitment C={commitment:?}, and Nova recursive proof summary."
            ),
        },
        config: DemoConfig {
            scale,
            total_iters,
            num_steps,
            num_iters_per_step,
            state_height: IH,
            state_width: IW,
            time_embedding_dim: TE,
            lookup_range: format!("[{},{}]", config.relu_min, config.relu_max),
            clip_max: config.clip_max,
        },
        trajectory,
        nova_steps: build_nova_step_views(num_steps, num_iters_per_step),
        proof: ProofSummary {
            recursive_verified,
            compressed_verified,
            primary_constraints: constraints.0,
            secondary_constraints: constraints.1,
            primary_variables: variables.0,
            secondary_variables: variables.1,
            recursive_prove_ms,
            recursive_verify_ms,
            compressed_prove_ms,
            compressed_verify_ms,
            proof_size_bytes,
        },
        comparisons: vec![
            ComparisonSeries {
                name: "Update mode rescale count".to_string(),
                metrics: vec![
                    ComparisonMetric {
                        label: "DoubleFloor".to_string(),
                        value: double_rescales,
                        unit: "rescale".to_string(),
                    },
                    ComparisonMetric {
                        label: "FusedFloor".to_string(),
                        value: fused_rescales,
                        unit: "rescale".to_string(),
                    },
                ],
            },
            ComparisonSeries {
                name: "Conv sparse Toeplitz complexity".to_string(),
                metrics: vec![
                    ComparisonMetric {
                        label: "Dense equivalent".to_string(),
                        value: dense_mul,
                        unit: "mul".to_string(),
                    },
                    ComparisonMetric {
                        label: "Sparse Conv".to_string(),
                        value: sparse_conv_mul,
                        unit: "mul".to_string(),
                    },
                ],
            },
            ComparisonSeries {
                name: "Observed constraints".to_string(),
                metrics: vec![ComparisonMetric {
                    label: "Fused Conv commitment".to_string(),
                    value: constraints.0 as f64,
                    unit: "primary constraints".to_string(),
                }],
            },
        ],
    };

    fs::create_dir_all("outputs/visual_demo")?;
    let path = "outputs/visual_demo/denoise_demo.json";
    fs::write(path, serde_json::to_string_pretty(&demo)?)?;
    println!("wrote {path}");
    println!("expected final output y = {:?}", expected_y);
    println!("proof size = {proof_size_bytes} bytes");

    Ok(())
}
