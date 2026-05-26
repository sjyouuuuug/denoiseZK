use crate::{
    denoise_fixed_point_conv::{
        build_fixed_point_denoise_conv_placeholder_circuit,
        build_fixed_point_denoise_conv_step_circuits, generate_fixed_point_denoise_conv_trace,
        FixedDenoiseConvPublicParams, FixedDenoiseConvStepParams,
    },
    denoise_fixed_point_time_embedding::{
        build_fixed_point_denoise_time_embedding_placeholder_circuit,
        build_fixed_point_denoise_time_embedding_step_circuits,
        generate_fixed_point_denoise_time_embedding_trace, generate_simple_time_table,
        FixedDenoiseTimeEmbeddingPublicParams, FixedDenoiseTimeEmbeddingStepParams,
        PublicFixedPointDenoiseTimeEmbeddingCircuit,
    },
    fixed_point::{
        encode_f64_round, set_signed_range_check_mode, FixedPointConfig, SignedRangeCheckMode,
    },
    layers::conv2d::{Conv2dPadding, Conv2dRealShape},
    models::denoise_update::{DenoiseUpdateMode, DenoiseUpdateWitness},
    nova_ivc::{E1, E2, F1, G1, S1, S2},
};
use ff::Field;
use flate2::{write::ZlibEncoder, Compression};
use nova_snark::{
    frontend::{num::AllocatedNum, shape_cs::ShapeCS, ConstraintSystem},
    nova::{CompressedSNARK, PublicParams, RecursiveSNARK},
    traits::{circuit::StepCircuit, snark::RelaxedR1CSSNARKTrait, Engine},
};
use std::{
    panic::{self, AssertUnwindSafe},
    time::Instant,
};

use super::schema::{ExperimentResult, ExperimentStatus, RangeMode, RunMode};

type MlpCircuit<const N: usize, const TE: usize, const IN: usize, const H: usize> =
    PublicFixedPointDenoiseTimeEmbeddingCircuit<G1, N, TE, IN, H>;

fn ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
}

#[derive(Clone, Debug)]
struct ProofStats {
    constraints: (usize, usize),
    variables: (usize, usize),
    proof_size_bytes: Option<usize>,
    recursive_prove_ms: Option<f64>,
    compressed_prove_ms: Option<f64>,
    compressed_verify_ms: Option<f64>,
    setup_ms: f64,
}

fn apply_range_mode(mode: RangeMode) {
    match mode {
        RangeMode::OneHot => set_signed_range_check_mode(SignedRangeCheckMode::OneHot),
        RangeMode::Bits => set_signed_range_check_mode(SignedRangeCheckMode::Bits),
    }
}

fn experiment_config(range_mode: RangeMode) -> FixedPointConfig {
    let config = FixedPointConfig::from_real_bounds(16, -4, 4, 2);
    match range_mode {
        RangeMode::OneHot => config.with_integer_ranges(-64, 63, -128, 127),
        RangeMode::Bits => config.with_integer_ranges(-256, 255, -128, 127),
    }
}

fn classify_error(message: String) -> (ExperimentStatus, String) {
    let lower = message.to_lowercase();
    let status = if lower.contains("outside")
        || lower.contains("range")
        || lower.contains("overflow")
        || lower.contains("table")
    {
        ExperimentStatus::Overflow
    } else {
        ExperimentStatus::Failed
    };
    (status, message)
}

fn panic_message(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else if let Some(s) = payload.downcast_ref::<&'static str>() {
        (*s).to_string()
    } else {
        "panic with non-string payload".to_string()
    }
}

fn execute_nova<C>(
    placeholder: &C,
    circuits: &[C],
    z0: &[F1],
    num_steps: usize,
    run_mode: RunMode,
) -> ProofStats
where
    C: StepCircuit<F1> + Clone + Send + Sync,
    CompressedSNARK<E1, E2, C, S1, S2>: serde::Serialize,
{
    if run_mode == RunMode::BuildOnly {
        let setup_start = Instant::now();
        let mut cs = ShapeCS::<E1>::new();
        let z = (0..placeholder.arity())
            .map(|i| {
                AllocatedNum::alloc_input(cs.namespace(|| format!("z_{i}")), || Ok(F1::ZERO))
                    .expect("failed to allocate build-only input")
            })
            .collect::<Vec<_>>();
        placeholder
            .synthesize(&mut cs, &z)
            .expect("build-only synthesis failed");
        return ProofStats {
            constraints: (cs.num_constraints(), 0),
            variables: (cs.num_inputs() + cs.num_aux(), 0),
            proof_size_bytes: None,
            recursive_prove_ms: None,
            compressed_prove_ms: None,
            compressed_verify_ms: None,
            setup_ms: ms(setup_start),
        };
    }

    let setup_start = Instant::now();
    let pp = PublicParams::<E1, E2, C>::setup(placeholder, &*S1::ck_floor(), &*S2::ck_floor())
        .expect("failed to setup public parameters");
    let setup_ms = ms(setup_start);
    let constraints = pp.num_constraints();
    let variables = pp.num_variables();

    assert!(
        !circuits.is_empty(),
        "circuits must not be empty for proof modes"
    );
    let recursive_start = Instant::now();
    let mut recursive_snark =
        RecursiveSNARK::<E1, E2, C>::new(&pp, &circuits[0], z0).expect("recursive init failed");
    for circuit in circuits {
        recursive_snark
            .prove_step(&pp, circuit)
            .expect("recursive prove_step failed");
    }
    recursive_snark
        .verify(&pp, num_steps, z0)
        .expect("recursive verify failed");
    let recursive_prove_ms = ms(recursive_start);

    if run_mode == RunMode::RecursiveOnly {
        return ProofStats {
            constraints,
            variables,
            proof_size_bytes: None,
            recursive_prove_ms: Some(recursive_prove_ms),
            compressed_prove_ms: None,
            compressed_verify_ms: None,
            setup_ms,
        };
    }

    let (pk, vk) = CompressedSNARK::<_, _, _, S1, S2>::setup(&pp).expect("compressed setup failed");
    let compressed_start = Instant::now();
    let compressed_snark = CompressedSNARK::<_, _, _, S1, S2>::prove(&pp, &pk, &recursive_snark)
        .expect("compressed prove failed");
    let compressed_prove_ms = ms(compressed_start);
    let mut encoder = ZlibEncoder::new(Vec::new(), Compression::default());
    bincode::serde::encode_into_std_write(
        &compressed_snark,
        &mut encoder,
        bincode::config::legacy(),
    )
    .expect("compressed proof serialization failed");
    let proof_size_bytes = encoder.finish().expect("proof compression failed").len();
    let verify_start = Instant::now();
    compressed_snark
        .verify(&vk, num_steps, z0)
        .expect("compressed verify failed");
    let compressed_verify_ms = ms(verify_start);

    ProofStats {
        constraints,
        variables,
        proof_size_bytes: Some(proof_size_bytes),
        recursive_prove_ms: Some(recursive_prove_ms),
        compressed_prove_ms: Some(compressed_prove_ms),
        compressed_verify_ms: Some(compressed_verify_ms),
        setup_ms,
    }
}

fn fill_stats(result: &mut ExperimentResult, stats: ProofStats) {
    result.primary_constraints = Some(stats.constraints.0);
    result.secondary_constraints = Some(stats.constraints.1);
    result.primary_variables = Some(stats.variables.0);
    result.secondary_variables = Some(stats.variables.1);
    result.proof_size_bytes = stats.proof_size_bytes;
    result.recursive_prove_ms = stats.recursive_prove_ms;
    result.compressed_prove_ms = stats.compressed_prove_ms;
    result.compressed_verify_ms = stats.compressed_verify_ms;
    result.setup_ms = Some(stats.setup_ms);
    result.set_status(if result.run_mode == "BuildOnly" {
        ExperimentStatus::BuildOk
    } else {
        ExperimentStatus::Ok
    });
}

fn x0_vector<const N: usize>(scale: i64) -> [i64; N] {
    std::array::from_fn(|i| match i % 8 {
        0 => encode_f64_round(0.75, scale),
        1 => encode_f64_round(-0.50, scale),
        2 => encode_f64_round(0.25, scale),
        3 => 0,
        4 => encode_f64_round(-0.25, scale),
        5 => encode_f64_round(0.50, scale),
        6 => encode_f64_round(-0.75, scale),
        _ => encode_f64_round(0.25, scale),
    })
}

fn schedule(t: usize) -> (f64, f64) {
    (0.875 - 0.005 * (t as f64), 0.125 + 0.005 * (t as f64))
}

fn mlp_params<const N: usize, const TE: usize, const IN: usize, const H: usize>(
    total_iters: usize,
    scale: i64,
) -> Vec<FixedDenoiseTimeEmbeddingStepParams<N, TE, IN, H>> {
    (0..total_iters)
        .map(|t| {
            let mut w1 = [[0.0f64; IN]; H];
            let mut b1 = [0.0f64; H];
            let mut w2 = [[0.0f64; H]; N];
            let b2 = [0.0f64; N];
            for r in 0..H {
                w1[r][r % N] = 0.125;
                if TE > 0 {
                    w1[r][N + (r % TE)] = if r % 2 == 0 { 0.0625 } else { -0.0625 };
                }
                b1[r] = if r % 3 == 0 { 0.0625 } else { 0.0 };
            }
            for r in 0..N {
                w2[r][r % H] = 0.125;
                if H > 1 {
                    w2[r][(r + 1) % H] = -0.0625;
                }
            }
            let (alpha, beta) = schedule(t);
            FixedDenoiseTimeEmbeddingStepParams::from_f64(w1, b1, w2, b2, alpha, beta, scale)
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
pub fn run_mlp_case<const N: usize, const TE: usize, const IN: usize, const H: usize>(
    case: &str,
    group: &str,
    update_mode: DenoiseUpdateMode,
    range_mode: RangeMode,
    run_mode: RunMode,
    total_iters: usize,
    num_steps: usize,
    num_iters_per_step: usize,
) -> ExperimentResult {
    let update_mode_s = format!("{update_mode:?}");
    let mut result = ExperimentResult::new(
        case,
        group,
        "MLP",
        &update_mode_s,
        N,
        Some(H),
        None,
        None,
        None,
        None,
        total_iters,
        num_steps,
        num_iters_per_step,
        16,
        range_mode,
        run_mode,
    );
    let run = panic::catch_unwind(AssertUnwindSafe(|| {
        assert_eq!(IN, N + TE);
        apply_range_mode(range_mode);
        let config = experiment_config(range_mode);
        let scale = config.scale;
        let params_seq = mlp_params::<N, TE, IN, H>(total_iters, scale);
        let time_table = generate_simple_time_table::<TE>(total_iters, scale);
        let public_params = FixedDenoiseTimeEmbeddingPublicParams::new(
            params_seq,
            time_table.clone(),
            config.clone(),
        )
        .with_update_mode(update_mode);
        let witness_start = Instant::now();
        let (z0, trace) = generate_fixed_point_denoise_time_embedding_trace::<
            <E1 as Engine>::Scalar,
            N,
            TE,
            IN,
            H,
        >(&public_params, x0_vector::<N>(scale));
        let witness_ms = ms(witness_start);
        let mut placeholder =
            build_fixed_point_denoise_time_embedding_placeholder_circuit::<N, TE, IN, H>(
                total_iters,
                num_iters_per_step,
                config.clone(),
                time_table.clone(),
            );
        placeholder.update_mode = update_mode;
        if update_mode == DenoiseUpdateMode::FusedFloor {
            for it in &mut placeholder.seq {
                it.update_witness = DenoiseUpdateWitness::zero_fused_floor();
            }
        }
        let mut circuits = build_fixed_point_denoise_time_embedding_step_circuits(
            &trace,
            num_steps,
            num_iters_per_step,
            total_iters,
            config,
            time_table,
        );
        for circuit in &mut circuits {
            circuit.update_mode = update_mode;
        }
        let stats = execute_nova::<MlpCircuit<N, TE, IN, H>>(
            &placeholder,
            &circuits,
            &z0,
            num_steps,
            run_mode,
        );
        result.witness_gen_ms = Some(witness_ms);
        fill_stats(&mut result, stats);
    }));
    if let Err(payload) = run {
        let (status, error) = classify_error(panic_message(payload));
        result.set_status(status);
        result.error = Some(error);
    }
    result
}

fn conv_params<const TE: usize, const KH: usize, const KW: usize>(
    total_iters: usize,
    scale: i64,
) -> Vec<FixedDenoiseConvStepParams<TE, KH, KW>> {
    (0..total_iters)
        .map(|t| {
            let mut kernel = [[0.0f64; KW]; KH];
            kernel[KH / 2][KW / 2] = 0.25;
            if KH > 1 {
                kernel[0][KW / 2] = 0.0625;
                kernel[KH - 1][KW / 2] = 0.0625;
            }
            if KW > 1 {
                kernel[KH / 2][0] = 0.0625;
                kernel[KH / 2][KW - 1] = 0.0625;
            }
            let mut time_w = [0.0f64; TE];
            if TE > 0 {
                time_w[0] = 0.0625;
            }
            if TE > 1 {
                time_w[1] = -0.0625;
            }
            let (alpha, beta) = schedule(t);
            FixedDenoiseConvStepParams::from_f64(kernel, 0.0, time_w, 0.0, alpha, beta, scale)
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
pub fn run_conv_case<
    const N: usize,
    const IH: usize,
    const IW: usize,
    const TE: usize,
    const KH: usize,
    const KW: usize,
    const OH: usize,
    const OW: usize,
>(
    case: &str,
    group: &str,
    update_mode: DenoiseUpdateMode,
    range_mode: RangeMode,
    run_mode: RunMode,
    total_iters: usize,
    num_steps: usize,
    num_iters_per_step: usize,
) -> ExperimentResult {
    let update_mode_s = format!("{update_mode:?}");
    let mut result = ExperimentResult::new(
        case,
        group,
        "Conv",
        &update_mode_s,
        N,
        None,
        Some(IH),
        Some(IW),
        Some(KH),
        Some(KW),
        total_iters,
        num_steps,
        num_iters_per_step,
        16,
        range_mode,
        run_mode,
    );
    let run = panic::catch_unwind(AssertUnwindSafe(|| {
        assert_eq!(N, IH * IW);
        assert_eq!(OH, IH);
        assert_eq!(OW, IW);
        apply_range_mode(range_mode);
        let config = experiment_config(range_mode);
        let scale = config.scale;
        let padding = Conv2dPadding {
            top: KH / 2,
            bottom: KH / 2,
            left: KW / 2,
            right: KW / 2,
        };
        let real_shape = Conv2dRealShape::full::<IH, IW, KH, KW, OH, OW>();
        let time_table = generate_simple_time_table::<TE>(total_iters, scale);
        let public_params = FixedDenoiseConvPublicParams::new(
            conv_params::<TE, KH, KW>(total_iters, scale),
            time_table.clone(),
            config.clone(),
            padding.clone(),
            real_shape.clone(),
            TE,
        )
        .with_update_mode(update_mode);
        let witness_start = Instant::now();
        let (z0, trace) = generate_fixed_point_denoise_conv_trace::<
            <E1 as Engine>::Scalar,
            N,
            IH,
            IW,
            TE,
            KH,
            KW,
            OH,
            OW,
        >(&public_params, x0_vector::<N>(scale));
        let witness_ms = ms(witness_start);
        let mut placeholder =
            build_fixed_point_denoise_conv_placeholder_circuit::<N, IH, IW, TE, KH, KW, OH, OW>(
                total_iters,
                num_iters_per_step,
                config.clone(),
                padding.clone(),
                time_table.clone(),
            );
        placeholder.update_mode = update_mode;
        if update_mode == DenoiseUpdateMode::FusedFloor {
            for it in &mut placeholder.seq {
                it.update_witness = DenoiseUpdateWitness::zero_fused_floor();
            }
        }
        let mut circuits = build_fixed_point_denoise_conv_step_circuits(
            &trace,
            num_steps,
            num_iters_per_step,
            total_iters,
            config,
            padding,
            time_table,
        );
        for circuit in &mut circuits {
            circuit.update_mode = update_mode;
        }
        let stats = execute_nova(&placeholder, &circuits, &z0, num_steps, run_mode);
        result.witness_gen_ms = Some(witness_ms);
        fill_stats(&mut result, stats);
    }));
    if let Err(payload) = run {
        let (status, error) = classify_error(panic_message(payload));
        result.set_status(status);
        result.error = Some(error);
    }
    result
}
