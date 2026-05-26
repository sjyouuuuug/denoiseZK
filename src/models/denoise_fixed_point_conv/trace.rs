use ff::PrimeField;

use crate::{
    activations::clipped_relu::field_from_i64,
    commitment::{toy_hash_block_prefixes_i64_from_field, TOY_HASH_BASE_U64},
    fixed_point::{rescale_with_remainder, FixedPointConfig},
    layers::conv2d::{
        apply_conv2d_fixed_point_with_witness, is_real_input_coord, is_real_kernel_coord,
        is_real_output_coord, Conv2dFixedPointWitness, Conv2dRealShape,
    },
    public_state::CommittedDenoiseStateLayout,
};

use super::params::FixedDenoiseConvPublicParams;
use crate::models::denoise_update::{compute_denoise_update_witness, DenoiseUpdateWitness};

#[derive(Clone, Debug)]
pub struct FixedDenoiseConvIteration<
    F: PrimeField + Copy,
    const N: usize,
    const IH: usize,
    const IW: usize,
    const TE: usize,
    const KH: usize,
    const KW: usize,
    const OH: usize,
    const OW: usize,
> {
    pub t_int: i64,
    pub time_emb_int: [i64; TE],
    pub x_i_int: [i64; N],
    pub time_raw_int: i64,
    pub time_quotient_int: i64,
    pub time_remainder_int: i64,
    pub time_bias_int: i64,
    pub conv_witness: Conv2dFixedPointWitness<OH, OW>,
    pub epsilon_int: [i64; N],
    pub alpha_x_int: [i64; N],
    pub alpha_remainder_int: [i64; N],
    pub beta_epsilon_int: [i64; N],
    pub beta_remainder_int: [i64; N],
    pub update_witness: DenoiseUpdateWitness<N>,
    pub x_i_plus_1_int: [i64; N],
    pub epsilon: [F; N],
    pub x_i_plus_1: [F; N],
}

fn assert_in_range(name: &str, value: i64, min: i64, max: i64) {
    assert!(
        min <= value && value <= max,
        "{name}={value} outside range [{min},{max}]"
    );
}

pub fn assert_flat_image_padding_zero<const IH: usize, const IW: usize>(
    flat: &[i64],
    shape: &Conv2dRealShape,
    label: &str,
) {
    assert_eq!(flat.len(), IH * IW, "{label} length must equal IH*IW");
    for row in 0..IH {
        for col in 0..IW {
            if !is_real_input_coord(row, col, shape) {
                let idx = row * IW + col;
                assert_eq!(
                    flat[idx], 0,
                    "{label} spatial padding at row={row}, col={col}, idx={idx} must be zero, got {}",
                    flat[idx]
                );
            }
        }
    }
}

pub fn assert_flat_output_padding_zero<const OH: usize, const OW: usize>(
    flat: &[i64],
    shape: &Conv2dRealShape,
    label: &str,
) {
    assert_eq!(flat.len(), OH * OW, "{label} length must equal OH*OW");
    for row in 0..OH {
        for col in 0..OW {
            if !is_real_output_coord(row, col, shape) {
                let idx = row * OW + col;
                assert_eq!(
                    flat[idx], 0,
                    "{label} output padding at row={row}, col={col}, idx={idx} must be zero, got {}",
                    flat[idx]
                );
            }
        }
    }
}

pub fn assert_kernel_padding_zero<const KH: usize, const KW: usize>(
    kernel: &[[i64; KW]; KH],
    shape: &Conv2dRealShape,
) {
    for row in 0..KH {
        for col in 0..KW {
            if !is_real_kernel_coord(row, col, shape) {
                assert_eq!(
                    kernel[row][col], 0,
                    "kernel padding at row={row}, col={col} must be zero, got {}",
                    kernel[row][col]
                );
            }
        }
    }
}

fn check_conv_ranges<const OH: usize, const OW: usize>(
    step: usize,
    witness: &Conv2dFixedPointWitness<OH, OW>,
    config: &FixedPointConfig,
) {
    for oy in 0..OH {
        for ox in 0..OW {
            assert_in_range(
                &format!("step={step} conv_q[{oy},{ox}]"),
                witness.quotient[oy][ox],
                config.quotient_min,
                config.quotient_max,
            );
            assert_in_range(
                &format!("step={step} conv_pre[{oy},{ox}]"),
                witness.output[oy][ox],
                config.relu_min,
                config.relu_max,
            );
        }
    }
}

fn build_z0<F: PrimeField, const N: usize, const TE: usize, const KH: usize, const KW: usize>(
    public_params: &FixedDenoiseConvPublicParams<TE, KH, KW>,
    x0: &[i64; N],
    expected_output: Option<&[i64; N]>,
) -> Vec<F> {
    let mut z0: Vec<F> = x0.iter().map(|&v| field_from_i64::<F>(v)).collect();
    if let Some(y) = expected_output {
        z0.extend(y.iter().map(|&v| field_from_i64::<F>(v)));
    }
    z0.push(F::ZERO);
    for step in &public_params.params_seq {
        z0.extend(step.flatten_field::<F>());
    }
    for row in &public_params.time_table {
        for &value in row {
            z0.push(field_from_i64::<F>(value));
        }
    }
    z0
}

fn simulate_fixed_point_denoise_conv_trace<
    F: PrimeField + Copy,
    const N: usize,
    const IH: usize,
    const IW: usize,
    const TE: usize,
    const KH: usize,
    const KW: usize,
    const OH: usize,
    const OW: usize,
>(
    public_params: &FixedDenoiseConvPublicParams<TE, KH, KW>,
    x0: [i64; N],
) -> Vec<FixedDenoiseConvIteration<F, N, IH, IW, TE, KH, KW, OH, OW>> {
    assert_eq!(N, IH * IW, "N must equal IH*IW");
    assert_eq!(
        N,
        OH * OW,
        "conv backend requires output size == input size"
    );
    assert_eq!(OH, IH, "denoise conv backend currently requires OH == IH");
    assert_eq!(OW, IW, "denoise conv backend currently requires OW == IW");
    public_params
        .real_shape
        .assert_fits::<IH, IW, KH, KW, OH, OW>();

    let table = public_params.config.clipped_relu_table();
    let mut x_i = x0;
    let mut seq = Vec::with_capacity(public_params.total_iters());

    for (t, step) in public_params.params_seq.iter().enumerate() {
        assert_kernel_padding_zero::<KH, KW>(&step.kernel, &public_params.real_shape);
        assert_flat_image_padding_zero::<IH, IW>(
            &x_i,
            &public_params.real_shape,
            &format!("step={t} x_i"),
        );
        for (j, &value) in x_i.iter().enumerate() {
            assert_in_range(
                &format!("step={t} x_i[{j}]"),
                value,
                public_params.config.value_min,
                public_params.config.value_max,
            );
        }

        let time_emb = public_params.time_table[t];
        for (j, &value) in time_emb.iter().enumerate().skip(public_params.te_real) {
            assert_eq!(
                value, 0,
                "step={t} time_emb padding coord {j} must be zero, got {value}"
            );
            assert_eq!(
                step.time_w[j], 0,
                "step={t} time_w padding coord {j} must be zero, got {}",
                step.time_w[j]
            );
        }
        let mut time_raw = 0i64;
        for j in 0..TE {
            time_raw += step.time_w[j] * time_emb[j];
        }
        let (time_q, time_r) = rescale_with_remainder(time_raw, public_params.config.scale);
        assert_in_range(
            &format!("step={t} time_q"),
            time_q,
            public_params.config.quotient_min,
            public_params.config.quotient_max,
        );
        let time_bias = time_q + step.time_b;
        assert_in_range(
            &format!("step={t} time_bias"),
            time_bias,
            public_params.config.value_min,
            public_params.config.value_max,
        );
        let effective_bias = step.conv_bias + time_bias;

        let mut image = [[0i64; IW]; IH];
        for r in 0..IH {
            for c in 0..IW {
                image[r][c] = x_i[r * IW + c];
            }
        }
        let conv_witness = apply_conv2d_fixed_point_with_witness::<IH, IW, KH, KW, OH, OW>(
            &image,
            &step.kernel,
            effective_bias,
            &public_params.padding,
            public_params.config.scale,
        );
        check_conv_ranges(t, &conv_witness, &public_params.config);

        let mut epsilon_int = [0i64; N];
        let mut epsilon = [F::ZERO; N];
        for oy in 0..OH {
            for ox in 0..OW {
                let pre = conv_witness.output[oy][ox];
                assert!(
                    table.contains(pre),
                    "step={t} conv_pre[{oy},{ox}]={pre} outside clipped relu range [{},{}]",
                    table.min,
                    table.max
                );
                let clipped = table.clipped_relu(pre);
                assert_in_range(
                    &format!("step={t} epsilon[{oy},{ox}]"),
                    clipped,
                    public_params.config.value_min,
                    public_params.config.value_max,
                );
                epsilon_int[oy * OW + ox] = clipped;
                epsilon[oy * OW + ox] = field_from_i64::<F>(clipped);
            }
        }
        assert_flat_output_padding_zero::<OH, OW>(
            &epsilon_int,
            &public_params.real_shape,
            &format!("step={t} epsilon"),
        );

        let update_witness = compute_denoise_update_witness(
            &x_i,
            &epsilon_int,
            step.alpha,
            step.beta,
            public_params.config.scale,
            public_params.update_mode,
        );
        let mut alpha_x_int = [0i64; N];
        let mut alpha_remainder_int = [0i64; N];
        let mut beta_epsilon_int = [0i64; N];
        let mut beta_remainder_int = [0i64; N];
        let x_next = *update_witness.x_next();
        if let DenoiseUpdateWitness::DoubleFloor {
            alpha_q,
            alpha_r,
            beta_q,
            beta_r,
            ..
        } = &update_witness
        {
            alpha_x_int = *alpha_q;
            alpha_remainder_int = *alpha_r;
            beta_epsilon_int = *beta_q;
            beta_remainder_int = *beta_r;
        }
        let mut x_next_f = [F::ZERO; N];
        for j in 0..N {
            match &update_witness {
                DenoiseUpdateWitness::DoubleFloor {
                    alpha_q, beta_q, ..
                } => {
                    assert_in_range(
                        &format!("step={t} alpha_x[{j}]"),
                        alpha_q[j],
                        public_params.config.quotient_min,
                        public_params.config.quotient_max,
                    );
                    assert_in_range(
                        &format!("step={t} beta_epsilon[{j}]"),
                        beta_q[j],
                        public_params.config.quotient_min,
                        public_params.config.quotient_max,
                    );
                }
                DenoiseUpdateWitness::FusedFloor { fused_q, .. } => {
                    assert_in_range(
                        &format!("step={t} fused_update[{j}]"),
                        fused_q[j],
                        public_params.config.quotient_min,
                        public_params.config.quotient_max,
                    );
                }
            }
            assert_in_range(
                &format!("step={t} x_next[{j}]"),
                x_next[j],
                public_params.config.value_min,
                public_params.config.value_max,
            );
            x_next_f[j] = field_from_i64::<F>(x_next[j]);
        }
        assert_flat_image_padding_zero::<IH, IW>(
            &x_next,
            &public_params.real_shape,
            &format!("step={t} x_next"),
        );

        seq.push(FixedDenoiseConvIteration {
            t_int: t as i64,
            time_emb_int: time_emb,
            x_i_int: x_i,
            time_raw_int: time_raw,
            time_quotient_int: time_q,
            time_remainder_int: time_r,
            time_bias_int: time_bias,
            conv_witness,
            epsilon_int,
            alpha_x_int,
            alpha_remainder_int,
            beta_epsilon_int,
            beta_remainder_int,
            update_witness,
            x_i_plus_1_int: x_next,
            epsilon,
            x_i_plus_1: x_next_f,
        });
        x_i = x_next;
    }

    seq
}

pub fn generate_fixed_point_denoise_conv_trace<
    F: PrimeField + Copy,
    const N: usize,
    const IH: usize,
    const IW: usize,
    const TE: usize,
    const KH: usize,
    const KW: usize,
    const OH: usize,
    const OW: usize,
>(
    public_params: &FixedDenoiseConvPublicParams<TE, KH, KW>,
    x0: [i64; N],
) -> (
    Vec<F>,
    Vec<FixedDenoiseConvIteration<F, N, IH, IW, TE, KH, KW, OH, OW>>,
) {
    let seq = simulate_fixed_point_denoise_conv_trace::<F, N, IH, IW, TE, KH, KW, OH, OW>(
        public_params,
        x0,
    );
    let z0 = build_z0(public_params, &x0, None);
    (z0, seq)
}

pub fn generate_fixed_point_denoise_conv_trace_with_expected_output<
    F: PrimeField + Copy,
    const N: usize,
    const IH: usize,
    const IW: usize,
    const TE: usize,
    const KH: usize,
    const KW: usize,
    const OH: usize,
    const OW: usize,
>(
    public_params: &FixedDenoiseConvPublicParams<TE, KH, KW>,
    x0: [i64; N],
    expected_output: [i64; N],
) -> (
    Vec<F>,
    Vec<FixedDenoiseConvIteration<F, N, IH, IW, TE, KH, KW, OH, OW>>,
) {
    assert_flat_image_padding_zero::<IH, IW>(
        &expected_output,
        &public_params.real_shape,
        "expected_output",
    );
    let seq = simulate_fixed_point_denoise_conv_trace::<F, N, IH, IW, TE, KH, KW, OH, OW>(
        public_params,
        x0,
    );
    let z0 = build_z0(public_params, &x0, Some(&expected_output));
    (z0, seq)
}

pub fn generate_fixed_point_denoise_conv_trace_with_computed_output<
    F: PrimeField + Copy,
    const N: usize,
    const IH: usize,
    const IW: usize,
    const TE: usize,
    const KH: usize,
    const KW: usize,
    const OH: usize,
    const OW: usize,
>(
    public_params: &FixedDenoiseConvPublicParams<TE, KH, KW>,
    x0: [i64; N],
) -> (
    Vec<F>,
    Vec<FixedDenoiseConvIteration<F, N, IH, IW, TE, KH, KW, OH, OW>>,
    [i64; N],
) {
    let seq = simulate_fixed_point_denoise_conv_trace::<F, N, IH, IW, TE, KH, KW, OH, OW>(
        public_params,
        x0,
    );
    let expected_output = seq.last().map(|last| last.x_i_plus_1_int).unwrap_or(x0);
    let z0 = build_z0(public_params, &x0, Some(&expected_output));
    (z0, seq, expected_output)
}

pub fn compute_fixed_point_denoise_conv_param_hash_witnesses<
    F: PrimeField,
    const TE: usize,
    const KH: usize,
    const KW: usize,
>(
    public_params: &FixedDenoiseConvPublicParams<TE, KH, KW>,
) -> (Vec<Vec<F>>, F) {
    let mut h = F::ZERO;
    let mut all = Vec::with_capacity(public_params.total_iters());
    for step in &public_params.params_seq {
        let (prefixes, h_next) =
            toy_hash_block_prefixes_i64_from_field(&step.flatten_i64(), TOY_HASH_BASE_U64, h);
        all.push(prefixes);
        h = h_next;
    }
    (all, h)
}

pub fn build_fixed_point_denoise_conv_z0_with_commitment<
    F: PrimeField,
    const N: usize,
    const IH: usize,
    const IW: usize,
    const TE: usize,
    const KH: usize,
    const KW: usize,
>(
    public_params: &FixedDenoiseConvPublicParams<TE, KH, KW>,
    x0: [i64; N],
    expected_output: [i64; N],
    final_commitment: F,
) -> Vec<F> {
    assert_flat_image_padding_zero::<IH, IW>(
        &expected_output,
        &public_params.real_shape,
        "expected_output",
    );
    let layout = CommittedDenoiseStateLayout::new(
        N,
        true,
        public_params.total_iters(),
        public_params.block_len(),
        TE,
    );
    let mut z0 = Vec::with_capacity(layout.state_len());
    z0.extend(x0.into_iter().map(field_from_i64::<F>));
    z0.extend(expected_output.into_iter().map(field_from_i64::<F>));
    z0.push(F::ZERO);
    z0.push(final_commitment);
    z0.push(F::ZERO);
    for step in &public_params.params_seq {
        z0.extend(step.flatten_field::<F>());
    }
    for row in &public_params.time_table {
        for &value in row {
            z0.push(field_from_i64::<F>(value));
        }
    }
    layout.assert_state_len(z0.len());
    z0
}

pub fn generate_fixed_point_denoise_conv_trace_with_commitment<
    F: PrimeField + Copy,
    const N: usize,
    const IH: usize,
    const IW: usize,
    const TE: usize,
    const KH: usize,
    const KW: usize,
    const OH: usize,
    const OW: usize,
>(
    public_params: &FixedDenoiseConvPublicParams<TE, KH, KW>,
    x0: [i64; N],
) -> (
    Vec<F>,
    Vec<FixedDenoiseConvIteration<F, N, IH, IW, TE, KH, KW, OH, OW>>,
    Vec<Vec<F>>,
    [i64; N],
    F,
) {
    let seq = simulate_fixed_point_denoise_conv_trace::<F, N, IH, IW, TE, KH, KW, OH, OW>(
        public_params,
        x0,
    );
    let expected_output = seq.last().map(|last| last.x_i_plus_1_int).unwrap_or(x0);
    let (hash_witnesses, commitment) =
        compute_fixed_point_denoise_conv_param_hash_witnesses(public_params);
    let z0 = build_fixed_point_denoise_conv_z0_with_commitment::<F, N, IH, IW, TE, KH, KW>(
        public_params,
        x0,
        expected_output,
        commitment,
    );
    (z0, seq, hash_witnesses, expected_output, commitment)
}
