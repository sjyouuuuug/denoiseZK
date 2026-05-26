use ff::PrimeField;

use crate::{
    affine::fixed_point::apply_affine_fixed_point_with_witness,
    clipped_relu::field_from_i64,
    models::denoise_update::{compute_denoise_update_witness, DenoiseUpdateWitness},
};

use super::params::FixedDenoiseTimeEmbeddingPublicParams;

#[derive(Clone, Debug)]
pub struct FixedDenoiseTimeEmbeddingIteration<
    F: PrimeField + Copy,
    const N: usize,
    const TE: usize,
    const IN: usize,
    const H: usize,
> {
    pub t_int: i64,
    pub time_emb_int: [i64; TE],

    pub x_i: [F; N],
    pub x_i_int: [i64; N],

    pub mlp_input: [F; IN],
    pub mlp_input_int: [i64; IN],

    pub hidden_raw: [F; H],
    pub hidden_quotient: [F; H],
    pub hidden_remainder: [F; H],
    pub hidden_affine: [F; H],
    pub hidden_act: [F; H],
    pub epsilon: [F; N],

    pub hidden_raw_int: [i64; H],
    pub hidden_quotient_int: [i64; H],
    pub hidden_remainder_int: [i64; H],
    pub hidden_affine_int: [i64; H],
    pub hidden_act_int: [i64; H],
    pub epsilon_int: [i64; N],

    pub output_raw: [F; N],
    pub output_quotient: [F; N],
    pub output_remainder: [F; N],
    pub output_raw_int: [i64; N],
    pub output_quotient_int: [i64; N],
    pub output_remainder_int: [i64; N],

    pub alpha_x: [F; N],
    pub beta_epsilon: [F; N],
    pub x_i_plus_1: [F; N],

    pub alpha_x_int: [i64; N],
    pub beta_epsilon_int: [i64; N],
    pub x_i_plus_1_int: [i64; N],

    pub alpha_mul_raw_int: [i64; N],
    pub alpha_remainder_int: [i64; N],
    pub beta_mul_raw_int: [i64; N],
    pub beta_remainder_int: [i64; N],
    pub update_witness: DenoiseUpdateWitness<N>,
}

fn assert_in_range(value: i64, min: i64, max: i64, label: &str) {
    assert!(
        min <= value && value <= max,
        "{label} fixed-point value {value} is outside signed range [{min}, {max}]"
    );
}

pub fn generate_fixed_point_denoise_time_embedding_trace<
    F: PrimeField + Copy,
    const N: usize,
    const TE: usize,
    const IN: usize,
    const H: usize,
>(
    public_params: &FixedDenoiseTimeEmbeddingPublicParams<N, TE, IN, H>,
    x0: [i64; N],
) -> (
    Vec<F>,
    Vec<FixedDenoiseTimeEmbeddingIteration<F, N, TE, IN, H>>,
) {
    assert_eq!(IN, N + TE, "IN must equal N + TE");

    let mut z0: Vec<F> = x0.iter().map(|&v| field_from_i64::<F>(v)).collect();
    z0.push(F::ZERO);
    for step in &public_params.params_seq {
        z0.extend(step.flatten_field::<F>());
    }
    for row in &public_params.time_table {
        for &value in row {
            z0.push(field_from_i64::<F>(value));
        }
    }

    let table = public_params.config.clipped_relu_table();
    let mut x_i_int = x0;
    let mut seq = Vec::with_capacity(public_params.total_iters());

    for (t, step) in public_params.params_seq.iter().enumerate() {
        let t_int = t as i64;
        let time_emb_int = public_params.time_table[t];

        for j in 0..N {
            assert_in_range(
                x_i_int[j],
                public_params.config.value_min,
                public_params.config.value_max,
                "x_i",
            );
        }

        let mut mlp_input_int = [0i64; IN];
        mlp_input_int[..N].copy_from_slice(&x_i_int);
        mlp_input_int[N..(N + TE)].copy_from_slice(&time_emb_int);

        let (hidden_raw_int, hidden_quotient_int, hidden_remainder_int, hidden_affine_int) =
            apply_affine_fixed_point_with_witness(
                &step.w1,
                &step.b1,
                &mlp_input_int,
                public_params.config.scale,
            );

        let mut hidden_act_int = [0i64; H];
        for r in 0..H {
            assert_in_range(
                hidden_quotient_int[r],
                public_params.config.quotient_min,
                public_params.config.quotient_max,
                "hidden quotient",
            );
            assert!(
                table.contains(hidden_affine_int[r]),
                "step {t}: hidden affine fixed-point value {} is outside clipped ReLU table range [{}, {}]",
                hidden_affine_int[r],
                table.min,
                table.max
            );
            hidden_act_int[r] = table.clipped_relu(hidden_affine_int[r]);
        }

        let (output_raw_int, output_quotient_int, output_remainder_int, epsilon_int) =
            apply_affine_fixed_point_with_witness(
                &step.w2,
                &step.b2,
                &hidden_act_int,
                public_params.config.scale,
            );
        for j in 0..N {
            assert_in_range(
                output_quotient_int[j],
                public_params.config.quotient_min,
                public_params.config.quotient_max,
                "output quotient",
            );
            assert_in_range(
                epsilon_int[j],
                public_params.config.value_min,
                public_params.config.value_max,
                "epsilon",
            );
        }

        let update_witness = compute_denoise_update_witness(
            &x_i_int,
            &epsilon_int,
            step.alpha,
            step.beta,
            public_params.config.scale,
            public_params.update_mode,
        );
        let mut alpha_mul_raw_int = [0i64; N];
        let mut alpha_x_int = [0i64; N];
        let mut alpha_remainder_int = [0i64; N];
        let mut beta_mul_raw_int = [0i64; N];
        let mut beta_epsilon_int = [0i64; N];
        let mut beta_remainder_int = [0i64; N];
        if let DenoiseUpdateWitness::DoubleFloor {
            alpha_raw,
            alpha_q,
            alpha_r,
            beta_raw,
            beta_q,
            beta_r,
            ..
        } = &update_witness
        {
            alpha_mul_raw_int = *alpha_raw;
            alpha_x_int = *alpha_q;
            alpha_remainder_int = *alpha_r;
            beta_mul_raw_int = *beta_raw;
            beta_epsilon_int = *beta_q;
            beta_remainder_int = *beta_r;
        }
        let x_next_int = *update_witness.x_next();

        for j in 0..N {
            match &update_witness {
                DenoiseUpdateWitness::DoubleFloor {
                    alpha_q, beta_q, ..
                } => {
                    assert_in_range(
                        alpha_q[j],
                        public_params.config.quotient_min,
                        public_params.config.quotient_max,
                        "alpha quotient",
                    );
                    assert_in_range(
                        beta_q[j],
                        public_params.config.quotient_min,
                        public_params.config.quotient_max,
                        "beta quotient",
                    );
                }
                DenoiseUpdateWitness::FusedFloor { fused_q, .. } => {
                    assert_in_range(
                        fused_q[j],
                        public_params.config.quotient_min,
                        public_params.config.quotient_max,
                        "fused quotient",
                    );
                }
            }
            assert_in_range(
                x_next_int[j],
                public_params.config.value_min,
                public_params.config.value_max,
                "x_next",
            );
        }

        let mut x_i = [F::ZERO; N];
        let mut mlp_input = [F::ZERO; IN];
        let mut hidden_raw = [F::ZERO; H];
        let mut hidden_quotient = [F::ZERO; H];
        let mut hidden_remainder = [F::ZERO; H];
        let mut hidden_affine = [F::ZERO; H];
        let mut hidden_act = [F::ZERO; H];
        let mut output_raw = [F::ZERO; N];
        let mut output_quotient = [F::ZERO; N];
        let mut output_remainder = [F::ZERO; N];
        let mut epsilon = [F::ZERO; N];
        let mut alpha_x = [F::ZERO; N];
        let mut beta_epsilon = [F::ZERO; N];
        let mut x_i_plus_1 = [F::ZERO; N];

        for j in 0..N {
            x_i[j] = field_from_i64::<F>(x_i_int[j]);
            output_raw[j] = field_from_i64::<F>(output_raw_int[j]);
            output_quotient[j] = field_from_i64::<F>(output_quotient_int[j]);
            output_remainder[j] = field_from_i64::<F>(output_remainder_int[j]);
            epsilon[j] = field_from_i64::<F>(epsilon_int[j]);
            alpha_x[j] = field_from_i64::<F>(alpha_x_int[j]);
            beta_epsilon[j] = field_from_i64::<F>(beta_epsilon_int[j]);
            x_i_plus_1[j] = field_from_i64::<F>(x_next_int[j]);
        }
        for j in 0..IN {
            mlp_input[j] = field_from_i64::<F>(mlp_input_int[j]);
        }
        for r in 0..H {
            hidden_raw[r] = field_from_i64::<F>(hidden_raw_int[r]);
            hidden_quotient[r] = field_from_i64::<F>(hidden_quotient_int[r]);
            hidden_remainder[r] = field_from_i64::<F>(hidden_remainder_int[r]);
            hidden_affine[r] = field_from_i64::<F>(hidden_affine_int[r]);
            hidden_act[r] = field_from_i64::<F>(hidden_act_int[r]);
        }

        seq.push(FixedDenoiseTimeEmbeddingIteration {
            t_int,
            time_emb_int,
            x_i,
            x_i_int,
            mlp_input,
            mlp_input_int,
            hidden_raw,
            hidden_quotient,
            hidden_remainder,
            hidden_affine,
            hidden_act,
            epsilon,
            hidden_raw_int,
            hidden_quotient_int,
            hidden_remainder_int,
            hidden_affine_int,
            hidden_act_int,
            epsilon_int,
            output_raw,
            output_quotient,
            output_remainder,
            output_raw_int,
            output_quotient_int,
            output_remainder_int,
            alpha_x,
            beta_epsilon,
            x_i_plus_1,
            alpha_x_int,
            beta_epsilon_int,
            x_i_plus_1_int: x_next_int,
            alpha_mul_raw_int,
            alpha_remainder_int,
            beta_mul_raw_int,
            beta_remainder_int,
            update_witness,
        });

        x_i_int = x_next_int;
    }

    (z0, seq)
}
