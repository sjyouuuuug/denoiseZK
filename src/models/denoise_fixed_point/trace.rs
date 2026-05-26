use ff::PrimeField;

use crate::{
    clipped_relu::field_from_i64,
    fixed_point::rescale_with_remainder,
    mlp_fixed_point_clipped_relu_lookup::{
        generate_fixed_point_mlp_trace, FixedMlpClippedReluPublicParams,
        FixedPointMlpClippedReluIteration,
    },
};

use super::params::FixedDenoisePublicParams;

#[derive(Clone, Debug)]
pub struct FixedDenoiseIteration<F: PrimeField + Copy, const N: usize, const H: usize> {
    pub x_i: [F; N],

    pub hidden_affine: [F; H],
    pub hidden_act: [F; H],
    pub epsilon: [F; N],

    pub x_i_int: [i64; N],
    pub hidden_affine_int: [i64; H],
    pub hidden_act_int: [i64; H],
    pub epsilon_int: [i64; N],

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

    pub mlp_witness: FixedPointMlpClippedReluIteration<F, N, H>,
}

fn assert_in_range(value: i64, min: i64, max: i64, label: &str) {
    assert!(
        min <= value && value <= max,
        "{label} fixed-point value {value} is outside signed range [{min}, {max}]"
    );
}

pub fn generate_fixed_point_denoise_trace<F: PrimeField + Copy, const N: usize, const H: usize>(
    public_params: &FixedDenoisePublicParams<N, H>,
    x0: [i64; N],
) -> (Vec<F>, Vec<FixedDenoiseIteration<F, N, H>>) {
    let mut z0: Vec<F> = x0.iter().map(|&v| field_from_i64::<F>(v)).collect();
    for step in &public_params.params_seq {
        z0.extend(step.flatten_field::<F>());
    }

    let mut x_i_int = x0;
    let mut seq = Vec::with_capacity(public_params.params_seq.len());

    for step in &public_params.params_seq {
        for j in 0..N {
            assert_in_range(
                x_i_int[j],
                public_params.config.value_min,
                public_params.config.value_max,
                "x_i",
            );
        }

        let mlp_public = FixedMlpClippedReluPublicParams::new(
            vec![step.mlp.clone()],
            public_params.config.clone(),
        );
        let (_, mlp_trace) = generate_fixed_point_mlp_trace::<F, N, H>(&mlp_public, x_i_int);
        let mlp_it = mlp_trace.into_iter().next().expect("one MLP iteration");
        let epsilon_int = mlp_it.x_i_plus_1_int;

        let mut alpha_mul_raw_int = [0i64; N];
        let mut alpha_x_int = [0i64; N];
        let mut alpha_remainder_int = [0i64; N];
        let mut beta_mul_raw_int = [0i64; N];
        let mut beta_epsilon_int = [0i64; N];
        let mut beta_remainder_int = [0i64; N];
        let mut x_next_int = [0i64; N];

        for j in 0..N {
            assert_in_range(
                epsilon_int[j],
                public_params.config.value_min,
                public_params.config.value_max,
                "epsilon",
            );

            alpha_mul_raw_int[j] = step.alpha * x_i_int[j];
            let (alpha_q, alpha_r) =
                rescale_with_remainder(alpha_mul_raw_int[j], public_params.config.scale);
            assert!(0 <= alpha_r && alpha_r < public_params.config.scale);
            assert_in_range(
                alpha_q,
                public_params.config.quotient_min,
                public_params.config.quotient_max,
                "alpha quotient",
            );
            alpha_x_int[j] = alpha_q;
            alpha_remainder_int[j] = alpha_r;

            beta_mul_raw_int[j] = step.beta * epsilon_int[j];
            let (beta_q, beta_r) =
                rescale_with_remainder(beta_mul_raw_int[j], public_params.config.scale);
            assert!(0 <= beta_r && beta_r < public_params.config.scale);
            assert_in_range(
                beta_q,
                public_params.config.quotient_min,
                public_params.config.quotient_max,
                "beta quotient",
            );
            beta_epsilon_int[j] = beta_q;
            beta_remainder_int[j] = beta_r;

            x_next_int[j] = alpha_q + beta_q;
            assert_in_range(
                x_next_int[j],
                public_params.config.value_min,
                public_params.config.value_max,
                "x_next",
            );
        }

        let mut x_i = [F::ZERO; N];
        let mut alpha_x = [F::ZERO; N];
        let mut beta_epsilon = [F::ZERO; N];
        let mut x_i_plus_1 = [F::ZERO; N];
        for j in 0..N {
            x_i[j] = field_from_i64::<F>(x_i_int[j]);
            alpha_x[j] = field_from_i64::<F>(alpha_x_int[j]);
            beta_epsilon[j] = field_from_i64::<F>(beta_epsilon_int[j]);
            x_i_plus_1[j] = field_from_i64::<F>(x_next_int[j]);
        }

        seq.push(FixedDenoiseIteration {
            x_i,
            hidden_affine: mlp_it.hidden_affine,
            hidden_act: mlp_it.hidden_act,
            epsilon: mlp_it.x_i_plus_1,
            x_i_int,
            hidden_affine_int: mlp_it.hidden_affine_int,
            hidden_act_int: mlp_it.hidden_act_int,
            epsilon_int,
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
            mlp_witness: mlp_it,
        });

        x_i_int = x_next_int;
    }

    (z0, seq)
}
