use ff::PrimeField;

use crate::{
    affine::fixed_point::apply_affine_fixed_point_with_witness, clipped_relu::field_from_i64,
};

use super::params::FixedMlpClippedReluPublicParams;

#[derive(Clone, Debug)]
pub struct FixedPointMlpClippedReluIteration<F: PrimeField + Copy, const N: usize, const H: usize> {
    pub x_i: [F; N],

    pub hidden_raw_int: [i64; H],
    pub hidden_quotient_int: [i64; H],
    pub hidden_remainder_int: [i64; H],
    pub hidden_affine_int: [i64; H],
    pub hidden_act_int: [i64; H],

    pub output_raw_int: [i64; N],
    pub output_quotient_int: [i64; N],
    pub output_remainder_int: [i64; N],
    pub x_i_plus_1_int: [i64; N],

    pub hidden_raw: [F; H],
    pub hidden_quotient: [F; H],
    pub hidden_remainder: [F; H],
    pub hidden_affine: [F; H],
    pub hidden_act: [F; H],

    pub output_raw: [F; N],
    pub output_quotient: [F; N],
    pub output_remainder: [F; N],
    pub x_i_plus_1: [F; N],
}

fn assert_in_range(value: i64, min: i64, max: i64, label: &str) {
    assert!(
        min <= value && value <= max,
        "{label} fixed-point value {value} is outside signed range [{min}, {max}]"
    );
}

/// Generate z0 and private trace for fixed-point public-parameter MLP.
/// Public state layout:
/// z = [x | params_0 | params_1 | ... | params_{T-1}],
/// where params_i = [W1_i row-major | b1_i | W2_i row-major | b2_i].
pub fn generate_fixed_point_mlp_trace<F: PrimeField + Copy, const N: usize, const H: usize>(
    public_params: &FixedMlpClippedReluPublicParams<N, H>,
    x0: [i64; N],
) -> (Vec<F>, Vec<FixedPointMlpClippedReluIteration<F, N, H>>) {
    let config = &public_params.config;
    let table = config.clipped_relu_table();

    let mut z0: Vec<F> = x0.iter().map(|&v| field_from_i64::<F>(v)).collect();
    for step in &public_params.params_seq {
        z0.extend(step.flatten_field::<F>());
    }

    let mut x_i_int = x0;
    let mut seq = Vec::with_capacity(public_params.params_seq.len());

    for (step_idx, step) in public_params.params_seq.iter().enumerate() {
        for j in 0..N {
            assert_in_range(x_i_int[j], config.value_min, config.value_max, "x_i");
        }

        let (hidden_raw_int, hidden_quotient_int, hidden_remainder_int, hidden_affine_int) =
            apply_affine_fixed_point_with_witness(&step.w1, &step.b1, &x_i_int, config.scale);

        let mut hidden_act_int = [0i64; H];
        for r in 0..H {
            assert_in_range(
                hidden_quotient_int[r],
                config.quotient_min,
                config.quotient_max,
                "hidden quotient",
            );
            assert!(
                table.contains(hidden_affine_int[r]),
                "step {step_idx}: hidden affine fixed-point value {} is outside clipped ReLU table range [{}, {}]",
                hidden_affine_int[r],
                table.min,
                table.max
            );
            hidden_act_int[r] = table.clipped_relu(hidden_affine_int[r]);
        }

        let (output_raw_int, output_quotient_int, output_remainder_int, x_next_int) =
            apply_affine_fixed_point_with_witness(
                &step.w2,
                &step.b2,
                &hidden_act_int,
                config.scale,
            );
        for r in 0..N {
            assert_in_range(
                output_quotient_int[r],
                config.quotient_min,
                config.quotient_max,
                "output quotient",
            );
            assert_in_range(x_next_int[r], config.value_min, config.value_max, "x_next");
        }

        let mut x_i = [F::ZERO; N];
        let mut hidden_raw = [F::ZERO; H];
        let mut hidden_quotient = [F::ZERO; H];
        let mut hidden_remainder = [F::ZERO; H];
        let mut hidden_affine = [F::ZERO; H];
        let mut hidden_act = [F::ZERO; H];
        let mut output_raw = [F::ZERO; N];
        let mut output_quotient = [F::ZERO; N];
        let mut output_remainder = [F::ZERO; N];
        let mut x_i_plus_1 = [F::ZERO; N];

        for r in 0..N {
            x_i[r] = field_from_i64::<F>(x_i_int[r]);
            output_raw[r] = field_from_i64::<F>(output_raw_int[r]);
            output_quotient[r] = field_from_i64::<F>(output_quotient_int[r]);
            output_remainder[r] = field_from_i64::<F>(output_remainder_int[r]);
            x_i_plus_1[r] = field_from_i64::<F>(x_next_int[r]);
        }
        for r in 0..H {
            hidden_raw[r] = field_from_i64::<F>(hidden_raw_int[r]);
            hidden_quotient[r] = field_from_i64::<F>(hidden_quotient_int[r]);
            hidden_remainder[r] = field_from_i64::<F>(hidden_remainder_int[r]);
            hidden_affine[r] = field_from_i64::<F>(hidden_affine_int[r]);
            hidden_act[r] = field_from_i64::<F>(hidden_act_int[r]);
        }

        seq.push(FixedPointMlpClippedReluIteration {
            x_i,
            hidden_raw_int,
            hidden_quotient_int,
            hidden_remainder_int,
            hidden_affine_int,
            hidden_act_int,
            output_raw_int,
            output_quotient_int,
            output_remainder_int,
            x_i_plus_1_int: x_next_int,
            hidden_raw,
            hidden_quotient,
            hidden_remainder,
            hidden_affine,
            hidden_act,
            output_raw,
            output_quotient,
            output_remainder,
            x_i_plus_1,
        });

        x_i_int = x_next_int;
    }

    (z0, seq)
}
