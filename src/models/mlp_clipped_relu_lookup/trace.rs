use ff::PrimeField;

use crate::clipped_relu::field_from_i64;

use super::params::{IntMlpClippedReluStepParams, MlpClippedReluPublicParams};

#[derive(Clone, Debug)]
pub struct MlpClippedReluIteration<F: PrimeField + Copy, const N: usize, const H: usize> {
    pub x_i: [F; N],
    pub hidden_affine: [F; H],
    pub hidden_act: [F; H],
    pub x_i_plus_1: [F; N],
    pub hidden_affine_int: [i64; H],
    pub hidden_act_int: [i64; H],
    pub x_i_plus_1_int: [i64; N],
}

fn apply_step_i64<const N: usize, const H: usize>(
    step: &IntMlpClippedReluStepParams<N, H>,
    x_i: &[i64; N],
    clip: impl Fn(i64) -> i64,
) -> ([i64; H], [i64; H], [i64; N]) {
    let mut hidden_affine = [0i64; H];
    let mut hidden_act = [0i64; H];
    let mut x_next = [0i64; N];

    for r in 0..H {
        let mut acc = step.b1[r];
        for c in 0..N {
            acc += step.w1[r][c] * x_i[c];
        }
        hidden_affine[r] = acc;
        hidden_act[r] = clip(acc);
    }

    for r in 0..N {
        let mut acc = step.b2[r];
        for c in 0..H {
            acc += step.w2[r][c] * hidden_act[c];
        }
        x_next[r] = acc;
    }

    (hidden_affine, hidden_act, x_next)
}

/// Generate z0 and the private execution trace.
///
/// Public state layout:
/// z = [x | params_0 | params_1 | ... | params_{T-1}]
/// where params_i = [W1_i row-major | b1_i | W2_i row-major | b2_i].
pub fn generate_mlp_clipped_relu_trace<F: PrimeField + Copy, const N: usize, const H: usize>(
    public_params: &MlpClippedReluPublicParams<N, H>,
    x0: [i64; N],
) -> (Vec<F>, Vec<MlpClippedReluIteration<F, N, H>>) {
    let mut z0: Vec<F> = x0.iter().map(|&v| field_from_i64::<F>(v)).collect();
    for step in &public_params.params_seq {
        z0.extend(step.flatten_field::<F>());
    }

    let table = &public_params.clipped_relu_table;
    let mut x_i_int = x0;
    let mut seq = Vec::with_capacity(public_params.params_seq.len());

    for (step_idx, step) in public_params.params_seq.iter().enumerate() {
        let (hidden_affine_int, hidden_act_int, x_next_int) =
            apply_step_i64(step, &x_i_int, |v| table.clipped_relu(v));

        for &v in hidden_affine_int.iter() {
            assert!(
                table.contains(v),
                "step {step_idx}: hidden affine output {v} is outside clipped ReLU table range [{}, {}]",
                table.min,
                table.max
            );
        }

        let mut x_i = [F::ZERO; N];
        let mut hidden_affine = [F::ZERO; H];
        let mut hidden_act = [F::ZERO; H];
        let mut x_i_plus_1 = [F::ZERO; N];

        for r in 0..N {
            x_i[r] = field_from_i64::<F>(x_i_int[r]);
            x_i_plus_1[r] = field_from_i64::<F>(x_next_int[r]);
        }
        for r in 0..H {
            hidden_affine[r] = field_from_i64::<F>(hidden_affine_int[r]);
            hidden_act[r] = field_from_i64::<F>(hidden_act_int[r]);
        }

        seq.push(MlpClippedReluIteration {
            x_i,
            hidden_affine,
            hidden_act,
            x_i_plus_1,
            hidden_affine_int,
            hidden_act_int,
            x_i_plus_1_int: x_next_int,
        });

        x_i_int = x_next_int;
    }

    (z0, seq)
}
