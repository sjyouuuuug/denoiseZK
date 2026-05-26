use ff::PrimeField;

use crate::relu::field_from_i64;

use super::params::IntAffineReluLookupParams;

#[derive(Clone, Debug)]
pub struct AffineReluLookupIteration<F: PrimeField + Copy, const N: usize> {
    pub x_i: [F; N],
    pub affine_out: [F; N],
    pub x_i_plus_1: [F; N],
    pub affine_out_int: [i64; N],
    pub x_i_plus_1_int: [i64; N],
}

pub fn generate_affine_relu_lookup_trace<F: PrimeField + Copy, const N: usize>(
    params: &IntAffineReluLookupParams<N>,
    x0: [i64; N],
    num_iters: usize,
) -> (Vec<F>, Vec<AffineReluLookupIteration<F, N>>) {
    let mut x_i_int = x0;
    let mut seq = Vec::with_capacity(num_iters);

    for _ in 0..num_iters {
        let mut affine_out_int = [0i64; N];
        let mut x_next_int = [0i64; N];

        for r in 0..N {
            let mut acc = params.b[r];
            for c in 0..N {
                acc += params.a[r][c] * x_i_int[c];
            }
            assert!(
                params.relu_table.contains(acc),
                "affine output {} is outside ReLU table range [{}, {}]",
                acc,
                params.relu_table.min,
                params.relu_table.max
            );
            affine_out_int[r] = acc;
            x_next_int[r] = params.relu_table.relu(acc);
        }

        let mut x_i = [F::ZERO; N];
        let mut affine_out = [F::ZERO; N];
        let mut x_i_plus_1 = [F::ZERO; N];
        for r in 0..N {
            x_i[r] = field_from_i64::<F>(x_i_int[r]);
            affine_out[r] = field_from_i64::<F>(affine_out_int[r]);
            x_i_plus_1[r] = field_from_i64::<F>(x_next_int[r]);
        }

        seq.push(AffineReluLookupIteration {
            x_i,
            affine_out,
            x_i_plus_1,
            affine_out_int,
            x_i_plus_1_int: x_next_int,
        });

        x_i_int = x_next_int;
    }

    let z0 = x0.iter().map(|&v| field_from_i64::<F>(v)).collect();
    (z0, seq)
}
