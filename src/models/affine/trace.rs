use ff::Field;

use super::{params::AffineParams, util::apply_affine};

#[derive(Clone, Debug)]
pub struct AffineIteration<F: Field + Copy, const N: usize> {
    pub x_i: [F; N],
    pub x_i_plus_1: [F; N],
}

pub fn generate_affine_trace<F: Field + Copy, const N: usize>(
    params: &AffineParams<F, N>,
    x0: [F; N],
    num_iters: usize,
) -> (Vec<F>, Vec<AffineIteration<F, N>>) {
    let mut x_i = x0;
    let mut seq = Vec::with_capacity(num_iters);

    for _ in 0..num_iters {
        let x_next = apply_affine(&params.a, &params.b, &x_i);
        seq.push(AffineIteration {
            x_i,
            x_i_plus_1: x_next,
        });
        x_i = x_next;
    }

    (x0.to_vec(), seq)
}
