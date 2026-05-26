use ff::PrimeField;

use crate::{
    clipped_relu::field_from_i64,
    fixed_point::{encode_f64_round, FixedPointConfig},
};

#[derive(Clone, Debug)]
pub struct FixedMlpClippedReluStepParams<const N: usize, const H: usize> {
    /// First affine layer: H x N, fixed-point encoded.
    pub w1: [[i64; N]; H],
    pub b1: [i64; H],
    /// Second affine layer: N x H, fixed-point encoded.
    pub w2: [[i64; H]; N],
    pub b2: [i64; N],
}

#[derive(Clone, Debug)]
pub struct FixedMlpClippedReluPublicParams<const N: usize, const H: usize> {
    pub params_seq: Vec<FixedMlpClippedReluStepParams<N, H>>,
    pub config: FixedPointConfig,
}

impl<const N: usize, const H: usize> FixedMlpClippedReluStepParams<N, H> {
    pub fn new(w1: [[i64; N]; H], b1: [i64; H], w2: [[i64; H]; N], b2: [i64; N]) -> Self {
        Self { w1, b1, w2, b2 }
    }

    pub fn from_f64(
        w1: [[f64; N]; H],
        b1: [f64; H],
        w2: [[f64; H]; N],
        b2: [f64; N],
        scale: i64,
    ) -> Self {
        let mut w1_i = [[0i64; N]; H];
        let mut b1_i = [0i64; H];
        let mut w2_i = [[0i64; H]; N];
        let mut b2_i = [0i64; N];
        for r in 0..H {
            for c in 0..N {
                w1_i[r][c] = encode_f64_round(w1[r][c], scale);
            }
            b1_i[r] = encode_f64_round(b1[r], scale);
        }
        for r in 0..N {
            for c in 0..H {
                w2_i[r][c] = encode_f64_round(w2[r][c], scale);
            }
            b2_i[r] = encode_f64_round(b2[r], scale);
        }
        Self::new(w1_i, b1_i, w2_i, b2_i)
    }

    pub fn block_len() -> usize {
        H * N + H + N * H + N
    }

    /// Flatten layout: [W1 row-major | b1 | W2 row-major | b2]
    pub fn flatten_i64(&self) -> Vec<i64> {
        let mut out = Vec::with_capacity(Self::block_len());
        for r in 0..H {
            for c in 0..N {
                out.push(self.w1[r][c]);
            }
        }
        for r in 0..H {
            out.push(self.b1[r]);
        }
        for r in 0..N {
            for c in 0..H {
                out.push(self.w2[r][c]);
            }
        }
        for r in 0..N {
            out.push(self.b2[r]);
        }
        out
    }

    pub fn flatten_field<F: PrimeField>(&self) -> Vec<F> {
        self.flatten_i64()
            .into_iter()
            .map(field_from_i64::<F>)
            .collect()
    }
}

impl<const N: usize, const H: usize> FixedMlpClippedReluPublicParams<N, H> {
    pub fn new(
        params_seq: Vec<FixedMlpClippedReluStepParams<N, H>>,
        config: FixedPointConfig,
    ) -> Self {
        Self { params_seq, config }
    }

    pub fn total_iters(&self) -> usize {
        self.params_seq.len()
    }
}
