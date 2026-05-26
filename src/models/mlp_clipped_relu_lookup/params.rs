use ff::PrimeField;

use crate::clipped_relu::{field_from_i64, ClippedReluLookupTable};

#[derive(Clone, Debug)]
pub struct IntMlpClippedReluStepParams<const N: usize, const H: usize> {
    /// First affine layer: H x N
    pub w1: [[i64; N]; H],
    pub b1: [i64; H],
    /// Second affine layer: N x H
    pub w2: [[i64; H]; N],
    pub b2: [i64; N],
}

#[derive(Clone, Debug)]
pub struct MlpClippedReluPublicParams<const N: usize, const H: usize> {
    pub params_seq: Vec<IntMlpClippedReluStepParams<N, H>>,
    pub clipped_relu_table: ClippedReluLookupTable,
}

impl<const N: usize, const H: usize> IntMlpClippedReluStepParams<N, H> {
    pub fn new(w1: [[i64; N]; H], b1: [i64; H], w2: [[i64; H]; N], b2: [i64; N]) -> Self {
        Self { w1, b1, w2, b2 }
    }

    pub fn block_len() -> usize {
        H * N + H + N * H + N
    }

    /// Flatten layout:
    /// [W1 row-major | b1 | W2 row-major | b2]
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

impl<const N: usize, const H: usize> MlpClippedReluPublicParams<N, H> {
    pub fn new(
        params_seq: Vec<IntMlpClippedReluStepParams<N, H>>,
        clipped_relu_table: ClippedReluLookupTable,
    ) -> Self {
        Self {
            params_seq,
            clipped_relu_table,
        }
    }

    pub fn total_iters(&self) -> usize {
        self.params_seq.len()
    }
}
