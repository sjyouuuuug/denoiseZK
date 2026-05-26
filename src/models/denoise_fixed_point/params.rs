use ff::PrimeField;

use crate::{
    clipped_relu::field_from_i64,
    fixed_point::{encode_f64_round, FixedPointConfig},
    mlp_fixed_point_clipped_relu_lookup::FixedMlpClippedReluStepParams,
};

#[derive(Clone, Debug)]
pub struct FixedDenoiseStepParams<const N: usize, const H: usize> {
    pub mlp: FixedMlpClippedReluStepParams<N, H>,
    pub alpha: i64,
    pub beta: i64,
}

#[derive(Clone, Debug)]
pub struct FixedDenoisePublicParams<const N: usize, const H: usize> {
    pub params_seq: Vec<FixedDenoiseStepParams<N, H>>,
    pub config: FixedPointConfig,
}

impl<const N: usize, const H: usize> FixedDenoiseStepParams<N, H> {
    pub fn new(mlp: FixedMlpClippedReluStepParams<N, H>, alpha: i64, beta: i64) -> Self {
        Self { mlp, alpha, beta }
    }

    pub fn from_f64(
        w1: [[f64; N]; H],
        b1: [f64; H],
        w2: [[f64; H]; N],
        b2: [f64; N],
        alpha: f64,
        beta: f64,
        scale: i64,
    ) -> Self {
        let mlp = FixedMlpClippedReluStepParams::from_f64(w1, b1, w2, b2, scale);
        Self::new(
            mlp,
            encode_f64_round(alpha, scale),
            encode_f64_round(beta, scale),
        )
    }

    pub fn block_len() -> usize {
        FixedMlpClippedReluStepParams::<N, H>::block_len() + 2
    }

    pub fn flatten_i64(&self) -> Vec<i64> {
        let mut out = self.mlp.flatten_i64();
        out.push(self.alpha);
        out.push(self.beta);
        out
    }

    pub fn flatten_field<F: PrimeField>(&self) -> Vec<F> {
        self.flatten_i64()
            .into_iter()
            .map(field_from_i64::<F>)
            .collect()
    }
}

impl<const N: usize, const H: usize> FixedDenoisePublicParams<N, H> {
    pub fn new(params_seq: Vec<FixedDenoiseStepParams<N, H>>, config: FixedPointConfig) -> Self {
        Self { params_seq, config }
    }

    pub fn total_iters(&self) -> usize {
        self.params_seq.len()
    }

    pub fn block_len(&self) -> usize {
        FixedDenoiseStepParams::<N, H>::block_len()
    }
}
