use ff::PrimeField;

use crate::{
    clipped_relu::field_from_i64,
    fixed_point::{encode_f64_round, FixedPointConfig},
    layers::conv2d::{Conv2dPadding, Conv2dRealShape},
    models::denoise_update::DenoiseUpdateMode,
};

#[derive(Clone, Debug)]
pub struct FixedDenoiseConvStepParams<const TE: usize, const KH: usize, const KW: usize> {
    pub kernel: [[i64; KW]; KH],
    pub conv_bias: i64,
    pub time_w: [i64; TE],
    pub time_b: i64,
    pub alpha: i64,
    pub beta: i64,
}

#[derive(Clone, Debug)]
pub struct FixedDenoiseConvPublicParams<const TE: usize, const KH: usize, const KW: usize> {
    pub params_seq: Vec<FixedDenoiseConvStepParams<TE, KH, KW>>,
    pub time_table: Vec<[i64; TE]>,
    pub config: FixedPointConfig,
    pub padding: Conv2dPadding,
    pub real_shape: Conv2dRealShape,
    pub te_real: usize,
    pub update_mode: DenoiseUpdateMode,
}

impl<const TE: usize, const KH: usize, const KW: usize> FixedDenoiseConvStepParams<TE, KH, KW> {
    pub fn new(
        kernel: [[i64; KW]; KH],
        conv_bias: i64,
        time_w: [i64; TE],
        time_b: i64,
        alpha: i64,
        beta: i64,
    ) -> Self {
        Self {
            kernel,
            conv_bias,
            time_w,
            time_b,
            alpha,
            beta,
        }
    }

    pub fn from_f64(
        kernel: [[f64; KW]; KH],
        conv_bias: f64,
        time_w: [f64; TE],
        time_b: f64,
        alpha: f64,
        beta: f64,
        scale: i64,
    ) -> Self {
        let mut kernel_i = [[0i64; KW]; KH];
        let mut time_w_i = [0i64; TE];
        for r in 0..KH {
            for c in 0..KW {
                kernel_i[r][c] = encode_f64_round(kernel[r][c], scale);
            }
        }
        for j in 0..TE {
            time_w_i[j] = encode_f64_round(time_w[j], scale);
        }
        Self::new(
            kernel_i,
            encode_f64_round(conv_bias, scale),
            time_w_i,
            encode_f64_round(time_b, scale),
            encode_f64_round(alpha, scale),
            encode_f64_round(beta, scale),
        )
    }

    pub fn block_len() -> usize {
        KH * KW + 1 + TE + 1 + 2
    }

    pub fn flatten_i64(&self) -> Vec<i64> {
        let mut out = Vec::with_capacity(Self::block_len());
        for r in 0..KH {
            for c in 0..KW {
                out.push(self.kernel[r][c]);
            }
        }
        out.push(self.conv_bias);
        out.extend(self.time_w);
        out.push(self.time_b);
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

impl<const TE: usize, const KH: usize, const KW: usize> FixedDenoiseConvPublicParams<TE, KH, KW> {
    pub fn new(
        params_seq: Vec<FixedDenoiseConvStepParams<TE, KH, KW>>,
        time_table: Vec<[i64; TE]>,
        config: FixedPointConfig,
        padding: Conv2dPadding,
        real_shape: Conv2dRealShape,
        te_real: usize,
    ) -> Self {
        assert_eq!(params_seq.len(), time_table.len());
        assert!(te_real <= TE);
        assert!(real_shape.kh_real <= KH);
        assert!(real_shape.kw_real <= KW);
        Self {
            params_seq,
            time_table,
            config,
            padding,
            real_shape,
            te_real,
            update_mode: DenoiseUpdateMode::DoubleFloor,
        }
    }

    pub fn with_update_mode(mut self, update_mode: DenoiseUpdateMode) -> Self {
        self.update_mode = update_mode;
        self
    }

    pub fn total_iters(&self) -> usize {
        self.params_seq.len()
    }

    pub fn block_len(&self) -> usize {
        FixedDenoiseConvStepParams::<TE, KH, KW>::block_len()
    }
}
