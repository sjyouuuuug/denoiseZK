use ff::PrimeField;

use crate::{
    clipped_relu::field_from_i64,
    fixed_point::{encode_f64_round, FixedPointConfig},
    models::denoise_update::DenoiseUpdateMode,
    padding::{matrix::pad_matrix_i64, vector::pad_vector_i64},
};

#[derive(Clone, Debug)]
pub struct FixedDenoiseTimeEmbeddingStepParams<
    const N: usize,
    const TE: usize,
    const IN: usize,
    const H: usize,
> {
    pub w1: [[i64; IN]; H],
    pub b1: [i64; H],
    pub w2: [[i64; H]; N],
    pub b2: [i64; N],
    pub alpha: i64,
    pub beta: i64,
}

#[derive(Clone, Debug)]
pub struct FixedDenoiseTimeEmbeddingPublicParams<
    const N: usize,
    const TE: usize,
    const IN: usize,
    const H: usize,
> {
    pub params_seq: Vec<FixedDenoiseTimeEmbeddingStepParams<N, TE, IN, H>>,
    pub time_table: Vec<[i64; TE]>,
    pub config: FixedPointConfig,
    pub update_mode: DenoiseUpdateMode,
}

impl<const N: usize, const TE: usize, const IN: usize, const H: usize>
    FixedDenoiseTimeEmbeddingStepParams<N, TE, IN, H>
{
    pub fn new(
        w1: [[i64; IN]; H],
        b1: [i64; H],
        w2: [[i64; H]; N],
        b2: [i64; N],
        alpha: i64,
        beta: i64,
    ) -> Self {
        assert_eq!(IN, N + TE, "IN must equal N + TE");
        Self {
            w1,
            b1,
            w2,
            b2,
            alpha,
            beta,
        }
    }

    pub fn from_f64(
        w1: [[f64; IN]; H],
        b1: [f64; H],
        w2: [[f64; H]; N],
        b2: [f64; N],
        alpha: f64,
        beta: f64,
        scale: i64,
    ) -> Self {
        let mut w1_i = [[0i64; IN]; H];
        let mut b1_i = [0i64; H];
        let mut w2_i = [[0i64; H]; N];
        let mut b2_i = [0i64; N];

        for r in 0..H {
            for c in 0..IN {
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

        Self::new(
            w1_i,
            b1_i,
            w2_i,
            b2_i,
            encode_f64_round(alpha, scale),
            encode_f64_round(beta, scale),
        )
    }

    pub fn block_len() -> usize {
        H * IN + H + N * H + N + 2
    }

    pub fn flatten_i64(&self) -> Vec<i64> {
        let mut out = Vec::with_capacity(Self::block_len());
        for r in 0..H {
            for c in 0..IN {
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

pub fn pad_denoise_time_embedding_step_params<
    const N_REAL: usize,
    const TE_REAL: usize,
    const IN_REAL: usize,
    const H_REAL: usize,
    const N_MAX: usize,
    const TE_MAX: usize,
    const IN_MAX: usize,
    const H_MAX: usize,
>(
    real: FixedDenoiseTimeEmbeddingStepParams<N_REAL, TE_REAL, IN_REAL, H_REAL>,
) -> FixedDenoiseTimeEmbeddingStepParams<N_MAX, TE_MAX, IN_MAX, H_MAX> {
    assert_eq!(
        IN_REAL,
        N_REAL + TE_REAL,
        "IN_REAL must equal N_REAL + TE_REAL"
    );
    assert_eq!(IN_MAX, N_MAX + TE_MAX, "IN_MAX must equal N_MAX + TE_MAX");
    assert!(N_REAL <= N_MAX, "N_REAL must be <= N_MAX");
    assert!(TE_REAL <= TE_MAX, "TE_REAL must be <= TE_MAX");
    assert!(H_REAL <= H_MAX, "H_REAL must be <= H_MAX");

    let mut w1 = [[0i64; IN_MAX]; H_MAX];
    for r in 0..H_REAL {
        for c in 0..N_REAL {
            w1[r][c] = real.w1[r][c];
        }
        for c in 0..TE_REAL {
            w1[r][N_MAX + c] = real.w1[r][N_REAL + c];
        }
    }

    FixedDenoiseTimeEmbeddingStepParams::new(
        w1,
        pad_vector_i64::<H_REAL, H_MAX>(real.b1),
        pad_matrix_i64::<N_REAL, H_REAL, N_MAX, H_MAX>(real.w2),
        pad_vector_i64::<N_REAL, N_MAX>(real.b2),
        real.alpha,
        real.beta,
    )
}

impl<const N: usize, const TE: usize, const IN: usize, const H: usize>
    FixedDenoiseTimeEmbeddingPublicParams<N, TE, IN, H>
{
    pub fn new(
        params_seq: Vec<FixedDenoiseTimeEmbeddingStepParams<N, TE, IN, H>>,
        time_table: Vec<[i64; TE]>,
        config: FixedPointConfig,
    ) -> Self {
        assert_eq!(IN, N + TE, "IN must equal N + TE");
        assert_eq!(
            params_seq.len(),
            time_table.len(),
            "params_seq and time_table must have the same length"
        );
        Self {
            params_seq,
            time_table,
            config,
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
        FixedDenoiseTimeEmbeddingStepParams::<N, TE, IN, H>::block_len()
    }

    pub fn time_table_len(&self) -> usize {
        self.time_table.len() * TE
    }
}
