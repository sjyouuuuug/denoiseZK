use crate::fixed_point::rescale_with_remainder;

use super::mode::DenoiseUpdateMode;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DenoiseUpdateWitness<const N: usize> {
    DoubleFloor {
        alpha_raw: [i64; N],
        alpha_q: [i64; N],
        alpha_r: [i64; N],
        beta_raw: [i64; N],
        beta_q: [i64; N],
        beta_r: [i64; N],
        x_next: [i64; N],
    },
    FusedFloor {
        fused_raw: [i64; N],
        fused_q: [i64; N],
        fused_r: [i64; N],
        x_next: [i64; N],
    },
}

impl<const N: usize> DenoiseUpdateWitness<N> {
    pub fn zero_double_floor() -> Self {
        Self::DoubleFloor {
            alpha_raw: [0; N],
            alpha_q: [0; N],
            alpha_r: [0; N],
            beta_raw: [0; N],
            beta_q: [0; N],
            beta_r: [0; N],
            x_next: [0; N],
        }
    }

    pub fn zero_fused_floor() -> Self {
        Self::FusedFloor {
            fused_raw: [0; N],
            fused_q: [0; N],
            fused_r: [0; N],
            x_next: [0; N],
        }
    }

    pub fn x_next(&self) -> &[i64; N] {
        match self {
            Self::DoubleFloor { x_next, .. } | Self::FusedFloor { x_next, .. } => x_next,
        }
    }
}

pub fn compute_denoise_update_witness<const N: usize>(
    x: &[i64; N],
    epsilon: &[i64; N],
    alpha: i64,
    beta: i64,
    scale: i64,
    mode: DenoiseUpdateMode,
) -> DenoiseUpdateWitness<N> {
    match mode {
        DenoiseUpdateMode::DoubleFloor => {
            let mut alpha_raw = [0i64; N];
            let mut alpha_q = [0i64; N];
            let mut alpha_r = [0i64; N];
            let mut beta_raw = [0i64; N];
            let mut beta_q = [0i64; N];
            let mut beta_r = [0i64; N];
            let mut x_next = [0i64; N];
            for j in 0..N {
                alpha_raw[j] = alpha * x[j];
                (alpha_q[j], alpha_r[j]) = rescale_with_remainder(alpha_raw[j], scale);
                beta_raw[j] = beta * epsilon[j];
                (beta_q[j], beta_r[j]) = rescale_with_remainder(beta_raw[j], scale);
                x_next[j] = alpha_q[j] + beta_q[j];
            }
            DenoiseUpdateWitness::DoubleFloor {
                alpha_raw,
                alpha_q,
                alpha_r,
                beta_raw,
                beta_q,
                beta_r,
                x_next,
            }
        }
        DenoiseUpdateMode::FusedFloor => {
            let mut fused_raw = [0i64; N];
            let mut fused_q = [0i64; N];
            let mut fused_r = [0i64; N];
            let mut x_next = [0i64; N];
            for j in 0..N {
                fused_raw[j] = alpha * x[j] + beta * epsilon[j];
                (fused_q[j], fused_r[j]) = rescale_with_remainder(fused_raw[j], scale);
                x_next[j] = fused_q[j];
            }
            DenoiseUpdateWitness::FusedFloor {
                fused_raw,
                fused_q,
                fused_r,
                x_next,
            }
        }
    }
}
