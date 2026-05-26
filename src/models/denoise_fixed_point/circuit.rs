use ff::PrimeField;
use nova_snark::{
    frontend::{num::AllocatedNum, ConstraintSystem, SynthesisError},
    traits::{circuit::StepCircuit, Group},
};

use crate::{
    clipped_relu::ClippedReluLookupTable,
    fixed_point::{enforce_floor_rescale, enforce_signed_range_bits, FixedPointConfig},
    mlp_fixed_point_clipped_relu_lookup::circuit::synthesize_fixed_point_mlp_iteration,
};

use super::{params::FixedDenoiseStepParams, trace::FixedDenoiseIteration};

#[derive(Clone, Debug)]
pub struct PublicFixedPointDenoiseCircuit<G: Group, const N: usize, const H: usize> {
    pub num_iters_per_step: usize,
    pub total_iters: usize,
    pub config: FixedPointConfig,
    pub clipped_relu_table: ClippedReluLookupTable,
    pub seq: Vec<FixedDenoiseIteration<G::Scalar, N, H>>,
}

fn param_block_len<const N: usize, const H: usize>() -> usize {
    FixedDenoiseStepParams::<N, H>::block_len()
}

fn mlp_block_len<const N: usize, const H: usize>() -> usize {
    param_block_len::<N, H>() - 2
}

fn state_len<const N: usize, const H: usize>(total_iters: usize) -> usize {
    N + total_iters * param_block_len::<N, H>()
}

fn alloc_zero<CS, F>(
    cs: &mut CS,
    name: impl FnOnce() -> String,
) -> Result<AllocatedNum<F>, SynthesisError>
where
    CS: ConstraintSystem<F>,
    F: PrimeField,
{
    let zero = AllocatedNum::alloc(cs.namespace(name), || Ok(F::ZERO))?;
    cs.enforce(
        || "enforce_zero",
        |lc| lc + zero.get_variable(),
        |lc| lc + CS::one(),
        |lc| lc,
    );
    Ok(zero)
}

impl<G: Group, const N: usize, const H: usize> StepCircuit<G::Scalar>
    for PublicFixedPointDenoiseCircuit<G, N, H>
{
    fn arity(&self) -> usize {
        state_len::<N, H>(self.total_iters)
    }

    fn synthesize<CS: ConstraintSystem<G::Scalar>>(
        &self,
        cs: &mut CS,
        z: &[AllocatedNum<G::Scalar>],
    ) -> Result<Vec<AllocatedNum<G::Scalar>>, SynthesisError> {
        let expected_len = state_len::<N, H>(self.total_iters);
        assert_eq!(
            z.len(),
            expected_len,
            "input state dimension must match circuit arity"
        );
        assert_eq!(
            self.seq.len(),
            self.num_iters_per_step,
            "witness length must equal num_iters_per_step"
        );

        let p = param_block_len::<N, H>();
        let mlp_p = mlp_block_len::<N, H>();
        let mut x_i = z[0..N].to_vec();

        for local_step in 0..self.num_iters_per_step {
            for (j, value) in x_i.iter().enumerate() {
                enforce_signed_range_bits(
                    &mut cs.namespace(|| format!("denoise_x_i_range_local_{local_step}_coord_{j}")),
                    value,
                    self.seq[local_step].x_i_int[j],
                    self.config.value_min,
                    self.config.value_max,
                    &format!("denoise_x_i_range_local_{local_step}_coord_{j}"),
                )?;
            }

            let base = N + local_step * p;
            let w1_base = base;
            let b1_base = w1_base + H * N;
            let w2_base = b1_base + H;
            let b2_base = w2_base + N * H;
            let alpha_idx = base + mlp_p;
            let beta_idx = alpha_idx + 1;

            let epsilon = synthesize_fixed_point_mlp_iteration(
                cs,
                z,
                &x_i,
                w1_base,
                b1_base,
                w2_base,
                b2_base,
                &self.seq[local_step].mlp_witness,
                &self.config,
                &self.clipped_relu_table,
                &format!("denoise_mlp_local_{local_step}"),
            )?;

            let mut x_next = Vec::with_capacity(N);
            for j in 0..N {
                enforce_signed_range_bits(
                    &mut cs.namespace(|| {
                        format!("denoise_epsilon_range_local_{local_step}_coord_{j}")
                    }),
                    &epsilon[j],
                    self.seq[local_step].epsilon_int[j],
                    self.config.value_min,
                    self.config.value_max,
                    &format!("denoise_epsilon_range_local_{local_step}_coord_{j}"),
                )?;

                let alpha_raw = z[alpha_idx].mul(
                    cs.namespace(|| format!("denoise_alpha_times_x_local_{local_step}_coord_{j}")),
                    &x_i[j],
                )?;
                let alpha_q = enforce_floor_rescale(
                    &mut cs.namespace(|| {
                        format!("denoise_alpha_rescale_local_{local_step}_coord_{j}")
                    }),
                    &alpha_raw,
                    self.seq[local_step].alpha_x_int[j],
                    self.seq[local_step].alpha_remainder_int[j],
                    self.config.scale,
                    self.config.quotient_min,
                    self.config.quotient_max,
                    &format!("denoise_alpha_rescale_local_{local_step}_coord_{j}"),
                )?;

                let beta_raw = z[beta_idx].mul(
                    cs.namespace(|| {
                        format!("denoise_beta_times_epsilon_local_{local_step}_coord_{j}")
                    }),
                    &epsilon[j],
                )?;
                let beta_q = enforce_floor_rescale(
                    &mut cs
                        .namespace(|| format!("denoise_beta_rescale_local_{local_step}_coord_{j}")),
                    &beta_raw,
                    self.seq[local_step].beta_epsilon_int[j],
                    self.seq[local_step].beta_remainder_int[j],
                    self.config.scale,
                    self.config.quotient_min,
                    self.config.quotient_max,
                    &format!("denoise_beta_rescale_local_{local_step}_coord_{j}"),
                )?;

                let sum = alpha_q.add(
                    cs.namespace(|| format!("denoise_update_sum_local_{local_step}_coord_{j}")),
                    &beta_q,
                )?;
                let out = AllocatedNum::alloc(
                    cs.namespace(|| format!("denoise_x_next_local_{local_step}_coord_{j}")),
                    || Ok(self.seq[local_step].x_i_plus_1[j]),
                )?;
                cs.enforce(
                    || format!("denoise_x_next_check_local_{local_step}_coord_{j}"),
                    |lc| lc + sum.get_variable(),
                    |lc| lc + CS::one(),
                    |lc| lc + out.get_variable(),
                );
                enforce_signed_range_bits(
                    &mut cs
                        .namespace(|| format!("denoise_x_next_range_local_{local_step}_coord_{j}")),
                    &out,
                    self.seq[local_step].x_i_plus_1_int[j],
                    self.config.value_min,
                    self.config.value_max,
                    &format!("denoise_x_next_range_local_{local_step}_coord_{j}"),
                )?;
                x_next.push(out);
            }

            x_i = x_next;
        }

        let mut z_out = Vec::with_capacity(expected_len);
        z_out.extend(x_i);

        let shift_blocks = self.num_iters_per_step;
        let remaining_blocks = self.total_iters.saturating_sub(shift_blocks);
        for block_idx in 0..remaining_blocks {
            let src_base = N + (block_idx + shift_blocks) * p;
            for t in 0..p {
                z_out.push(z[src_base + t].clone());
            }
        }

        for pad_block in 0..shift_blocks {
            for t in 0..p {
                let zero = alloc_zero(cs, || {
                    format!("denoise_pad_zero_block_{pad_block}_offset_{t}")
                })?;
                z_out.push(zero);
            }
        }

        Ok(z_out)
    }
}
