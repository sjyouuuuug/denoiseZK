use ff::PrimeField;
use nova_snark::{
    frontend::{num::AllocatedNum, ConstraintSystem, SynthesisError},
    traits::{circuit::StepCircuit, Group},
};

use crate::{
    clipped_relu::{clipped_relu_lookup, ClippedReluLookupTable},
    fixed_point::{enforce_floor_rescale, FixedPointConfig},
};

use super::{params::FixedMlpClippedReluStepParams, trace::FixedPointMlpClippedReluIteration};

#[derive(Clone, Debug)]
pub struct PublicFixedPointMlpClippedReluCircuit<G: Group, const N: usize, const H: usize> {
    pub num_iters_per_step: usize,
    pub total_iters: usize,
    pub config: FixedPointConfig,
    pub clipped_relu_table: ClippedReluLookupTable,
    pub seq: Vec<FixedPointMlpClippedReluIteration<G::Scalar, N, H>>,
}

fn param_block_len<const N: usize, const H: usize>() -> usize {
    FixedMlpClippedReluStepParams::<N, H>::block_len()
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

pub fn synthesize_fixed_point_mlp_iteration<CS, F, const N: usize, const H: usize>(
    cs: &mut CS,
    z: &[AllocatedNum<F>],
    x_i: &[AllocatedNum<F>],
    w1_base: usize,
    b1_base: usize,
    w2_base: usize,
    b2_base: usize,
    witness: &FixedPointMlpClippedReluIteration<F, N, H>,
    config: &FixedPointConfig,
    clipped_relu_table: &ClippedReluLookupTable,
    prefix: &str,
) -> Result<Vec<AllocatedNum<F>>, SynthesisError>
where
    CS: ConstraintSystem<F>,
    F: PrimeField,
{
    let mut hidden_act = Vec::with_capacity(H);
    for r in 0..H {
        let mut acc = z[w1_base + r * N].mul(
            cs.namespace(|| format!("{prefix}_w1_times_x_row_{r}_col_0")),
            &x_i[0],
        )?;
        for c in 1..N {
            let w = z[w1_base + r * N + c].clone();
            let product = w.mul(
                cs.namespace(|| format!("{prefix}_w1_times_x_row_{r}_col_{c}")),
                &x_i[c],
            )?;
            acc = acc.add(
                cs.namespace(|| format!("{prefix}_hidden_raw_acc_row_{r}_col_{c}")),
                &product,
            )?;
        }

        let hidden_raw = AllocatedNum::alloc(
            cs.namespace(|| format!("{prefix}_hidden_raw_row_{r}")),
            || Ok(witness.hidden_raw[r]),
        )?;
        cs.enforce(
            || format!("{prefix}_hidden_raw_check_row_{r}"),
            |lc| lc + acc.get_variable(),
            |lc| lc + CS::one(),
            |lc| lc + hidden_raw.get_variable(),
        );

        let hidden_q = enforce_floor_rescale(
            &mut cs.namespace(|| format!("{prefix}_hidden_rescale_row_{r}")),
            &hidden_raw,
            witness.hidden_quotient_int[r],
            witness.hidden_remainder_int[r],
            config.scale,
            config.quotient_min,
            config.quotient_max,
            &format!("{prefix}_hidden_rescale_row_{r}"),
        )?;

        let hidden_affine = hidden_q.add(
            cs.namespace(|| format!("{prefix}_hidden_add_bias_row_{r}")),
            &z[b1_base + r],
        )?;

        let act = clipped_relu_lookup(
            &mut cs.namespace(|| format!("{prefix}_hidden_clipped_relu_lookup_row_{r}")),
            &hidden_affine,
            witness.hidden_affine_int[r],
            witness.hidden_act[r],
            clipped_relu_table,
            &format!("{prefix}_hidden_act_row_{r}"),
        )?;
        hidden_act.push(act);
    }

    let mut output = Vec::with_capacity(N);
    for r in 0..N {
        let mut acc = z[w2_base + r * H].mul(
            cs.namespace(|| format!("{prefix}_w2_times_h_row_{r}_col_0")),
            &hidden_act[0],
        )?;
        for c in 1..H {
            let w = z[w2_base + r * H + c].clone();
            let product = w.mul(
                cs.namespace(|| format!("{prefix}_w2_times_h_row_{r}_col_{c}")),
                &hidden_act[c],
            )?;
            acc = acc.add(
                cs.namespace(|| format!("{prefix}_output_raw_acc_row_{r}_col_{c}")),
                &product,
            )?;
        }

        let output_raw = AllocatedNum::alloc(
            cs.namespace(|| format!("{prefix}_output_raw_row_{r}")),
            || Ok(witness.output_raw[r]),
        )?;
        cs.enforce(
            || format!("{prefix}_output_raw_check_row_{r}"),
            |lc| lc + acc.get_variable(),
            |lc| lc + CS::one(),
            |lc| lc + output_raw.get_variable(),
        );

        let output_q = enforce_floor_rescale(
            &mut cs.namespace(|| format!("{prefix}_output_rescale_row_{r}")),
            &output_raw,
            witness.output_quotient_int[r],
            witness.output_remainder_int[r],
            config.scale,
            config.quotient_min,
            config.quotient_max,
            &format!("{prefix}_output_rescale_row_{r}"),
        )?;

        let out = output_q.add(
            cs.namespace(|| format!("{prefix}_output_add_bias_row_{r}")),
            &z[b2_base + r],
        )?;
        output.push(out);
    }

    Ok(output)
}

impl<G: Group, const N: usize, const H: usize> StepCircuit<G::Scalar>
    for PublicFixedPointMlpClippedReluCircuit<G, N, H>
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
        let mut x_i = z[0..N].to_vec();

        for local_step in 0..self.num_iters_per_step {
            let base = N + local_step * p;
            let w1_base = base;
            let b1_base = w1_base + H * N;
            let w2_base = b1_base + H;
            let b2_base = w2_base + N * H;

            x_i = synthesize_fixed_point_mlp_iteration(
                cs,
                z,
                &x_i,
                w1_base,
                b1_base,
                w2_base,
                b2_base,
                &self.seq[local_step],
                &self.config,
                &self.clipped_relu_table,
                &format!("fp_mlp_local_{local_step}"),
            )?;
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
                let zero = alloc_zero(cs, || format!("fp_pad_zero_block_{pad_block}_offset_{t}"))?;
                z_out.push(zero);
            }
        }

        Ok(z_out)
    }
}
