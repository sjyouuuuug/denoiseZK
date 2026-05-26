use ff::{Field, PrimeField};
use nova_snark::{
    frontend::{num::AllocatedNum, ConstraintSystem, SynthesisError},
    traits::{circuit::StepCircuit, Group},
};

use crate::{
    commitment::{synthesize_toy_hash_sequence, TOY_HASH_BASE_U64},
    fixed_point::FixedPointConfig,
    public_state::PublicStateLayout,
};

use super::{
    circuit::{synthesize_fixed_point_mlp_iteration, PublicFixedPointMlpClippedReluCircuit},
    params::FixedMlpClippedReluStepParams,
    trace::FixedPointMlpClippedReluIteration,
};

#[derive(Clone, Debug)]
pub struct PublicFixedPointMlpCommitmentCircuit<G: Group, const N: usize, const H: usize> {
    pub num_iters_per_step: usize,
    pub total_iters: usize,
    pub config: FixedPointConfig,
    pub clipped_relu_table: crate::clipped_relu::ClippedReluLookupTable,
    pub seq: Vec<FixedPointMlpClippedReluIteration<G::Scalar, N, H>>,
    pub param_hash_witnesses: Vec<G::Scalar>,
}

fn param_block_len<const N: usize, const H: usize>() -> usize {
    FixedMlpClippedReluStepParams::<N, H>::block_len()
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
    for PublicFixedPointMlpCommitmentCircuit<G, N, H>
{
    fn arity(&self) -> usize {
        PublicStateLayout::new_with_commitment(
            N,
            true,
            true,
            self.total_iters,
            param_block_len::<N, H>(),
            0,
        )
        .state_len()
    }

    fn synthesize<CS: ConstraintSystem<G::Scalar>>(
        &self,
        cs: &mut CS,
        z: &[AllocatedNum<G::Scalar>],
    ) -> Result<Vec<AllocatedNum<G::Scalar>>, SynthesisError> {
        assert_eq!(
            self.seq.len(),
            self.num_iters_per_step,
            "witness length must equal num_iters_per_step"
        );
        let p = param_block_len::<N, H>();
        let layout = PublicStateLayout::new_with_commitment(N, true, true, self.total_iters, p, 0);
        layout.assert_state_len(z.len());
        assert_eq!(
            self.param_hash_witnesses.len(),
            self.total_iters * p,
            "hash witness length must equal full params queue length"
        );

        let commitment = &z[layout.commitment_index().expect("commitment present")];
        synthesize_toy_hash_sequence(
            &mut cs.namespace(|| "mlp_commitment_hash_params"),
            &z[layout.params_range()],
            G::Scalar::from(TOY_HASH_BASE_U64),
            G::Scalar::ZERO,
            commitment,
            &self.param_hash_witnesses,
            "mlp_commitment_hash_params",
        )?;

        let mut x_i = z[layout.x_range()].to_vec();
        for local_step in 0..self.num_iters_per_step {
            let base = layout.param_block_range(local_step).start;
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
                &format!("mlp_commitment_local_{local_step}"),
            )?;
        }

        let mut z_out = Vec::with_capacity(layout.state_len());
        z_out.extend(x_i);
        z_out.extend(z[layout.y_range().expect("output present")].iter().cloned());
        z_out.push(commitment.clone());
        z_out.push(z[layout.t_index()].clone());

        let shift_blocks = self.num_iters_per_step;
        let remaining_blocks = self.total_iters.saturating_sub(shift_blocks);
        for block_idx in 0..remaining_blocks {
            let src_base = layout.param_block_range(block_idx + shift_blocks).start;
            for offset in 0..p {
                z_out.push(z[src_base + offset].clone());
            }
        }
        for pad_block in 0..shift_blocks {
            for offset in 0..p {
                z_out.push(alloc_zero(cs, || {
                    format!("mlp_commitment_pad_zero_block_{pad_block}_offset_{offset}")
                })?);
            }
        }
        Ok(z_out)
    }
}

impl<G: Group, const N: usize, const H: usize> From<PublicFixedPointMlpCommitmentCircuit<G, N, H>>
    for PublicFixedPointMlpClippedReluCircuit<G, N, H>
{
    fn from(value: PublicFixedPointMlpCommitmentCircuit<G, N, H>) -> Self {
        Self {
            num_iters_per_step: value.num_iters_per_step,
            total_iters: value.total_iters,
            config: value.config,
            clipped_relu_table: value.clipped_relu_table,
            seq: value.seq,
        }
    }
}
