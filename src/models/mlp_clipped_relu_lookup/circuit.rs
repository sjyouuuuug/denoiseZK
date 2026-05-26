use ff::PrimeField;
use nova_snark::{
    frontend::{num::AllocatedNum, ConstraintSystem, SynthesisError},
    traits::{circuit::StepCircuit, Group},
};

use crate::clipped_relu::{clipped_relu_lookup, ClippedReluLookupTable};

use super::{params::IntMlpClippedReluStepParams, trace::MlpClippedReluIteration};

#[derive(Clone, Debug)]
pub struct PublicMlpClippedReluCircuit<G: Group, const N: usize, const H: usize> {
    pub num_iters_per_step: usize,
    pub total_iters: usize,
    pub clipped_relu_table: ClippedReluLookupTable,
    pub seq: Vec<MlpClippedReluIteration<G::Scalar, N, H>>,
}

fn param_block_len<const N: usize, const H: usize>() -> usize {
    IntMlpClippedReluStepParams::<N, H>::block_len()
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
    for PublicMlpClippedReluCircuit<G, N, H>
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

            // First affine layer: u = W1 x + b1, with W1 public in z.
            let mut hidden_act = Vec::with_capacity(H);
            for r in 0..H {
                let mut acc = z[b1_base + r].clone();
                for c in 0..N {
                    let w = z[w1_base + r * N + c].clone();
                    let product = w.mul(
                        cs.namespace(|| format!("w1_times_x_local_{local_step}_row_{r}_col_{c}")),
                        &x_i[c],
                    )?;
                    acc = acc.add(
                        cs.namespace(|| {
                            format!("hidden_affine_acc_local_{local_step}_row_{r}_col_{c}")
                        }),
                        &product,
                    )?;
                }

                let hidden_affine = AllocatedNum::alloc(
                    cs.namespace(|| format!("hidden_affine_local_{local_step}_row_{r}")),
                    || Ok(self.seq[local_step].hidden_affine[r]),
                )?;

                cs.enforce(
                    || format!("hidden_affine_check_local_{local_step}_row_{r}"),
                    |lc| lc + acc.get_variable(),
                    |lc| lc + CS::one(),
                    |lc| lc + hidden_affine.get_variable(),
                );

                let act = clipped_relu_lookup(
                    &mut cs.namespace(|| {
                        format!("hidden_clipped_relu_lookup_local_{local_step}_row_{r}")
                    }),
                    &hidden_affine,
                    self.seq[local_step].hidden_affine_int[r],
                    self.seq[local_step].hidden_act[r],
                    &self.clipped_relu_table,
                    &format!("hidden_act_local_{local_step}_row_{r}"),
                )?;
                hidden_act.push(act);
            }

            // Second affine layer: x_next = W2 h + b2, with W2 public in z.
            let mut x_next = Vec::with_capacity(N);
            for r in 0..N {
                let mut acc = z[b2_base + r].clone();
                for c in 0..H {
                    let w = z[w2_base + r * H + c].clone();
                    let product = w.mul(
                        cs.namespace(|| format!("w2_times_h_local_{local_step}_row_{r}_col_{c}")),
                        &hidden_act[c],
                    )?;
                    acc = acc.add(
                        cs.namespace(|| {
                            format!("output_affine_acc_local_{local_step}_row_{r}_col_{c}")
                        }),
                        &product,
                    )?;
                }

                let out = AllocatedNum::alloc(
                    cs.namespace(|| format!("x_next_local_{local_step}_row_{r}")),
                    || Ok(self.seq[local_step].x_i_plus_1[r]),
                )?;

                cs.enforce(
                    || format!("output_affine_check_local_{local_step}_row_{r}"),
                    |lc| lc + acc.get_variable(),
                    |lc| lc + CS::one(),
                    |lc| lc + out.get_variable(),
                );

                x_next.push(out);
            }

            x_i = x_next;
        }

        // Output state: [x_final | remaining public parameter blocks shifted left | zero padding].
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
                let zero = alloc_zero(cs, || format!("pad_zero_block_{pad_block}_offset_{t}"))?;
                z_out.push(zero);
            }
        }

        Ok(z_out)
    }
}
