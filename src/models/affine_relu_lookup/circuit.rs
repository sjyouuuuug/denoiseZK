use nova_snark::{
    frontend::{num::AllocatedNum, ConstraintSystem, SynthesisError},
    traits::{circuit::StepCircuit, Group},
};

use crate::relu::relu_lookup;

use super::{params::AffineReluLookupParams, trace::AffineReluLookupIteration};

#[derive(Clone, Debug)]
pub struct AffineReluLookupCircuit<G: Group, const N: usize> {
    pub params: AffineReluLookupParams<G::Scalar, N>,
    pub seq: Vec<AffineReluLookupIteration<G::Scalar, N>>,
}

impl<G: Group, const N: usize> StepCircuit<G::Scalar> for AffineReluLookupCircuit<G, N> {
    fn arity(&self) -> usize {
        N
    }

    fn synthesize<CS: ConstraintSystem<G::Scalar>>(
        &self,
        cs: &mut CS,
        z: &[AllocatedNum<G::Scalar>],
    ) -> Result<Vec<AllocatedNum<G::Scalar>>, SynthesisError> {
        if self.seq.is_empty() {
            return Ok(z.to_vec());
        }

        assert_eq!(z.len(), N, "input state dimension must match circuit arity");

        let mut x_i = z.to_vec();

        for i in 0..self.seq.len() {
            let mut x_next = Vec::with_capacity(N);

            for r in 0..N {
                let affine_out = AllocatedNum::alloc(
                    cs.namespace(|| format!("affine_out_{}_row_{}", i, r)),
                    || Ok(self.seq[i].affine_out[r]),
                )?;

                cs.enforce(
                    || format!("affine_only_{}_row_{}", i, r),
                    |lc| {
                        let mut lc = lc + (self.params.b[r], CS::one());
                        for c in 0..N {
                            lc = lc + (self.params.a[r][c], x_i[c].get_variable());
                        }
                        lc
                    },
                    |lc| lc + CS::one(),
                    |lc| lc + affine_out.get_variable(),
                );

                let relu_out = relu_lookup(
                    &mut cs.namespace(|| format!("relu_lookup_{}_row_{}", i, r)),
                    &affine_out,
                    self.seq[i].affine_out_int[r],
                    self.seq[i].x_i_plus_1[r],
                    &self.params.relu_table,
                    &format!("iter_{}_row_{}", i, r),
                )?;

                x_next.push(relu_out);
            }

            x_i = x_next;
        }

        Ok(x_i)
    }
}
