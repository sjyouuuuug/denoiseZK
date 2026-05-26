use nova_snark::{
    frontend::{num::AllocatedNum, ConstraintSystem, SynthesisError},
    traits::{circuit::StepCircuit, Group},
};

use super::{params::AffineParams, trace::AffineIteration};

#[derive(Clone, Debug)]
pub struct AffineCircuit<G: Group, const N: usize> {
    pub params: AffineParams<G::Scalar, N>,
    pub seq: Vec<AffineIteration<G::Scalar, N>>,
}

impl<G: Group, const N: usize> StepCircuit<G::Scalar> for AffineCircuit<G, N> {
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
                let out =
                    AllocatedNum::alloc(cs.namespace(|| format!("x_{}_row_{}", i + 1, r)), || {
                        Ok(self.seq[i].x_i_plus_1[r])
                    })?;

                cs.enforce(
                    || format!("affine_step_{}_row_{}", i, r),
                    |lc| {
                        let mut lc = lc + (self.params.b[r], CS::one());
                        for c in 0..N {
                            lc = lc + (self.params.a[r][c], x_i[c].get_variable());
                        }
                        lc
                    },
                    |lc| lc + CS::one(),
                    |lc| lc + out.get_variable(),
                );

                x_next.push(out);
            }

            x_i = x_next;
        }

        Ok(x_i)
    }
}
