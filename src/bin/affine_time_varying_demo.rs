use denoise::nova_ivc::{E1, E2, G1, S1, S2};
use ff::{Field, PrimeField};
use nova_snark::{
    frontend::{num::AllocatedNum, ConstraintSystem, SynthesisError},
    nova::{CompressedSNARK, PublicParams, RecursiveSNARK},
    traits::{circuit::StepCircuit, snark::RelaxedR1CSSNARKTrait, Engine, Group},
};
use std::time::Instant;

// ======================================================
// Public-parameter version:
// prove x_{i+1} = A_i x_i + b_i for a PUBLIC parameter sequence.
//
// Public state layout:
// z = [x | params_0 | params_1 | ... | params_{T-1}]
//
// where each params_i is flattened as:
// [A_i row-major | b_i]
//
// Each recursive step consumes `num_iters_per_step` parameter blocks
// from the front, computes the corresponding local iterations,
// and outputs:
//
// z' = [x' | params_{K} | params_{K+1} | ... | params_{T-1} | zero-padding]
//
// This keeps the StepCircuit shape fixed across recursion.
// ======================================================

#[derive(Clone, Debug)]
struct AffineStepParams<F: Field + Copy, const N: usize> {
    a_i: [[F; N]; N],
    b_i: [F; N],
}

impl<F: Field + Copy, const N: usize> AffineStepParams<F, N> {
    fn new(a_i: [[F; N]; N], b_i: [F; N]) -> Self {
        Self { a_i, b_i }
    }

    fn zero() -> Self {
        Self {
            a_i: [[F::ZERO; N]; N],
            b_i: [F::ZERO; N],
        }
    }

    fn flatten(&self) -> Vec<F> {
        let mut out = Vec::with_capacity(N * N + N);
        for r in 0..N {
            for c in 0..N {
                out.push(self.a_i[r][c]);
            }
        }
        for r in 0..N {
            out.push(self.b_i[r]);
        }
        out
    }
}

#[derive(Clone, Debug)]
struct TimeVaryingAffineWitness<F: Field + Copy, const N: usize> {
    x_i_plus_1: [F; N],
}

#[derive(Clone, Debug)]
struct PublicTimeVaryingAffineCircuit<G: Group, const N: usize> {
    // one recursive step performs this many local affine iterations
    num_iters_per_step: usize,
    // total number of public parameter blocks carried in the state
    total_iters: usize,
    // witness only stores the per-local-step next states
    seq: Vec<TimeVaryingAffineWitness<G::Scalar, N>>,
}

fn params_block_len<const N: usize>() -> usize {
    N * N + N
}

fn state_len<const N: usize>(total_iters: usize) -> usize {
    N + total_iters * params_block_len::<N>()
}

fn apply_affine<F: Field + Copy, const N: usize>(
    a: &[[F; N]; N],
    b: &[F; N],
    x: &[F; N],
) -> [F; N] {
    let mut out = [F::ZERO; N];
    for r in 0..N {
        let mut acc = b[r];
        for c in 0..N {
            acc += a[r][c] * x[c];
        }
        out[r] = acc;
    }
    out
}

/// Generate the public initial state z0 and the private witness trajectory.
///
/// z0 = [x0 | flattened params_seq]
fn generate_public_trace<F: Field + Copy, const N: usize>(
    params_seq: &[AffineStepParams<F, N>],
    x0: [F; N],
) -> (Vec<F>, Vec<TimeVaryingAffineWitness<F, N>>) {
    let mut z0 = x0.to_vec();
    for params in params_seq {
        z0.extend(params.flatten());
    }

    let mut x_i = x0;
    let mut seq = Vec::with_capacity(params_seq.len());

    for params in params_seq {
        let x_next = apply_affine(&params.a_i, &params.b_i, &x_i);
        seq.push(TimeVaryingAffineWitness { x_i_plus_1: x_next });
        x_i = x_next;
    }

    (z0, seq)
}

/// Chunk the witness trajectory into one circuit per recursive step.
fn build_step_circuits<const N: usize>(
    witness_trace: &[TimeVaryingAffineWitness<<E1 as Engine>::Scalar, N>],
    num_steps: usize,
    num_iters_per_step: usize,
    total_iters: usize,
) -> Vec<PublicTimeVaryingAffineCircuit<G1, N>> {
    assert_eq!(
        witness_trace.len(),
        num_steps * num_iters_per_step,
        "witness_trace length must equal num_steps * num_iters_per_step"
    );

    (0..num_steps)
        .map(|i| PublicTimeVaryingAffineCircuit {
            num_iters_per_step,
            total_iters,
            seq: (0..num_iters_per_step)
                .map(|j| witness_trace[i * num_iters_per_step + j].clone())
                .collect(),
        })
        .collect()
}

fn alloc_zero<CS, F>(
    cs: &mut CS,
    name: impl FnOnce() -> String,
) -> Result<AllocatedNum<F>, SynthesisError>
where
    F: PrimeField,
    CS: ConstraintSystem<F>,
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

impl<G: Group, const N: usize> StepCircuit<G::Scalar> for PublicTimeVaryingAffineCircuit<G, N> {
    fn arity(&self) -> usize {
        state_len::<N>(self.total_iters)
    }

    fn synthesize<CS: ConstraintSystem<G::Scalar>>(
        &self,
        cs: &mut CS,
        z: &[AllocatedNum<G::Scalar>],
    ) -> Result<Vec<AllocatedNum<G::Scalar>>, SynthesisError> {
        let expected_len = state_len::<N>(self.total_iters);
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

        let p = params_block_len::<N>();

        // current x lives in the first N slots of z
        let mut x_i = z[0..N].to_vec();

        // perform num_iters_per_step local iterations
        for local_step in 0..self.num_iters_per_step {
            // parameter block for this local step is always taken
            // from the front part of the public parameter queue
            let base = N + local_step * p;

            let mut x_next = Vec::with_capacity(N);

            for r in 0..N {
                // b_i[r] is public, read from z
                let b_var = z[base + N * N + r].clone();

                let mut acc = b_var;
                for c in 0..N {
                    // A_i[r][c] is public, read from z
                    let a_var = z[base + r * N + c].clone();

                    // product = A_i[r][c] * x_i[c]
                    let product = a_var.mul(
                        cs.namespace(|| {
                            format!("a_times_x_local_{}_row_{}_col_{}", local_step, r, c)
                        }),
                        &x_i[c],
                    )?;

                    acc = acc.add(
                        cs.namespace(|| {
                            format!("row_acc_local_{}_row_{}_col_{}", local_step, r, c)
                        }),
                        &product,
                    )?;
                }

                // witness for x_{i+1}[r]
                let out = AllocatedNum::alloc(
                    cs.namespace(|| format!("x_next_local_{}_row_{}", local_step, r)),
                    || Ok(self.seq[local_step].x_i_plus_1[r]),
                )?;

                // enforce acc = out
                cs.enforce(
                    || format!("affine_step_output_local_{}_row_{}", local_step, r),
                    |lc| lc + acc.get_variable(),
                    |lc| lc + CS::one(),
                    |lc| lc + out.get_variable(),
                );

                x_next.push(out);
            }

            x_i = x_next;
        }

        // output state z':
        // [final x | remaining public params shifted left by num_iters_per_step blocks | zero padding]
        let mut z_out = Vec::with_capacity(expected_len);

        // first N entries are the updated x
        z_out.extend(x_i);

        // carry the remaining parameter blocks forward
        let shift_blocks = self.num_iters_per_step;
        let remaining_blocks = self.total_iters.saturating_sub(shift_blocks);

        // copy remaining public parameter variables
        for block_idx in 0..remaining_blocks {
            let src_base = N + (block_idx + shift_blocks) * p;
            for t in 0..p {
                z_out.push(z[src_base + t].clone());
            }
        }

        // append zero padding for the consumed blocks
        for pad_block in 0..shift_blocks {
            for t in 0..p {
                let zero = alloc_zero(cs, || format!("pad_zero_block_{}_offset_{}", pad_block, t))?;
                z_out.push(zero);
            }
        }

        Ok(z_out)
    }
}

fn build_placeholder_circuit<const N: usize>(
    total_iters: usize,
    num_iters_per_step: usize,
) -> PublicTimeVaryingAffineCircuit<G1, N> {
    PublicTimeVaryingAffineCircuit {
        num_iters_per_step,
        total_iters,
        seq: vec![
            TimeVaryingAffineWitness {
                x_i_plus_1: [<E1 as Engine>::Scalar::ZERO; N],
            };
            num_iters_per_step
        ],
    }
}

fn setup_public_params<const N: usize>(
    circuit: &PublicTimeVaryingAffineCircuit<G1, N>,
) -> PublicParams<E1, E2, PublicTimeVaryingAffineCircuit<G1, N>> {
    PublicParams::<E1, E2, PublicTimeVaryingAffineCircuit<G1, N>>::setup(
        circuit,
        &*S1::ck_floor(),
        &*S2::ck_floor(),
    )
    .expect("failed to setup public parameters")
}

fn run_recursive<const N: usize>(
    pp: &PublicParams<E1, E2, PublicTimeVaryingAffineCircuit<G1, N>>,
    circuits: &[PublicTimeVaryingAffineCircuit<G1, N>],
    z0: &[<E1 as Engine>::Scalar],
) -> RecursiveSNARK<E1, E2, PublicTimeVaryingAffineCircuit<G1, N>> {
    assert!(!circuits.is_empty(), "circuits must not be empty");

    let mut recursive_snark =
        RecursiveSNARK::<E1, E2, PublicTimeVaryingAffineCircuit<G1, N>>::new(pp, &circuits[0], z0)
            .expect("failed to initialize recursive SNARK");

    for (i, circuit) in circuits.iter().enumerate() {
        let start = Instant::now();
        let res = recursive_snark.prove_step(pp, circuit);
        assert!(res.is_ok(), "prove_step failed at step {i}");
        println!("RecursiveSNARK::prove_step {i}: took {:?}", start.elapsed());
    }

    recursive_snark
}

fn verify_recursive<const N: usize>(
    recursive_snark: &RecursiveSNARK<E1, E2, PublicTimeVaryingAffineCircuit<G1, N>>,
    pp: &PublicParams<E1, E2, PublicTimeVaryingAffineCircuit<G1, N>>,
    num_steps: usize,
    z0: &[<E1 as Engine>::Scalar],
) {
    let start = Instant::now();
    let res = recursive_snark.verify(pp, num_steps, z0);
    println!(
        "RecursiveSNARK::verify: {:?}, took {:?}",
        res.is_ok(),
        start.elapsed()
    );
    assert!(res.is_ok(), "recursive verification failed");
}

fn compress_and_verify<const N: usize>(
    pp: &PublicParams<E1, E2, PublicTimeVaryingAffineCircuit<G1, N>>,
    recursive_snark: &RecursiveSNARK<E1, E2, PublicTimeVaryingAffineCircuit<G1, N>>,
    num_steps: usize,
    z0: &[<E1 as Engine>::Scalar],
) -> usize {
    let (pk, vk) = CompressedSNARK::<_, _, _, S1, S2>::setup(pp)
        .expect("failed to setup compressed SNARK keys");

    let start = Instant::now();
    let compressed_snark = CompressedSNARK::<_, _, _, S1, S2>::prove(pp, &pk, recursive_snark)
        .expect("failed to produce compressed SNARK");
    println!("CompressedSNARK::prove took {:?}", start.elapsed());

    let start = Instant::now();
    let res = compressed_snark.verify(&vk, num_steps, z0);
    println!(
        "CompressedSNARK::verify: {:?}, took {:?}",
        res.is_ok(),
        start.elapsed()
    );
    assert!(res.is_ok(), "compressed verification failed");

    bincode::serde::encode_to_vec(&compressed_snark, bincode::config::legacy())
        .expect("failed to serialize compressed SNARK")
        .len()
}

fn main() {
    const N: usize = 2;

    println!("Nova public-parameter affine demo: x_(i+1) = A_i x_i + b_i");
    println!("=========================================================");

    let num_steps = 3;
    let num_iters_per_step = 2;
    let total_iters = num_steps * num_iters_per_step;

    let one = <E1 as Engine>::Scalar::ONE;
    let two = <E1 as Engine>::Scalar::from(2u64);
    let three = <E1 as Engine>::Scalar::from(3u64);
    let four = <E1 as Engine>::Scalar::from(4u64);
    let five = <E1 as Engine>::Scalar::from(5u64);
    let seven = <E1 as Engine>::Scalar::from(7u64);
    let eight = <E1 as Engine>::Scalar::from(8u64);

    let params_seq: Vec<AffineStepParams<<E1 as Engine>::Scalar, N>> = vec![
        AffineStepParams::new([[two, one], [one, three]], [one, five]),
        AffineStepParams::new([[three, one], [two, one]], [two, one]),
        AffineStepParams::new([[one, four], [one, two]], [three, one]),
        AffineStepParams::new([[two, two], [one, one]], [one, two]),
        AffineStepParams::new([[four, one], [three, one]], [two, three]),
        AffineStepParams::new([[one, five], [two, one]], [seven, eight]),
    ];
    assert_eq!(
        params_seq.len(),
        total_iters,
        "params_seq length must match total iterations"
    );

    let x0: [<E1 as Engine>::Scalar; N] = [one, two];

    println!("Preparing public parameters...");
    let start = Instant::now();
    let placeholder = build_placeholder_circuit(total_iters, num_iters_per_step);
    let pp = setup_public_params(&placeholder);
    println!("PublicParams::setup took {:?}", start.elapsed());
    println!(
        "Number of constraints per step (primary, secondary): {:?}",
        pp.num_constraints()
    );
    println!(
        "Number of variables per step (primary, secondary): {:?}",
        pp.num_variables()
    );

    println!("Generating public trace...");
    let (z0, witness_trace) = generate_public_trace(&params_seq, x0);
    let circuits = build_step_circuits(&witness_trace, num_steps, num_iters_per_step, total_iters);

    println!("Public state length = {}", z0.len());

    println!("Generating RecursiveSNARK...");
    let recursive_snark = run_recursive(&pp, &circuits, &z0);

    println!("Verifying RecursiveSNARK...");
    verify_recursive(&recursive_snark, &pp, num_steps, &z0);

    println!("Generating and verifying CompressedSNARK...");
    let proof_size = compress_and_verify(&pp, &recursive_snark, num_steps, &z0);
    println!("CompressedSNARK size: {} bytes", proof_size);
    println!("=========================================================");
}
