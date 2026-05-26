use ff::PrimeField;
use nova_snark::frontend::{num::AllocatedNum, ConstraintSystem, SynthesisError};

use crate::{clipped_relu::field_from_i64, fixed_point::enforce_floor_rescale};

use super::{mode::DenoiseUpdateMode, trace::DenoiseUpdateWitness};

pub fn synthesize_denoise_update<CS, F, const N: usize>(
    cs: &mut CS,
    x_vars: &[AllocatedNum<F>],
    epsilon_vars: &[AllocatedNum<F>],
    alpha_var: &AllocatedNum<F>,
    beta_var: &AllocatedNum<F>,
    scale: i64,
    quotient_min: i64,
    quotient_max: i64,
    mode: DenoiseUpdateMode,
    witness: &DenoiseUpdateWitness<N>,
    label: &str,
) -> Result<Vec<AllocatedNum<F>>, SynthesisError>
where
    CS: ConstraintSystem<F>,
    F: PrimeField,
{
    assert_eq!(x_vars.len(), N, "x_vars length must equal N");
    assert_eq!(epsilon_vars.len(), N, "epsilon_vars length must equal N");
    let mut x_next = Vec::with_capacity(N);
    match (mode, witness) {
        (
            DenoiseUpdateMode::DoubleFloor,
            DenoiseUpdateWitness::DoubleFloor {
                alpha_q,
                alpha_r,
                beta_q,
                beta_r,
                x_next: x_next_int,
                ..
            },
        ) => {
            for j in 0..N {
                let alpha_raw = alpha_var.mul(
                    cs.namespace(|| format!("{label}_alpha_times_x_coord_{j}")),
                    &x_vars[j],
                )?;
                let alpha_q_var = enforce_floor_rescale(
                    &mut cs.namespace(|| format!("{label}_alpha_rescale_coord_{j}")),
                    &alpha_raw,
                    alpha_q[j],
                    alpha_r[j],
                    scale,
                    quotient_min,
                    quotient_max,
                    &format!("{label}_alpha_rescale_coord_{j}"),
                )?;
                let beta_raw = beta_var.mul(
                    cs.namespace(|| format!("{label}_beta_times_epsilon_coord_{j}")),
                    &epsilon_vars[j],
                )?;
                let beta_q_var = enforce_floor_rescale(
                    &mut cs.namespace(|| format!("{label}_beta_rescale_coord_{j}")),
                    &beta_raw,
                    beta_q[j],
                    beta_r[j],
                    scale,
                    quotient_min,
                    quotient_max,
                    &format!("{label}_beta_rescale_coord_{j}"),
                )?;
                let sum = alpha_q_var.add(
                    cs.namespace(|| format!("{label}_double_sum_coord_{j}")),
                    &beta_q_var,
                )?;
                let out = AllocatedNum::alloc(
                    cs.namespace(|| format!("{label}_x_next_coord_{j}")),
                    || Ok(field_from_i64::<F>(x_next_int[j])),
                )?;
                cs.enforce(
                    || format!("{label}_x_next_check_coord_{j}"),
                    |lc| lc + sum.get_variable(),
                    |lc| lc + CS::one(),
                    |lc| lc + out.get_variable(),
                );
                x_next.push(out);
            }
        }
        (
            DenoiseUpdateMode::FusedFloor,
            DenoiseUpdateWitness::FusedFloor {
                fused_q, fused_r, ..
            },
        ) => {
            for j in 0..N {
                let alpha_prod = alpha_var.mul(
                    cs.namespace(|| format!("{label}_fused_alpha_times_x_coord_{j}")),
                    &x_vars[j],
                )?;
                let beta_prod = beta_var.mul(
                    cs.namespace(|| format!("{label}_fused_beta_times_epsilon_coord_{j}")),
                    &epsilon_vars[j],
                )?;
                let fused_raw = alpha_prod.add(
                    cs.namespace(|| format!("{label}_fused_raw_coord_{j}")),
                    &beta_prod,
                )?;
                let out = enforce_floor_rescale(
                    &mut cs.namespace(|| format!("{label}_fused_rescale_coord_{j}")),
                    &fused_raw,
                    fused_q[j],
                    fused_r[j],
                    scale,
                    quotient_min,
                    quotient_max,
                    &format!("{label}_fused_rescale_coord_{j}"),
                )?;
                x_next.push(out);
            }
        }
        _ => panic!("denoise update mode and witness variant do not match"),
    }
    Ok(x_next)
}
