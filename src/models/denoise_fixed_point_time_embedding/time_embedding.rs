use ff::PrimeField;
use nova_snark::frontend::{num::AllocatedNum, ConstraintSystem, SynthesisError};

use crate::{clipped_relu::field_from_i64, fixed_point::encode_f64_round};

fn enforce_boolean<CS, F>(cs: &mut CS, bit: &AllocatedNum<F>, name: impl FnOnce() -> String)
where
    CS: ConstraintSystem<F>,
    F: PrimeField,
{
    cs.enforce(
        name,
        |lc| lc + bit.get_variable(),
        |lc| lc + bit.get_variable() - CS::one(),
        |lc| lc,
    );
}

pub fn generate_simple_time_table<const TE: usize>(
    total_iters: usize,
    scale: i64,
) -> Vec<[i64; TE]> {
    assert!(total_iters > 0, "total_iters must be positive");
    let denom = if total_iters > 1 {
        (total_iters - 1) as f64
    } else {
        1.0
    };

    (0..total_iters)
        .map(|t| {
            let u = (t as f64) / denom;
            let mut row = [0i64; TE];
            if TE >= 1 {
                row[0] = encode_f64_round(u, scale);
            }
            if TE >= 2 {
                row[1] = encode_f64_round(1.0 - u, scale);
            }
            if TE >= 3 {
                row[2] = encode_f64_round(u * u, scale);
            }
            row
        })
        .collect()
}

pub fn pad_time_table_vec<const TE_REAL: usize, const TE_MAX: usize>(
    table: Vec<[i64; TE_REAL]>,
) -> Vec<[i64; TE_MAX]> {
    assert!(TE_REAL <= TE_MAX, "TE_REAL must be <= TE_MAX");
    table
        .into_iter()
        .map(|row| {
            let mut padded = [0i64; TE_MAX];
            padded[..TE_REAL].copy_from_slice(&row);
            padded
        })
        .collect()
}

pub fn synthesize_time_embedding_lookup<CS, F, const TE: usize>(
    cs: &mut CS,
    t_var: &AllocatedNum<F>,
    t_value: i64,
    table_flat_vars: &[AllocatedNum<F>],
    table_values: &[[i64; TE]],
    prefix: &str,
) -> Result<Vec<AllocatedNum<F>>, SynthesisError>
where
    CS: ConstraintSystem<F>,
    F: PrimeField,
{
    let total_iters = table_values.len();
    assert_eq!(
        table_flat_vars.len(),
        total_iters * TE,
        "time table variable length must match table_values"
    );
    assert!(
        0 <= t_value && (t_value as usize) < total_iters,
        "timestep {t_value} is outside time table range"
    );

    let mut selectors = Vec::with_capacity(total_iters);
    for k in 0..total_iters {
        let sel = AllocatedNum::alloc(cs.namespace(|| format!("{prefix}_selector_{k}")), || {
            Ok(if k == t_value as usize {
                F::ONE
            } else {
                F::ZERO
            })
        })?;
        enforce_boolean(cs, &sel, || format!("{prefix}_selector_bool_{k}"));
        selectors.push(sel);
    }

    cs.enforce(
        || format!("{prefix}_selector_one_hot"),
        |lc| {
            selectors
                .iter()
                .fold(lc, |acc, sel| acc + sel.get_variable())
        },
        |lc| lc + CS::one(),
        |lc| lc + CS::one(),
    );

    cs.enforce(
        || format!("{prefix}_selected_timestep"),
        |lc| {
            selectors.iter().enumerate().fold(lc, |acc, (k, sel)| {
                acc + (F::from(k as u64), sel.get_variable())
            })
        },
        |lc| lc + CS::one(),
        |lc| lc + t_var.get_variable(),
    );

    let mut embedding = Vec::with_capacity(TE);
    for j in 0..TE {
        let mut products = Vec::with_capacity(total_iters);
        for k in 0..total_iters {
            let table_var = &table_flat_vars[k * TE + j];
            let product = table_var.mul(
                cs.namespace(|| format!("{prefix}_table_times_selector_row_{k}_coord_{j}")),
                &selectors[k],
            )?;
            products.push(product);
        }

        let emb_value = table_values[t_value as usize][j];
        let emb = AllocatedNum::alloc(
            cs.namespace(|| format!("{prefix}_embedding_coord_{j}")),
            || Ok(field_from_i64::<F>(emb_value)),
        )?;

        cs.enforce(
            || format!("{prefix}_embedding_lookup_coord_{j}"),
            |lc| {
                products
                    .iter()
                    .fold(lc, |acc, product| acc + product.get_variable())
            },
            |lc| lc + CS::one(),
            |lc| lc + emb.get_variable(),
        );
        embedding.push(emb);
    }

    Ok(embedding)
}
