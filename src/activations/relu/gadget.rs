use ff::PrimeField;
use nova_snark::frontend::{num::AllocatedNum, ConstraintSystem, SynthesisError};

use super::table::ReluLookupTable;

pub fn field_from_i64<F: PrimeField>(value: i64) -> F {
    if value >= 0 {
        F::from(value as u64)
    } else {
        -F::from((-value) as u64)
    }
}

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

pub fn relu_lookup<CS, F>(
    cs: &mut CS,
    input: &AllocatedNum<F>,
    input_value: i64,
    output_value: F,
    table: &ReluLookupTable,
    prefix: &str,
) -> Result<AllocatedNum<F>, SynthesisError>
where
    CS: ConstraintSystem<F>,
    F: PrimeField,
{
    assert!(
        table.contains(input_value),
        "input witness not in ReLU table domain"
    );

    let entries = table.entries();
    let mut selectors = Vec::with_capacity(entries.len());

    for (idx, (x, _)) in entries.iter().enumerate() {
        let sel = AllocatedNum::alloc(cs.namespace(|| format!("{prefix}_selector_{idx}")), || {
            Ok(if *x == input_value { F::ONE } else { F::ZERO })
        })?;
        enforce_boolean(cs, &sel, || format!("{prefix}_selector_bool_{idx}"));
        selectors.push(sel);
    }

    // Sum of selectors is exactly 1.
    cs.enforce(
        || format!("{prefix}_one_hot_sum"),
        |lc| {
            selectors
                .iter()
                .fold(lc, |acc, sel| acc + sel.get_variable())
        },
        |lc| lc + CS::one(),
        |lc| lc + CS::one(),
    );

    // Lookup input value: input = sum_j selector_j * table_x_j
    cs.enforce(
        || format!("{prefix}_input_lookup"),
        |lc| {
            entries
                .iter()
                .zip(selectors.iter())
                .fold(lc, |acc, ((x, _), sel)| {
                    acc + (field_from_i64::<F>(*x), sel.get_variable())
                })
        },
        |lc| lc + CS::one(),
        |lc| lc + input.get_variable(),
    );

    let output = AllocatedNum::alloc(cs.namespace(|| format!("{prefix}_output")), || {
        Ok(output_value)
    })?;

    // Lookup output value: output = sum_j selector_j * relu(table_x_j)
    cs.enforce(
        || format!("{prefix}_output_lookup"),
        |lc| {
            entries
                .iter()
                .zip(selectors.iter())
                .fold(lc, |acc, ((_, y), sel)| {
                    acc + (field_from_i64::<F>(*y), sel.get_variable())
                })
        },
        |lc| lc + CS::one(),
        |lc| lc + output.get_variable(),
    );

    Ok(output)
}
