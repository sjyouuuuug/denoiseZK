use ff::PrimeField;
use nova_snark::frontend::{num::AllocatedNum, ConstraintSystem, SynthesisError};

pub fn synthesize_toy_hash_update<CS, F>(
    cs: &mut CS,
    h_prev: &AllocatedNum<F>,
    value: &AllocatedNum<F>,
    base: F,
    h_next_value: F,
    label: &str,
) -> Result<AllocatedNum<F>, SynthesisError>
where
    CS: ConstraintSystem<F>,
    F: PrimeField,
{
    let h_next = AllocatedNum::alloc(cs.namespace(|| format!("{label}_h_next")), || {
        Ok(h_next_value)
    })?;
    cs.enforce(
        || format!("{label}_hash_update"),
        |lc| lc + h_prev.get_variable(),
        |lc| lc + (base, CS::one()),
        |lc| lc + h_next.get_variable() - value.get_variable(),
    );
    Ok(h_next)
}

pub fn synthesize_toy_hash_sequence_return<CS, F>(
    cs: &mut CS,
    values: &[AllocatedNum<F>],
    base: F,
    init: F,
    h_values: &[F],
    label: &str,
) -> Result<AllocatedNum<F>, SynthesisError>
where
    CS: ConstraintSystem<F>,
    F: PrimeField,
{
    assert_eq!(
        values.len(),
        h_values.len(),
        "hash witness length must match values length"
    );
    let mut h = AllocatedNum::alloc(cs.namespace(|| format!("{label}_h_init")), || Ok(init))?;
    cs.enforce(
        || format!("{label}_h_init_check"),
        |lc| lc + h.get_variable(),
        |lc| lc + CS::one(),
        |lc| lc + (init, CS::one()),
    );
    for (i, value) in values.iter().enumerate() {
        h = synthesize_toy_hash_update(
            &mut cs.namespace(|| format!("{label}_step_{i}")),
            &h,
            value,
            base,
            h_values[i],
            &format!("{label}_step_{i}"),
        )?;
    }
    Ok(h)
}

pub fn synthesize_toy_hash_block<CS, F>(
    cs: &mut CS,
    h_start: &AllocatedNum<F>,
    values: &[AllocatedNum<F>],
    base: F,
    h_values: &[F],
    label: &str,
) -> Result<AllocatedNum<F>, SynthesisError>
where
    CS: ConstraintSystem<F>,
    F: PrimeField,
{
    assert_eq!(
        values.len(),
        h_values.len(),
        "hash block witness length must match values length"
    );
    let mut h = h_start.clone();
    for (i, value) in values.iter().enumerate() {
        h = synthesize_toy_hash_update(
            &mut cs.namespace(|| format!("{label}_step_{i}")),
            &h,
            value,
            base,
            h_values[i],
            &format!("{label}_step_{i}"),
        )?;
    }
    Ok(h)
}

pub fn synthesize_toy_hash_sequence<CS, F>(
    cs: &mut CS,
    values: &[AllocatedNum<F>],
    base: F,
    init: F,
    expected_commitment: &AllocatedNum<F>,
    h_values: &[F],
    label: &str,
) -> Result<(), SynthesisError>
where
    CS: ConstraintSystem<F>,
    F: PrimeField,
{
    let h_final = synthesize_toy_hash_sequence_return(cs, values, base, init, h_values, label)?;
    cs.enforce(
        || format!("{label}_commitment_eq"),
        |lc| lc + h_final.get_variable(),
        |lc| lc + CS::one(),
        |lc| lc + expected_commitment.get_variable(),
    );
    Ok(())
}
