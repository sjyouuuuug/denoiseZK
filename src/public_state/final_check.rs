use ff::PrimeField;
use nova_snark::frontend::{num::AllocatedNum, ConstraintSystem, SynthesisError};

pub fn synthesize_is_equal_to_constant<CS, F>(
    cs: &mut CS,
    value: &AllocatedNum<F>,
    value_int: i64,
    target: F,
    target_int: i64,
    label: &str,
) -> Result<AllocatedNum<F>, SynthesisError>
where
    CS: ConstraintSystem<F>,
    F: PrimeField,
{
    let is_equal_value = if value_int == target_int {
        F::ONE
    } else {
        F::ZERO
    };
    let flag = AllocatedNum::alloc(cs.namespace(|| format!("{label}_is_equal")), || {
        Ok(is_equal_value)
    })?;

    let a_value = crate::clipped_relu::field_from_i64::<F>(value_int) - target;
    let inv_value = if value_int == target_int {
        F::ZERO
    } else {
        Option::<F>::from(a_value.invert()).ok_or(SynthesisError::DivisionByZero)?
    };
    let inv = AllocatedNum::alloc(cs.namespace(|| format!("{label}_inverse")), || {
        Ok(inv_value)
    })?;

    cs.enforce(
        || format!("{label}_flag_boolean"),
        |lc| lc + flag.get_variable(),
        |lc| lc + flag.get_variable() - CS::one(),
        |lc| lc,
    );
    cs.enforce(
        || format!("{label}_a_times_flag_zero"),
        |lc| lc + value.get_variable() - (target, CS::one()),
        |lc| lc + flag.get_variable(),
        |lc| lc,
    );
    cs.enforce(
        || format!("{label}_a_times_inv"),
        |lc| lc + value.get_variable() - (target, CS::one()),
        |lc| lc + inv.get_variable(),
        |lc| lc + CS::one() - flag.get_variable(),
    );

    Ok(flag)
}

pub fn enforce_equal_if<CS, F>(
    cs: &mut CS,
    flag: &AllocatedNum<F>,
    lhs: &AllocatedNum<F>,
    rhs: &AllocatedNum<F>,
    label: &str,
) -> Result<(), SynthesisError>
where
    CS: ConstraintSystem<F>,
    F: PrimeField,
{
    cs.enforce(
        || label.to_string(),
        |lc| lc + flag.get_variable(),
        |lc| lc + lhs.get_variable() - rhs.get_variable(),
        |lc| lc,
    );
    Ok(())
}
