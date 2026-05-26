use ff::PrimeField;
use nova_snark::frontend::{num::AllocatedNum, ConstraintSystem, SynthesisError};
use std::sync::atomic::{AtomicU8, Ordering};

use crate::clipped_relu::field_from_i64;

use super::range_bits::enforce_signed_range_bits;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SignedRangeCheckMode {
    OneHot,
    Bits,
}

static SIGNED_RANGE_CHECK_MODE: AtomicU8 = AtomicU8::new(1);

pub fn set_signed_range_check_mode(mode: SignedRangeCheckMode) {
    let value = match mode {
        SignedRangeCheckMode::OneHot => 0,
        SignedRangeCheckMode::Bits => 1,
    };
    SIGNED_RANGE_CHECK_MODE.store(value, Ordering::SeqCst);
}

pub fn signed_range_check_mode() -> SignedRangeCheckMode {
    match SIGNED_RANGE_CHECK_MODE.load(Ordering::SeqCst) {
        0 => SignedRangeCheckMode::OneHot,
        _ => SignedRangeCheckMode::Bits,
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

/// Prove remainder is in {0, ..., scale-1} by one-hot lookup.
pub fn enforce_remainder_range<CS, F>(
    cs: &mut CS,
    remainder: &AllocatedNum<F>,
    remainder_value: i64,
    scale: i64,
    prefix: &str,
) -> Result<(), SynthesisError>
where
    CS: ConstraintSystem<F>,
    F: PrimeField,
{
    assert!(scale > 0, "scale must be positive");
    assert!(
        0 <= remainder_value && remainder_value < scale,
        "remainder witness {remainder_value} is outside [0, {})",
        scale
    );

    let mut selectors = Vec::with_capacity(scale as usize);
    for v in 0..scale {
        let sel = AllocatedNum::alloc(
            cs.namespace(|| format!("{prefix}_rem_selector_{v}")),
            || {
                Ok(if v == remainder_value {
                    F::ONE
                } else {
                    F::ZERO
                })
            },
        )?;
        enforce_boolean(cs, &sel, || format!("{prefix}_rem_selector_bool_{v}"));
        selectors.push(sel);
    }

    cs.enforce(
        || format!("{prefix}_rem_one_hot_sum"),
        |lc| {
            selectors
                .iter()
                .fold(lc, |acc, sel| acc + sel.get_variable())
        },
        |lc| lc + CS::one(),
        |lc| lc + CS::one(),
    );

    cs.enforce(
        || format!("{prefix}_rem_lookup_value"),
        |lc| {
            selectors.iter().enumerate().fold(lc, |acc, (v, sel)| {
                acc + (F::from(v as u64), sel.get_variable())
            })
        },
        |lc| lc + CS::one(),
        |lc| lc + remainder.get_variable(),
    );

    Ok(())
}

/// Prove value is in {min, ..., max} by one-hot lookup over signed integers.
pub fn enforce_signed_range<CS, F>(
    cs: &mut CS,
    value: &AllocatedNum<F>,
    value_witness: i64,
    min: i64,
    max: i64,
    prefix: &str,
) -> Result<(), SynthesisError>
where
    CS: ConstraintSystem<F>,
    F: PrimeField,
{
    assert!(min <= max, "invalid signed range");
    assert!(
        min <= value_witness && value_witness <= max,
        "signed range witness {value_witness} is outside [{min}, {max}]"
    );

    let range_len = (max - min + 1) as usize;
    let mut selectors = Vec::with_capacity(range_len);
    for (idx, v) in (min..=max).enumerate() {
        let sel = AllocatedNum::alloc(
            cs.namespace(|| format!("{prefix}_signed_selector_{idx}")),
            || Ok(if v == value_witness { F::ONE } else { F::ZERO }),
        )?;
        enforce_boolean(cs, &sel, || format!("{prefix}_signed_selector_bool_{idx}"));
        selectors.push((v, sel));
    }

    cs.enforce(
        || format!("{prefix}_signed_one_hot_sum"),
        |lc| {
            selectors
                .iter()
                .fold(lc, |acc, (_, sel)| acc + sel.get_variable())
        },
        |lc| lc + CS::one(),
        |lc| lc + CS::one(),
    );

    cs.enforce(
        || format!("{prefix}_signed_lookup_value"),
        |lc| {
            selectors.iter().fold(lc, |acc, (v, sel)| {
                acc + (field_from_i64::<F>(*v), sel.get_variable())
            })
        },
        |lc| lc + CS::one(),
        |lc| lc + value.get_variable(),
    );

    Ok(())
}

/// Enforce numerator = quotient * scale + remainder and 0 <= remainder < scale.
/// With the quotient signed range, this encodes quotient = floor(numerator / scale)
/// for positive scale inside the configured integer domain.
pub fn enforce_floor_rescale<CS, F>(
    cs: &mut CS,
    numerator: &AllocatedNum<F>,
    quotient_value: i64,
    remainder_value: i64,
    scale: i64,
    quotient_min: i64,
    quotient_max: i64,
    prefix: &str,
) -> Result<AllocatedNum<F>, SynthesisError>
where
    CS: ConstraintSystem<F>,
    F: PrimeField,
{
    assert!(scale > 0, "scale must be positive");
    let quotient = AllocatedNum::alloc(cs.namespace(|| format!("{prefix}_quotient")), || {
        Ok(field_from_i64::<F>(quotient_value))
    })?;
    let remainder = AllocatedNum::alloc(cs.namespace(|| format!("{prefix}_remainder")), || {
        Ok(field_from_i64::<F>(remainder_value))
    })?;

    match signed_range_check_mode() {
        SignedRangeCheckMode::OneHot => enforce_signed_range(
            cs,
            &quotient,
            quotient_value,
            quotient_min,
            quotient_max,
            &format!("{prefix}_quotient_range"),
        )?,
        SignedRangeCheckMode::Bits => enforce_signed_range_bits(
            cs,
            &quotient,
            quotient_value,
            quotient_min,
            quotient_max,
            &format!("{prefix}_quotient_range"),
        )?,
    }
    enforce_remainder_range(cs, &remainder, remainder_value, scale, prefix)?;

    cs.enforce(
        || format!("{prefix}_floor_rescale_relation"),
        |lc| lc + (field_from_i64::<F>(scale), quotient.get_variable()) + remainder.get_variable(),
        |lc| lc + CS::one(),
        |lc| lc + numerator.get_variable(),
    );

    Ok(quotient)
}
