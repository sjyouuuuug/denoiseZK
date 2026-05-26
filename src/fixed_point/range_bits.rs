use ff::PrimeField;
use nova_snark::frontend::{num::AllocatedNum, ConstraintSystem, SynthesisError};

use crate::clipped_relu::field_from_i64;

pub fn ceil_log2_power_of_two(width: i64) -> usize {
    assert!(width > 0, "range width must be positive");
    assert!(
        (width & (width - 1)) == 0,
        "range width must be a power of two"
    );
    width.trailing_zeros() as usize
}

pub fn offset_binary_bits(value: i64, min: i64, max: i64) -> Vec<bool> {
    assert!(min <= max, "invalid signed range");
    let width = max
        .checked_sub(min)
        .and_then(|d| d.checked_add(1))
        .expect("signed range width overflow");
    let k = ceil_log2_power_of_two(width);
    assert!(
        min <= value && value <= max,
        "signed range witness {value} is outside [{min}, {max}]"
    );
    let u = value
        .checked_sub(min)
        .expect("offset binary witness overflow");
    (0..k).map(|i| ((u >> i) & 1) == 1).collect()
}

/// Prove `value` is in `[min, max]` using offset-binary bit decomposition.
///
/// This gadget intentionally supports only ranges whose width is a power of two.
/// For example, use `[-128, 127]` rather than `[-128, 128]`.
pub fn enforce_signed_range_bits<CS, F>(
    cs: &mut CS,
    value: &AllocatedNum<F>,
    value_int: i64,
    min: i64,
    max: i64,
    label: &str,
) -> Result<(), SynthesisError>
where
    CS: ConstraintSystem<F>,
    F: PrimeField,
{
    assert!(min <= max, "invalid signed range");
    let width = max
        .checked_sub(min)
        .and_then(|d| d.checked_add(1))
        .expect("signed range width overflow");
    let k = ceil_log2_power_of_two(width);
    let bits = offset_binary_bits(value_int, min, max);
    assert_eq!(bits.len(), k);

    let mut bit_vars = Vec::with_capacity(k);
    for (i, bit) in bits.into_iter().enumerate() {
        let bit_var = AllocatedNum::alloc(cs.namespace(|| format!("{label}_bit_{i}")), || {
            Ok(if bit { F::ONE } else { F::ZERO })
        })?;
        cs.enforce(
            || format!("{label}_bit_{i}_boolean"),
            |lc| lc + bit_var.get_variable(),
            |lc| lc + bit_var.get_variable() - CS::one(),
            |lc| lc,
        );
        bit_vars.push(bit_var);
    }

    let offset = field_from_i64::<F>(-min);
    cs.enforce(
        || format!("{label}_offset_binary_sum"),
        |lc| {
            let mut acc = lc + value.get_variable() + (offset, CS::one());
            for (i, bit) in bit_vars.iter().enumerate() {
                acc = acc - (F::from(1u64 << i), bit.get_variable());
            }
            acc
        },
        |lc| lc + CS::one(),
        |lc| lc,
    );

    Ok(())
}
