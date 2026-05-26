use ff::PrimeField;
use nova_snark::frontend::{num::AllocatedNum, ConstraintSystem, SynthesisError};

pub fn enforce_zero<CS, F>(
    cs: &mut CS,
    value: &AllocatedNum<F>,
    label: impl FnOnce() -> String,
) -> Result<(), SynthesisError>
where
    CS: ConstraintSystem<F>,
    F: PrimeField,
{
    cs.enforce(
        label,
        |lc| lc + value.get_variable(),
        |lc| lc + CS::one(),
        |lc| lc,
    );
    Ok(())
}

pub fn enforce_zero_padding_vector<CS, F, const REAL: usize, const MAX: usize>(
    cs: &mut CS,
    values: &[AllocatedNum<F>],
    prefix: &str,
) -> Result<(), SynthesisError>
where
    CS: ConstraintSystem<F>,
    F: PrimeField,
{
    assert_eq!(values.len(), MAX, "values length must equal MAX");
    assert!(REAL <= MAX, "REAL must be <= MAX");
    for (i, value) in values.iter().enumerate().skip(REAL) {
        enforce_zero(cs, value, || format!("{prefix}_pad_zero_{i}"))?;
    }
    Ok(())
}

pub fn enforce_zero_padding_matrix_flat<CS, F>(
    cs: &mut CS,
    flat_values: &[AllocatedNum<F>],
    rows_real: usize,
    cols_real: usize,
    rows_max: usize,
    cols_max: usize,
    prefix: &str,
) -> Result<(), SynthesisError>
where
    CS: ConstraintSystem<F>,
    F: PrimeField,
{
    assert_eq!(
        flat_values.len(),
        rows_max * cols_max,
        "flat_values length must equal rows_max * cols_max"
    );
    assert!(rows_real <= rows_max, "rows_real must be <= rows_max");
    assert!(cols_real <= cols_max, "cols_real must be <= cols_max");
    for r in 0..rows_max {
        for c in 0..cols_max {
            if r >= rows_real || c >= cols_real {
                enforce_zero(cs, &flat_values[r * cols_max + c], || {
                    format!("{prefix}_pad_zero_row_{r}_col_{c}")
                })?;
            }
        }
    }
    Ok(())
}
