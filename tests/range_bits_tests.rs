use denoise::{
    fixed_point::{enforce_signed_range_bits, offset_binary_bits},
    nova_ivc::F1,
};
use nova_snark::frontend::{num::AllocatedNum, test_cs::TestConstraintSystem, ConstraintSystem};

fn assert_range_bits_satisfied(value: i64, min: i64, max: i64) {
    let mut cs = TestConstraintSystem::<F1>::new();
    let value_var = AllocatedNum::alloc(cs.namespace(|| "value"), || {
        Ok(denoise::clipped_relu::field_from_i64::<F1>(value))
    })
    .unwrap();
    enforce_signed_range_bits(&mut cs, &value_var, value, min, max, "range").unwrap();
    assert!(cs.is_satisfied());
}

#[test]
fn range_bits_tests() {
    accepts_zero_in_minus128_127();
}

#[test]
fn accepts_zero_in_minus128_127() {
    assert_range_bits_satisfied(0, -128, 127);
}

#[test]
fn accepts_negative_boundary() {
    assert_range_bits_satisfied(-128, -128, 127);
}

#[test]
fn accepts_positive_boundary() {
    assert_range_bits_satisfied(127, -128, 127);
}

#[test]
#[should_panic(expected = "outside")]
fn rejects_value_below_min() {
    assert_range_bits_satisfied(-129, -128, 127);
}

#[test]
#[should_panic(expected = "outside")]
fn rejects_value_above_max() {
    assert_range_bits_satisfied(128, -128, 127);
}

#[test]
#[should_panic(expected = "power of two")]
fn rejects_non_power_of_two_width() {
    assert_range_bits_satisfied(0, -64, 64);
}

#[test]
fn bit_decomposition_matches_offset_binary() {
    let bits = offset_binary_bits(-5, -128, 127);
    let reconstructed: i64 = bits
        .iter()
        .enumerate()
        .map(|(i, bit)| if *bit { 1i64 << i } else { 0 })
        .sum();
    assert_eq!(reconstructed, 123);
}
