use denoise::{
    commitment::{
        field_from_i64, toy_hash_block_prefixes_i64_from_field, toy_hash_i64_as_field,
        TOY_HASH_BASE_U64,
    },
    nova_ivc::F1,
    public_state::CommittedDenoiseStateLayout,
};
use ff::Field;
use nova_snark::frontend::{num::AllocatedNum, test_cs::TestConstraintSystem, ConstraintSystem};

#[test]
fn commitment_accumulator_tests() {
    toy_hash_block_matches_full_hash();
}

#[test]
fn toy_hash_block_matches_full_hash() {
    let block1 = [1, -2, 3];
    let block2 = [4, 5];
    let (_p1, h1) =
        toy_hash_block_prefixes_i64_from_field::<F1>(&block1, TOY_HASH_BASE_U64, F1::ZERO);
    let (_p2, h2) = toy_hash_block_prefixes_i64_from_field::<F1>(&block2, TOY_HASH_BASE_U64, h1);
    let flat = [1, -2, 3, 4, 5];
    assert_eq!(h2, toy_hash_i64_as_field::<F1>(&flat, TOY_HASH_BASE_U64, 0));
}

#[test]
fn committed_layout_ranges_are_correct() {
    let layout = CommittedDenoiseStateLayout::new(4, true, 3, 10, 2);
    assert_eq!(layout.x_range(), 0..4);
    assert_eq!(layout.y_range(), Some(4..8));
    assert_eq!(layout.h_index(), 8);
    assert_eq!(layout.c_index(), 9);
    assert_eq!(layout.t_index(), 10);
    assert_eq!(layout.params_base(), 11);
    assert_eq!(layout.param_block_range(1), 21..31);
    assert_eq!(layout.time_table_range(), 41..47);
    assert_eq!(layout.state_len(), 47);
}

#[test]
fn committed_layout_without_output_ranges_are_correct() {
    let layout = CommittedDenoiseStateLayout::new(4, false, 3, 10, 2);
    assert_eq!(layout.x_range(), 0..4);
    assert_eq!(layout.y_range(), None);
    assert_eq!(layout.h_index(), 4);
    assert_eq!(layout.c_index(), 5);
    assert_eq!(layout.t_index(), 6);
    assert_eq!(layout.params_base(), 7);
    assert_eq!(layout.state_len(), 43);
}

#[test]
fn final_check_flag_witness_values_are_unambiguous() {
    let target = 4;
    let flags: Vec<_> = (1..=4).map(|t_next| i64::from(t_next == target)).collect();
    assert_eq!(flags, vec![0, 0, 0, 1]);
    assert_eq!(field_from_i64::<F1>(flags[3]), F1::ONE);
}

#[test]
fn is_equal_to_constant_gadget_handles_equal_case() {
    let mut cs = TestConstraintSystem::<F1>::new();
    let value = AllocatedNum::alloc(cs.namespace(|| "value"), || Ok(field_from_i64(4))).unwrap();
    let flag = denoise::public_state::synthesize_is_equal_to_constant(
        &mut cs.namespace(|| "eq"),
        &value,
        4,
        field_from_i64(4),
        4,
        "eq",
    )
    .unwrap();
    assert_eq!(flag.get_value().unwrap(), F1::ONE);
    assert!(cs.is_satisfied());
}

#[test]
fn is_equal_to_constant_gadget_handles_not_equal_case() {
    let mut cs = TestConstraintSystem::<F1>::new();
    let value = AllocatedNum::alloc(cs.namespace(|| "value"), || Ok(field_from_i64(3))).unwrap();
    let flag = denoise::public_state::synthesize_is_equal_to_constant(
        &mut cs.namespace(|| "neq"),
        &value,
        3,
        field_from_i64(4),
        4,
        "neq",
    )
    .unwrap();
    assert_eq!(flag.get_value().unwrap(), F1::ZERO);
    assert!(cs.is_satisfied());
}

#[test]
fn conditional_equality_enforces_when_flag_one() {
    let mut cs = TestConstraintSystem::<F1>::new();
    let flag = AllocatedNum::alloc(cs.namespace(|| "flag"), || Ok(F1::ONE)).unwrap();
    let lhs = AllocatedNum::alloc(cs.namespace(|| "lhs"), || Ok(field_from_i64(7))).unwrap();
    let rhs = AllocatedNum::alloc(cs.namespace(|| "rhs"), || Ok(field_from_i64(7))).unwrap();
    denoise::public_state::enforce_equal_if(
        &mut cs.namespace(|| "eq_if"),
        &flag,
        &lhs,
        &rhs,
        "eq_if",
    )
    .unwrap();
    assert!(cs.is_satisfied());
}
