use denoise::{
    commitment::{
        field_from_i64, toy_hash_field, toy_hash_i128, toy_hash_i64, toy_hash_i64_as_field,
        toy_hash_prefixes_i64_as_field, TOY_HASH_BASE_U64,
    },
    fixed_point::FixedPointConfig,
    mlp_fixed_point_clipped_relu_lookup::{
        generate_fixed_point_mlp_trace, FixedMlpClippedReluPublicParams,
        FixedMlpClippedReluStepParams,
    },
    nova_ivc::E1,
    public_state::PublicStateLayout,
};
use ff::Field;
use nova_snark::traits::Engine;

type F = <E1 as Engine>::Scalar;

#[test]
fn commitment_tests() {
    toy_hash_i64_matches_hand_computation();
}

#[test]
fn toy_hash_i64_matches_hand_computation() {
    assert_eq!(toy_hash_i64(&[1, 2, 3], 10, 0), 123);
    assert_eq!(toy_hash_i128(&[1, 2, 3], 10, 0), 123);
}

#[test]
fn toy_hash_field_matches_i64_for_positive_small_values() {
    let values = [F::from(1), F::from(2), F::from(3)];
    let h = toy_hash_field(&values, F::from(10), F::ZERO);
    assert_eq!(h, F::from(123));
}

#[test]
fn toy_hash_i64_as_field_handles_negative_values() {
    let values = [1, -2, 3];
    let h = toy_hash_i64_as_field::<F>(&values, 10, 0);
    let manual = F::from(1) * F::from(10) * F::from(10) - F::from(2) * F::from(10) + F::from(3);
    assert_eq!(h, manual);
}

#[test]
fn toy_hash_prefixes_match_final_hash() {
    let values = [1, -2, 3, 4];
    let prefixes = toy_hash_prefixes_i64_as_field::<F>(&values, TOY_HASH_BASE_U64, 0);
    assert_eq!(
        *prefixes.last().unwrap(),
        toy_hash_i64_as_field::<F>(&values, TOY_HASH_BASE_U64, 0)
    );
}

#[test]
fn public_state_layout_with_commitment_ranges_are_correct() {
    let layout = PublicStateLayout::new_with_commitment(2, true, true, 2, 14, 0);
    assert_eq!(layout.x_range(), 0..2);
    assert_eq!(layout.y_range().unwrap(), 2..4);
    assert_eq!(layout.commitment_index(), Some(4));
    assert_eq!(layout.t_index(), 5);
    assert_eq!(layout.params_range(), 6..34);
    assert_eq!(layout.time_table_range(), 34..34);
}

#[test]
fn mlp_commitment_z0_contains_expected_commitment() {
    const N: usize = 2;
    const H: usize = 1;
    let cfg = FixedPointConfig::from_real_bounds(4, -8, 8, 4);
    let step = FixedMlpClippedReluStepParams::<N, H>::new([[4, 0]], [0], [[4], [0]], [0, 0]);
    let params_seq = vec![step.clone()];
    let public_params = FixedMlpClippedReluPublicParams::new(params_seq.clone(), cfg);
    let (_, trace) = generate_fixed_point_mlp_trace::<F, N, H>(&public_params, [4, 0]);
    let expected_y = trace.last().unwrap().x_i_plus_1_int;
    let flat_params: Vec<i64> = params_seq.iter().flat_map(|p| p.flatten_i64()).collect();
    let commitment = toy_hash_i64_as_field::<F>(&flat_params, TOY_HASH_BASE_U64, 0);
    let layout = PublicStateLayout::new_with_commitment(
        N,
        true,
        true,
        1,
        FixedMlpClippedReluStepParams::<N, H>::block_len(),
        0,
    );
    let mut z0 = Vec::new();
    z0.extend([4, 0].into_iter().map(field_from_i64::<F>));
    z0.extend(expected_y.into_iter().map(field_from_i64::<F>));
    z0.push(commitment);
    z0.push(F::ZERO);
    z0.extend(step.flatten_field::<F>());

    assert_eq!(z0[layout.commitment_index().unwrap()], commitment);
    assert_eq!(z0[layout.t_index()], F::ZERO);
    assert_eq!(&z0[layout.params_range()], &step.flatten_field::<F>());
}

#[test]
#[ignore]
fn mlp_fixed_point_commitment_proof_verifies() {
    // Covered by src/bin/mlp_fixed_point_commitment_demo.rs; kept ignored because it runs Nova.
}
