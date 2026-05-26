use denoise::models::denoise_update::{
    compute_denoise_update_witness, DenoiseUpdateMode, DenoiseUpdateWitness,
};

#[test]
fn denoise_update_tests() {
    double_floor_matches_hand_computation();
}

#[test]
fn double_floor_matches_hand_computation() {
    let witness = compute_denoise_update_witness(
        &[16, -8],
        &[4, 12],
        14,
        2,
        16,
        DenoiseUpdateMode::DoubleFloor,
    );
    match witness {
        DenoiseUpdateWitness::DoubleFloor {
            alpha_raw,
            alpha_q,
            alpha_r,
            beta_raw,
            beta_q,
            beta_r,
            x_next,
        } => {
            assert_eq!(alpha_raw, [224, -112]);
            assert_eq!(alpha_q, [14, -7]);
            assert_eq!(alpha_r, [0, 0]);
            assert_eq!(beta_raw, [8, 24]);
            assert_eq!(beta_q, [0, 1]);
            assert_eq!(beta_r, [8, 8]);
            assert_eq!(x_next, [14, -6]);
        }
        _ => panic!("expected double-floor witness"),
    }
}

#[test]
fn fused_floor_matches_hand_computation() {
    let witness = compute_denoise_update_witness(
        &[16, -8],
        &[4, 12],
        14,
        2,
        16,
        DenoiseUpdateMode::FusedFloor,
    );
    match witness {
        DenoiseUpdateWitness::FusedFloor {
            fused_raw,
            fused_q,
            fused_r,
            x_next,
        } => {
            assert_eq!(fused_raw, [232, -88]);
            assert_eq!(fused_q, [14, -6]);
            assert_eq!(fused_r, [8, 8]);
            assert_eq!(x_next, [14, -6]);
        }
        _ => panic!("expected fused-floor witness"),
    }
}

#[test]
fn double_and_fused_can_differ() {
    let x = [1];
    let epsilon = [1];
    let double =
        compute_denoise_update_witness(&x, &epsilon, 7, 7, 10, DenoiseUpdateMode::DoubleFloor);
    let fused =
        compute_denoise_update_witness(&x, &epsilon, 7, 7, 10, DenoiseUpdateMode::FusedFloor);
    assert_eq!(*double.x_next(), [0]);
    assert_eq!(*fused.x_next(), [1]);
}
