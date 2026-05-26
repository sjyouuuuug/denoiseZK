use denoise::{
    clipped_relu::{field_from_i64, ClippedReluLookupTable},
    mlp_clipped_relu_lookup::{
        build_mlp_clipped_relu_placeholder_circuit, build_mlp_clipped_relu_step_circuits,
        generate_mlp_clipped_relu_trace, run_mlp_clipped_relu_recursive,
        setup_mlp_clipped_relu_public_params, verify_mlp_clipped_relu_recursive,
        IntMlpClippedReluStepParams, MlpClippedReluPublicParams,
    },
    nova_ivc::E1,
};
use nova_snark::traits::Engine;

type F = <E1 as Engine>::Scalar;

#[test]
fn clipped_relu_table_has_expected_semantics() {
    let table = ClippedReluLookupTable::new(-4, 6, 3);

    assert_eq!(table.size(), 11);
    assert!(table.contains(-4));
    assert!(table.contains(6));
    assert!(!table.contains(-5));
    assert!(!table.contains(7));

    assert_eq!(table.clipped_relu(-4), 0);
    assert_eq!(table.clipped_relu(-1), 0);
    assert_eq!(table.clipped_relu(0), 0);
    assert_eq!(table.clipped_relu(1), 1);
    assert_eq!(table.clipped_relu(3), 3);
    assert_eq!(table.clipped_relu(4), 3);
    assert_eq!(table.clipped_relu(6), 3);

    let entries = table.entries();
    assert_eq!(entries.first(), Some(&(-4, 0)));
    assert_eq!(entries.last(), Some(&(6, 3)));
}

#[test]
#[should_panic(expected = "outside clipped ReLU table range")]
fn clipped_relu_panics_outside_table_domain() {
    let table = ClippedReluLookupTable::new(-2, 2, 1);
    let _ = table.clipped_relu(3);
}

#[test]
fn mlp_step_param_block_len_and_flatten_layout_are_correct() {
    const N: usize = 2;
    const H: usize = 3;

    let params = IntMlpClippedReluStepParams::<N, H>::new(
        [[1, 2], [3, 4], [5, 6]],
        [7, 8, 9],
        [[10, 11, 12], [13, 14, 15]],
        [16, 17],
    );

    assert_eq!(
        IntMlpClippedReluStepParams::<N, H>::block_len(),
        H * N + H + N * H + N
    );
    assert_eq!(
        params.flatten_i64(),
        vec![
            // W1 row-major, H x N
            1, 2, 3, 4, 5, 6, // b1
            7, 8, 9, // W2 row-major, N x H
            10, 11, 12, 13, 14, 15, // b2
            16, 17,
        ]
    );
}

#[test]
fn generate_trace_matches_hand_computed_two_layer_mlp() {
    const N: usize = 2;
    const H: usize = 2;

    let table = ClippedReluLookupTable::new(-16, 16, 4);
    let step = IntMlpClippedReluStepParams::<N, H>::new(
        // hidden_affine = W1*x + b1
        [[1, -1], [2, 1]],
        [0, -3],
        // x_next = W2*h + b2
        [[1, 2], [-1, 1]],
        [1, 0],
    );
    let public_params = MlpClippedReluPublicParams::new(vec![step], table.clone());
    let x0 = [2, 3];

    let (z0, trace) = generate_mlp_clipped_relu_trace::<F, N, H>(&public_params, x0);

    assert_eq!(
        z0.len(),
        N + IntMlpClippedReluStepParams::<N, H>::block_len()
    );
    assert_eq!(z0[0], field_from_i64::<F>(2));
    assert_eq!(z0[1], field_from_i64::<F>(3));

    assert_eq!(trace.len(), 1);
    let it = &trace[0];

    // hidden_affine = [1*2 + (-1)*3 + 0, 2*2 + 1*3 - 3] = [-1, 4]
    assert_eq!(it.hidden_affine_int, [-1, 4]);
    // clipped_relu([-1, 4]) with clip_max=4 is [0, 4]
    assert_eq!(it.hidden_act_int, [0, 4]);
    // x_next = [1*0 + 2*4 + 1, -1*0 + 1*4 + 0] = [9, 4]
    assert_eq!(it.x_i_plus_1_int, [9, 4]);

    assert_eq!(
        it.hidden_affine,
        [field_from_i64::<F>(-1), field_from_i64::<F>(4)]
    );
    assert_eq!(
        it.hidden_act,
        [field_from_i64::<F>(0), field_from_i64::<F>(4)]
    );
    assert_eq!(
        it.x_i_plus_1,
        [field_from_i64::<F>(9), field_from_i64::<F>(4)]
    );
}

#[test]
#[should_panic(expected = "outside clipped ReLU table range")]
fn generate_trace_rejects_hidden_affine_outside_lookup_range() {
    const N: usize = 2;
    const H: usize = 1;

    let table = ClippedReluLookupTable::new(-4, 4, 2);
    let step = IntMlpClippedReluStepParams::<N, H>::new([[10, 0]], [0], [[1], [0]], [0, 0]);
    let public_params = MlpClippedReluPublicParams::new(vec![step], table);

    // hidden_affine = 10, outside [-4, 4].
    let _ = generate_mlp_clipped_relu_trace::<F, N, H>(&public_params, [1, 0]);
}

#[test]
fn build_step_circuits_chunks_trace_and_carries_public_metadata() {
    const N: usize = 2;
    const H: usize = 2;

    let table = ClippedReluLookupTable::new(-16, 16, 4);
    let params_seq = vec![
        IntMlpClippedReluStepParams::<N, H>::new(
            [[1, 0], [0, 1]],
            [0, 0],
            [[1, 0], [0, 1]],
            [0, 0],
        ),
        IntMlpClippedReluStepParams::<N, H>::new(
            [[1, 1], [1, 0]],
            [0, 1],
            [[1, 1], [0, 1]],
            [1, 0],
        ),
        IntMlpClippedReluStepParams::<N, H>::new(
            [[0, 1], [1, 1]],
            [1, 0],
            [[1, 0], [1, 1]],
            [0, 1],
        ),
        IntMlpClippedReluStepParams::<N, H>::new(
            [[1, -1], [0, 1]],
            [0, 0],
            [[1, 0], [0, 1]],
            [0, 0],
        ),
    ];
    let public_params = MlpClippedReluPublicParams::new(params_seq, table.clone());
    let (_z0, trace) = generate_mlp_clipped_relu_trace::<F, N, H>(&public_params, [1, 1]);

    let circuits = build_mlp_clipped_relu_step_circuits(&trace, 2, 2, 4, table.clone());
    assert_eq!(circuits.len(), 2);
    assert_eq!(circuits[0].seq.len(), 2);
    assert_eq!(circuits[1].seq.len(), 2);
    assert_eq!(circuits[0].total_iters, 4);
    assert_eq!(circuits[0].num_iters_per_step, 2);
    assert_eq!(circuits[0].clipped_relu_table.min, table.min);
    assert_eq!(circuits[0].clipped_relu_table.max, table.max);
    assert_eq!(circuits[0].clipped_relu_table.clip_max, table.clip_max);
}

// This is a heavier end-to-end test. Run it explicitly with:
//   cargo test --release small_recursive_mlp_clipped_relu_proof_verifies -- --ignored --nocapture
#[test]
#[ignore]
fn small_recursive_mlp_clipped_relu_proof_verifies() {
    const N: usize = 2;
    const H: usize = 2;

    let num_steps = 1;
    let num_iters_per_step = 1;
    let total_iters = num_steps * num_iters_per_step;
    let table = ClippedReluLookupTable::new(-8, 8, 3);

    let params_seq = vec![IntMlpClippedReluStepParams::<N, H>::new(
        [[1, -1], [1, 0]],
        [0, 0],
        [[1, 1], [0, 1]],
        [0, 0],
    )];
    let public_params = MlpClippedReluPublicParams::new(params_seq, table.clone());
    let (z0, trace) = generate_mlp_clipped_relu_trace::<F, N, H>(&public_params, [1, 1]);

    let placeholder =
        build_mlp_clipped_relu_placeholder_circuit(total_iters, num_iters_per_step, table.clone());
    let pp = setup_mlp_clipped_relu_public_params(&placeholder);
    let circuits = build_mlp_clipped_relu_step_circuits(
        &trace,
        num_steps,
        num_iters_per_step,
        total_iters,
        table,
    );

    let recursive_snark = run_mlp_clipped_relu_recursive(&pp, &circuits, &z0);
    verify_mlp_clipped_relu_recursive(&recursive_snark, &pp, num_steps, &z0);
}
