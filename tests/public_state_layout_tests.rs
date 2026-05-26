use denoise::public_state::PublicStateLayout;

#[test]
fn public_state_layout_tests() {
    let layout = PublicStateLayout::new(2, true, 1, 5, 3);
    assert_eq!(layout.state_len(), 2 + 2 + 1 + 5 + 3);
}

#[test]
fn layout_without_output_has_expected_ranges() {
    let layout = PublicStateLayout::new(4, false, 3, 10, 2);
    assert_eq!(layout.x_range(), 0..4);
    assert_eq!(layout.y_range(), None);
    assert_eq!(layout.t_index(), 4);
    assert_eq!(layout.params_base(), 5);
    assert_eq!(layout.params_range(), 5..35);
    assert_eq!(layout.time_table_range(), 35..41);
}

#[test]
fn layout_with_output_has_expected_ranges() {
    let layout = PublicStateLayout::new(4, true, 3, 10, 2);
    assert_eq!(layout.x_range(), 0..4);
    assert_eq!(layout.y_range(), Some(4..8));
    assert_eq!(layout.t_index(), 8);
    assert_eq!(layout.params_base(), 9);
    assert_eq!(layout.params_range(), 9..39);
    assert_eq!(layout.time_table_range(), 39..45);
}

#[test]
fn layout_with_output_and_commitment_has_expected_ranges() {
    let layout = PublicStateLayout::new_with_commitment(4, true, true, 3, 10, 2);
    assert_eq!(layout.x_range(), 0..4);
    assert_eq!(layout.y_range(), Some(4..8));
    assert_eq!(layout.commitment_index(), Some(8));
    assert_eq!(layout.t_index(), 9);
    assert_eq!(layout.params_base(), 10);
    assert_eq!(layout.params_range(), 10..40);
    assert_eq!(layout.time_table_range(), 40..46);
}

#[test]
fn layout_with_commitment_without_output_has_expected_ranges() {
    let layout = PublicStateLayout::new_with_commitment(4, false, true, 3, 10, 2);
    assert_eq!(layout.x_range(), 0..4);
    assert_eq!(layout.y_range(), None);
    assert_eq!(layout.commitment_index(), Some(4));
    assert_eq!(layout.t_index(), 5);
    assert_eq!(layout.params_base(), 6);
    assert_eq!(layout.params_range(), 6..36);
    assert_eq!(layout.time_table_range(), 36..42);
}

#[test]
fn param_block_range_is_correct() {
    let layout = PublicStateLayout::new(2, false, 4, 7, 3);
    assert_eq!(layout.param_block_range(0), 3..10);
    assert_eq!(layout.param_block_range(2), 17..24);
}

#[test]
fn time_table_row_range_is_correct() {
    let layout = PublicStateLayout::new(2, true, 4, 7, 3);
    assert_eq!(layout.time_table_row_range(0), 33..36);
    assert_eq!(layout.time_table_row_range(3), 42..45);
}

#[test]
fn state_len_is_correct() {
    let without_output = PublicStateLayout::new(4, false, 3, 10, 2);
    let with_output = PublicStateLayout::new(4, true, 3, 10, 2);
    assert_eq!(without_output.state_len(), 41);
    assert_eq!(with_output.state_len(), 45);
    without_output.assert_state_len(41);
    with_output.assert_state_len(45);
}

#[test]
#[should_panic(expected = "parameter block index")]
fn out_of_range_block_idx_panics() {
    let layout = PublicStateLayout::new(2, false, 4, 7, 3);
    let _ = layout.param_block_range(4);
}

#[test]
#[should_panic(expected = "time table row index")]
fn out_of_range_time_row_idx_panics() {
    let layout = PublicStateLayout::new(2, false, 4, 7, 3);
    let _ = layout.time_table_row_range(4);
}
