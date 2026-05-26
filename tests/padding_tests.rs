use denoise::padding::{
    assert_zero_padding_matrix, assert_zero_padding_vector, pad_matrix_i64, pad_vector_i64,
};

#[test]
fn pad_vector_i64_pads_with_zero() {
    assert_eq!(pad_vector_i64::<2, 4>([1, 2]), [1, 2, 0, 0]);
}

#[test]
fn pad_matrix_i64_pads_rows_and_cols() {
    let padded = pad_matrix_i64::<2, 2, 3, 4>([[1, 2], [3, 4]]);
    assert_eq!(padded, [[1, 2, 0, 0], [3, 4, 0, 0], [0, 0, 0, 0]]);
}

#[test]
#[should_panic(expected = "index 3")]
fn assert_zero_padding_vector_rejects_nonzero_padding() {
    assert_zero_padding_vector::<2, 4>(&[1, 2, 0, 5]);
}

#[test]
#[should_panic(expected = "row 2, col 1")]
fn assert_zero_padding_matrix_rejects_nonzero_padding() {
    let padded = [[1, 2, 0], [3, 4, 0], [0, 9, 0]];
    assert_zero_padding_matrix::<2, 2, 3, 3>(&padded);
}
