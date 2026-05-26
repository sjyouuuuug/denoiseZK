pub fn pad_matrix_i64<
    const R_REAL: usize,
    const C_REAL: usize,
    const R_MAX: usize,
    const C_MAX: usize,
>(
    input: [[i64; C_REAL]; R_REAL],
) -> [[i64; C_MAX]; R_MAX] {
    assert!(R_REAL <= R_MAX, "R_REAL must be <= R_MAX");
    assert!(C_REAL <= C_MAX, "C_REAL must be <= C_MAX");
    let mut out = [[0i64; C_MAX]; R_MAX];
    for r in 0..R_REAL {
        out[r][..C_REAL].copy_from_slice(&input[r]);
    }
    out
}

pub fn assert_zero_padding_matrix<
    const R_REAL: usize,
    const C_REAL: usize,
    const R_MAX: usize,
    const C_MAX: usize,
>(
    padded: &[[i64; C_MAX]; R_MAX],
) {
    assert!(R_REAL <= R_MAX, "R_REAL must be <= R_MAX");
    assert!(C_REAL <= C_MAX, "C_REAL must be <= C_MAX");
    for (r, row) in padded.iter().enumerate() {
        for (c, value) in row.iter().enumerate() {
            if r >= R_REAL || c >= C_REAL {
                assert!(
                    *value == 0,
                    "matrix padding entry at row {r}, col {c} must be zero, got {value}"
                );
            }
        }
    }
}
