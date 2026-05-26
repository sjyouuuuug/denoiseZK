pub fn pad_vector_i64<const REAL: usize, const MAX: usize>(input: [i64; REAL]) -> [i64; MAX] {
    assert!(REAL <= MAX, "REAL must be <= MAX");
    let mut out = [0i64; MAX];
    out[..REAL].copy_from_slice(&input);
    out
}

pub fn assert_zero_padding_vector<const REAL: usize, const MAX: usize>(padded: &[i64; MAX]) {
    assert!(REAL <= MAX, "REAL must be <= MAX");
    for (i, value) in padded.iter().enumerate().skip(REAL) {
        assert!(
            *value == 0,
            "vector padding entry at index {i} must be zero, got {value}"
        );
    }
}

pub fn slice_real_vector<const REAL: usize, const MAX: usize>(padded: &[i64; MAX]) -> [i64; REAL] {
    assert!(REAL <= MAX, "REAL must be <= MAX");
    let mut out = [0i64; REAL];
    out.copy_from_slice(&padded[..REAL]);
    out
}
