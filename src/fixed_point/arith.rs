/// Mathematical floor division by a positive denominator.
/// Examples: floor_div(7,4)=1, floor_div(-7,4)=-2.
pub fn floor_div(numerator: i64, denominator: i64) -> i64 {
    assert!(denominator > 0, "denominator must be positive");
    let q = numerator / denominator; // trunc toward zero in Rust
    let r = numerator % denominator;
    if r != 0 && numerator < 0 {
        q - 1
    } else {
        q
    }
}

/// Returns (q, r) such that numerator = q * scale + r and 0 <= r < scale.
pub fn rescale_with_remainder(numerator: i64, scale: i64) -> (i64, i64) {
    let q = floor_div(numerator, scale);
    let r = numerator - q * scale;
    debug_assert!(0 <= r && r < scale);
    (q, r)
}

pub fn floor_rescale(numerator: i64, scale: i64) -> i64 {
    rescale_with_remainder(numerator, scale).0
}

pub fn mul_accumulate<const OUT: usize, const IN: usize>(
    w: &[[i64; IN]; OUT],
    x: &[i64; IN],
) -> [i64; OUT] {
    let mut out = [0i64; OUT];
    for r in 0..OUT {
        let mut acc = 0i64;
        for c in 0..IN {
            acc += w[r][c] * x[c];
        }
        out[r] = acc;
    }
    out
}
