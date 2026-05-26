use crate::fixed_point::{floor_rescale, mul_accumulate, rescale_with_remainder};

/// Fixed-point affine with public integer parameters:
/// y = floor((W*x)/scale) + b.
pub fn apply_affine_fixed_point<const OUT: usize, const IN: usize>(
    w: &[[i64; IN]; OUT],
    b: &[i64; OUT],
    x: &[i64; IN],
    scale: i64,
) -> [i64; OUT] {
    let raw = mul_accumulate(w, x);
    let mut y = [0i64; OUT];
    for r in 0..OUT {
        y[r] = floor_rescale(raw[r], scale) + b[r];
    }
    y
}

/// Returns (raw_sum, quotient, remainder, y) per output coordinate.
pub fn apply_affine_fixed_point_with_witness<const OUT: usize, const IN: usize>(
    w: &[[i64; IN]; OUT],
    b: &[i64; OUT],
    x: &[i64; IN],
    scale: i64,
) -> ([i64; OUT], [i64; OUT], [i64; OUT], [i64; OUT]) {
    let raw = mul_accumulate(w, x);
    let mut quotient = [0i64; OUT];
    let mut remainder = [0i64; OUT];
    let mut y = [0i64; OUT];
    for r in 0..OUT {
        let (q, rem) = rescale_with_remainder(raw[r], scale);
        quotient[r] = q;
        remainder[r] = rem;
        y[r] = q + b[r];
    }
    (raw, quotient, remainder, y)
}
