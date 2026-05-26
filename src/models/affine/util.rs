use ff::Field;

pub fn apply_affine<F: Field + Copy, const N: usize>(
    a: &[[F; N]; N],
    b: &[F; N],
    x: &[F; N],
) -> [F; N] {
    let mut out = [F::ZERO; N];
    for r in 0..N {
        let mut acc = b[r];
        for c in 0..N {
            acc += a[r][c] * x[c];
        }
        out[r] = acc;
    }
    out
}
