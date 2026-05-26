use ff::Field;

#[derive(Clone, Debug)]
pub struct AffineParams<F: Field + Copy, const N: usize> {
    pub a: [[F; N]; N],
    pub b: [F; N],
}

impl<F: Field + Copy, const N: usize> AffineParams<F, N> {
    pub fn new(a: [[F; N]; N], b: [F; N]) -> Self {
        Self { a, b }
    }
}
