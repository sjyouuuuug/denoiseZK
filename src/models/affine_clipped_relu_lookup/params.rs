use ff::PrimeField;

use crate::clipped_relu::{field_from_i64, ClippedReluLookupTable};

#[derive(Clone, Debug)]
pub struct IntAffineClippedReluLookupParams<const N: usize> {
    pub a: [[i64; N]; N],
    pub b: [i64; N],
    pub clipped_relu_table: ClippedReluLookupTable,
}

#[derive(Clone, Debug)]
pub struct AffineClippedReluLookupParams<F: PrimeField + Copy, const N: usize> {
    pub a: [[F; N]; N],
    pub b: [F; N],
    pub clipped_relu_table: ClippedReluLookupTable,
}

impl<const N: usize> IntAffineClippedReluLookupParams<N> {
    pub fn new(a: [[i64; N]; N], b: [i64; N], clipped_relu_table: ClippedReluLookupTable) -> Self {
        Self {
            a,
            b,
            clipped_relu_table,
        }
    }

    pub fn to_field<F: PrimeField + Copy>(&self) -> AffineClippedReluLookupParams<F, N> {
        let mut a = [[F::ZERO; N]; N];
        let mut b = [F::ZERO; N];
        for r in 0..N {
            b[r] = field_from_i64::<F>(self.b[r]);
            for c in 0..N {
                a[r][c] = field_from_i64::<F>(self.a[r][c]);
            }
        }
        AffineClippedReluLookupParams {
            a,
            b,
            clipped_relu_table: self.clipped_relu_table.clone(),
        }
    }
}
