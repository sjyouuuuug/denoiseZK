use ff::PrimeField;

use crate::relu::{field_from_i64, ReluLookupTable};

#[derive(Clone, Debug)]
pub struct IntAffineReluLookupParams<const N: usize> {
    pub a: [[i64; N]; N],
    pub b: [i64; N],
    pub relu_table: ReluLookupTable,
}

#[derive(Clone, Debug)]
pub struct AffineReluLookupParams<F: PrimeField + Copy, const N: usize> {
    pub a: [[F; N]; N],
    pub b: [F; N],
    pub relu_table: ReluLookupTable,
}

impl<const N: usize> IntAffineReluLookupParams<N> {
    pub fn new(a: [[i64; N]; N], b: [i64; N], relu_table: ReluLookupTable) -> Self {
        Self { a, b, relu_table }
    }

    pub fn to_field<F: PrimeField + Copy>(&self) -> AffineReluLookupParams<F, N> {
        let mut a = [[F::ZERO; N]; N];
        let mut b = [F::ZERO; N];
        for r in 0..N {
            b[r] = field_from_i64::<F>(self.b[r]);
            for c in 0..N {
                a[r][c] = field_from_i64::<F>(self.a[r][c]);
            }
        }
        AffineReluLookupParams {
            a,
            b,
            relu_table: self.relu_table.clone(),
        }
    }
}
