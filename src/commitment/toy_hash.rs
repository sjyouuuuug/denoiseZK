use ff::PrimeField;

pub const TOY_HASH_BASE_U64: u64 = 131;

pub fn toy_hash_i64(values: &[i64], base: i64, init: i64) -> i64 {
    let mut h = init;
    for &v in values {
        h = h * base + v;
    }
    h
}

pub fn toy_hash_i128(values: &[i64], base: i128, init: i128) -> i128 {
    let mut h = init;
    for &v in values {
        h = h * base + v as i128;
    }
    h
}

pub fn field_from_i64<F: PrimeField>(v: i64) -> F {
    crate::clipped_relu::field_from_i64(v)
}

pub fn toy_hash_field<F: PrimeField>(values: &[F], base: F, init: F) -> F {
    let mut h = init;
    for &v in values {
        h = h * base + v;
    }
    h
}

pub fn toy_hash_i64_as_field<F: PrimeField>(values: &[i64], base: u64, init: i64) -> F {
    let values_f: Vec<F> = values.iter().map(|&v| field_from_i64(v)).collect();
    toy_hash_field(&values_f, F::from(base), field_from_i64(init))
}

pub fn toy_hash_prefixes_i64_as_field<F: PrimeField>(
    values: &[i64],
    base: u64,
    init: i64,
) -> Vec<F> {
    let base_f = F::from(base);
    let mut h = field_from_i64(init);
    let mut prefixes = Vec::with_capacity(values.len());
    for &v in values {
        h = h * base_f + field_from_i64::<F>(v);
        prefixes.push(h);
    }
    prefixes
}

pub fn toy_hash_block_prefixes_i64_from_field<F: PrimeField>(
    values: &[i64],
    base: u64,
    init: F,
) -> (Vec<F>, F) {
    let base_f = F::from(base);
    let mut h = init;
    let mut prefixes = Vec::with_capacity(values.len());
    for &v in values {
        h = h * base_f + field_from_i64::<F>(v);
        prefixes.push(h);
    }
    (prefixes, h)
}
