pub fn encode_f64_round(x: f64, scale: i64) -> i64 {
    assert!(scale > 0, "scale must be positive");
    (x * scale as f64).round() as i64
}

pub fn decode_i64_to_f64(x: i64, scale: i64) -> f64 {
    assert!(scale > 0, "scale must be positive");
    x as f64 / scale as f64
}
