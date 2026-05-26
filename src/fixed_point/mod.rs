pub mod arith;
pub mod config;
pub mod encode;
pub mod gadget;
pub mod range_bits;

pub use arith::{floor_div, floor_rescale, mul_accumulate, rescale_with_remainder};
pub use config::FixedPointConfig;
pub use encode::{decode_i64_to_f64, encode_f64_round};
pub use gadget::{
    enforce_floor_rescale, enforce_remainder_range, enforce_signed_range,
    set_signed_range_check_mode, signed_range_check_mode, SignedRangeCheckMode,
};
pub use range_bits::{ceil_log2_power_of_two, enforce_signed_range_bits, offset_binary_bits};
