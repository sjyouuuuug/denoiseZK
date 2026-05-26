pub mod gadget;
pub mod toy_hash;

pub use gadget::{
    synthesize_toy_hash_block, synthesize_toy_hash_sequence, synthesize_toy_hash_sequence_return,
    synthesize_toy_hash_update,
};
pub use toy_hash::{
    field_from_i64, toy_hash_block_prefixes_i64_from_field, toy_hash_field, toy_hash_i128,
    toy_hash_i64, toy_hash_i64_as_field, toy_hash_prefixes_i64_as_field, TOY_HASH_BASE_U64,
};
