use super::ClippedReluLookupTable;
use crate::fixed_point::FixedPointConfig;

pub fn fixed_point_clipped_relu(x: i64, config: &FixedPointConfig) -> i64 {
    let table = config.clipped_relu_table();
    table.clipped_relu(x)
}

pub fn fixed_point_clipped_relu_table(config: &FixedPointConfig) -> ClippedReluLookupTable {
    config.clipped_relu_table()
}
