use crate::clipped_relu::ClippedReluLookupTable;

#[derive(Clone, Debug)]
pub struct FixedPointConfig {
    pub scale: i64,
    pub relu_min: i64,
    pub relu_max: i64,
    pub clip_max: i64,
    pub quotient_min: i64,
    pub quotient_max: i64,
    pub value_min: i64,
    pub value_max: i64,
}

impl FixedPointConfig {
    /// Default: scale=16, ReLU lookup domain [-4,4] in real units,
    /// and clip upper bound 2.0 in real units.
    pub fn default_scale16() -> Self {
        Self::from_real_bounds(16, -4, 4, 2)
    }

    /// Bounds are given in real/integer units before multiplying by scale.
    /// Example: scale=16, clip_units=2 => clip_max=32.
    pub fn from_real_bounds(
        scale: i64,
        relu_min_units: i64,
        relu_max_units: i64,
        clip_units: i64,
    ) -> Self {
        assert!(scale > 0, "fixed-point scale must be positive");
        let relu_min = relu_min_units * scale;
        let relu_max = relu_max_units * scale;
        let clip_max = clip_units * scale;
        assert!(relu_min <= relu_max, "invalid ReLU domain");
        assert!(clip_max >= 0, "clip_max must be nonnegative");
        assert!(
            clip_max <= relu_max,
            "clip_max must fit inside ReLU table upper bound"
        );
        let value_min = -128;
        let value_max = 127;
        let quotient_min = -256;
        let quotient_max = 255;
        Self {
            scale,
            relu_min,
            relu_max,
            clip_max,
            quotient_min,
            quotient_max,
            value_min,
            value_max,
        }
    }

    pub fn with_integer_ranges(
        mut self,
        quotient_min: i64,
        quotient_max: i64,
        value_min: i64,
        value_max: i64,
    ) -> Self {
        assert!(
            quotient_min <= quotient_max,
            "invalid quotient signed range"
        );
        assert!(value_min <= value_max, "invalid value signed range");
        self.quotient_min = quotient_min;
        self.quotient_max = quotient_max;
        self.value_min = value_min;
        self.value_max = value_max;
        self
    }

    pub fn clipped_relu_table(&self) -> ClippedReluLookupTable {
        ClippedReluLookupTable::new(self.relu_min, self.relu_max, self.clip_max)
    }
}
