#[derive(Clone, Debug)]
pub struct ClippedReluLookupTable {
    pub min: i64,
    pub max: i64,
    pub clip_max: i64,
}

impl ClippedReluLookupTable {
    pub fn new(min: i64, max: i64, clip_max: i64) -> Self {
        assert!(min <= max, "lookup table requires min <= max");
        assert!(clip_max >= 0, "clipped ReLU requires nonnegative clip_max");
        assert!(
            clip_max <= max,
            "clip_max must fit inside the table domain upper bound"
        );
        Self { min, max, clip_max }
    }

    pub fn clipped_relu(&self, x: i64) -> i64 {
        assert!(
            self.contains(x),
            "input {x} outside clipped ReLU table range [{}, {}]",
            self.min,
            self.max
        );
        x.max(0).min(self.clip_max)
    }

    pub fn contains(&self, x: i64) -> bool {
        self.min <= x && x <= self.max
    }

    pub fn entries(&self) -> Vec<(i64, i64)> {
        (self.min..=self.max)
            .map(|x| (x, x.max(0).min(self.clip_max)))
            .collect()
    }

    pub fn size(&self) -> usize {
        (self.max - self.min + 1) as usize
    }
}
