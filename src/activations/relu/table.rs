#[derive(Clone, Debug)]
pub struct ReluLookupTable {
    pub min: i64,
    pub max: i64,
}

impl ReluLookupTable {
    pub fn new(min: i64, max: i64) -> Self {
        assert!(min <= max, "lookup table requires min <= max");
        Self { min, max }
    }

    pub fn relu(&self, x: i64) -> i64 {
        assert!(
            self.contains(x),
            "input {x} outside lookup table range [{}, {}]",
            self.min,
            self.max
        );
        x.max(0)
    }

    pub fn contains(&self, x: i64) -> bool {
        self.min <= x && x <= self.max
    }

    pub fn entries(&self) -> Vec<(i64, i64)> {
        (self.min..=self.max).map(|x| (x, x.max(0))).collect()
    }

    pub fn size(&self) -> usize {
        (self.max - self.min + 1) as usize
    }
}
