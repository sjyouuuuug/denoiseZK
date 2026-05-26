#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DenoiseUpdateMode {
    DoubleFloor,
    FusedFloor,
}

impl Default for DenoiseUpdateMode {
    fn default() -> Self {
        Self::DoubleFloor
    }
}
