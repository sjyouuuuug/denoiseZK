use std::ops::Range;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PublicStateLayout {
    pub n: usize,
    pub has_output: bool,
    pub has_commitment: bool,
    pub total_iters: usize,
    pub param_block_len: usize,
    pub time_emb_dim: usize,
}

impl PublicStateLayout {
    pub fn new(
        n: usize,
        has_output: bool,
        total_iters: usize,
        param_block_len: usize,
        time_emb_dim: usize,
    ) -> Self {
        Self {
            n,
            has_output,
            has_commitment: false,
            total_iters,
            param_block_len,
            time_emb_dim,
        }
    }

    pub fn new_with_commitment(
        n: usize,
        has_output: bool,
        has_commitment: bool,
        total_iters: usize,
        param_block_len: usize,
        time_emb_dim: usize,
    ) -> Self {
        Self {
            n,
            has_output,
            has_commitment,
            total_iters,
            param_block_len,
            time_emb_dim,
        }
    }

    pub fn x_range(&self) -> Range<usize> {
        0..self.n
    }

    pub fn y_range(&self) -> Option<Range<usize>> {
        self.has_output.then_some(self.n..2 * self.n)
    }

    pub fn t_index(&self) -> usize {
        let base = if self.has_output { 2 * self.n } else { self.n };
        base + usize::from(self.has_commitment)
    }

    pub fn commitment_index(&self) -> Option<usize> {
        self.has_commitment
            .then_some(if self.has_output { 2 * self.n } else { self.n })
    }

    pub fn params_base(&self) -> usize {
        self.t_index() + 1
    }

    pub fn params_range(&self) -> Range<usize> {
        let start = self.params_base();
        start..start + self.total_iters * self.param_block_len
    }

    pub fn param_block_range(&self, block_idx: usize) -> Range<usize> {
        self.assert_block_idx(block_idx);
        let start = self.params_base() + block_idx * self.param_block_len;
        start..start + self.param_block_len
    }

    pub fn time_table_base(&self) -> usize {
        self.params_range().end
    }

    pub fn time_table_range(&self) -> Range<usize> {
        let start = self.time_table_base();
        start..start + self.total_iters * self.time_emb_dim
    }

    pub fn time_table_row_range(&self, row_idx: usize) -> Range<usize> {
        self.assert_time_row_idx(row_idx);
        let start = self.time_table_base() + row_idx * self.time_emb_dim;
        start..start + self.time_emb_dim
    }

    pub fn state_len(&self) -> usize {
        self.time_table_range().end
    }

    pub fn assert_state_len(&self, len: usize) {
        assert_eq!(
            len,
            self.state_len(),
            "public state length mismatch: got {len}, expected {}",
            self.state_len()
        );
    }

    pub fn assert_block_idx(&self, block_idx: usize) {
        assert!(
            block_idx < self.total_iters,
            "parameter block index {block_idx} out of range for total_iters {}",
            self.total_iters
        );
    }

    pub fn assert_time_row_idx(&self, row_idx: usize) {
        assert!(
            row_idx < self.total_iters,
            "time table row index {row_idx} out of range for total_iters {}",
            self.total_iters
        );
    }
}
