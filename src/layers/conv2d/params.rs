use crate::fixed_point::encode_f64_round;

#[derive(Clone, Debug)]
pub struct Conv2dPadding {
    pub top: usize,
    pub bottom: usize,
    pub left: usize,
    pub right: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Conv2dRealShape {
    pub ih_real: usize,
    pub iw_real: usize,
    pub kh_real: usize,
    pub kw_real: usize,
    pub oh_real: usize,
    pub ow_real: usize,
}

impl Conv2dRealShape {
    pub fn new(
        ih_real: usize,
        iw_real: usize,
        kh_real: usize,
        kw_real: usize,
        oh_real: usize,
        ow_real: usize,
    ) -> Self {
        Self {
            ih_real,
            iw_real,
            kh_real,
            kw_real,
            oh_real,
            ow_real,
        }
    }

    pub fn full<
        const IH: usize,
        const IW: usize,
        const KH: usize,
        const KW: usize,
        const OH: usize,
        const OW: usize,
    >() -> Self {
        Self::new(IH, IW, KH, KW, OH, OW)
    }

    pub fn n_real(&self) -> usize {
        self.ih_real * self.iw_real
    }

    pub fn output_real_len(&self) -> usize {
        self.oh_real * self.ow_real
    }

    pub fn assert_fits<
        const IH: usize,
        const IW: usize,
        const KH: usize,
        const KW: usize,
        const OH: usize,
        const OW: usize,
    >(
        &self,
    ) {
        assert!(self.ih_real <= IH, "ih_real must be <= IH");
        assert!(self.iw_real <= IW, "iw_real must be <= IW");
        assert!(self.kh_real <= KH, "kh_real must be <= KH");
        assert!(self.kw_real <= KW, "kw_real must be <= KW");
        assert!(self.oh_real <= OH, "oh_real must be <= OH");
        assert!(self.ow_real <= OW, "ow_real must be <= OW");
    }
}

pub fn is_real_input_coord(row: usize, col: usize, shape: &Conv2dRealShape) -> bool {
    row < shape.ih_real && col < shape.iw_real
}

pub fn is_real_output_coord(row: usize, col: usize, shape: &Conv2dRealShape) -> bool {
    row < shape.oh_real && col < shape.ow_real
}

pub fn is_real_kernel_coord(row: usize, col: usize, shape: &Conv2dRealShape) -> bool {
    row < shape.kh_real && col < shape.kw_real
}

pub fn compute_conv2d_output_dim(
    input: usize,
    pad_before: usize,
    pad_after: usize,
    kernel: usize,
) -> usize {
    assert!(
        input + pad_before + pad_after >= kernel,
        "invalid conv shape: padded input smaller than kernel"
    );
    input + pad_before + pad_after - kernel + 1
}

impl Default for Conv2dPadding {
    fn default() -> Self {
        Self {
            top: 0,
            bottom: 0,
            left: 0,
            right: 0,
        }
    }
}

impl Conv2dPadding {
    pub fn same_3x3() -> Self {
        Self {
            top: 1,
            bottom: 1,
            left: 1,
            right: 1,
        }
    }
}

#[derive(Clone, Debug)]
pub struct FixedConv2dParams<
    const IH: usize,
    const IW: usize,
    const KH: usize,
    const KW: usize,
    const OH: usize,
    const OW: usize,
> {
    pub kernel: [[i64; KW]; KH],
    pub bias: i64,
    pub padding: Conv2dPadding,
}

impl<
        const IH: usize,
        const IW: usize,
        const KH: usize,
        const KW: usize,
        const OH: usize,
        const OW: usize,
    > FixedConv2dParams<IH, IW, KH, KW, OH, OW>
{
    pub fn new(kernel: [[i64; KW]; KH], bias: i64, padding: Conv2dPadding) -> Self {
        Self {
            kernel,
            bias,
            padding,
        }
    }

    pub fn from_f64(
        kernel: [[f64; KW]; KH],
        bias: f64,
        padding: Conv2dPadding,
        scale: i64,
    ) -> Self {
        let mut kernel_i = [[0i64; KW]; KH];
        for r in 0..KH {
            for c in 0..KW {
                kernel_i[r][c] = encode_f64_round(kernel[r][c], scale);
            }
        }
        Self::new(kernel_i, encode_f64_round(bias, scale), padding)
    }

    pub fn block_len() -> usize {
        KH * KW + 1
    }

    pub fn flatten_i64(&self) -> Vec<i64> {
        let mut out = Vec::with_capacity(Self::block_len());
        for r in 0..KH {
            for c in 0..KW {
                out.push(self.kernel[r][c]);
            }
        }
        out.push(self.bias);
        out
    }
}
