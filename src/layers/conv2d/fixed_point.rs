use crate::fixed_point::rescale_with_remainder;

use super::params::{compute_conv2d_output_dim, Conv2dPadding};

#[derive(Clone, Debug)]
pub struct Conv2dFixedPointWitness<const OH: usize, const OW: usize> {
    pub raw: [[i64; OW]; OH],
    pub quotient: [[i64; OW]; OH],
    pub remainder: [[i64; OW]; OH],
    pub output: [[i64; OW]; OH],
}

pub fn apply_conv2d_fixed_point_with_witness<
    const IH: usize,
    const IW: usize,
    const KH: usize,
    const KW: usize,
    const OH: usize,
    const OW: usize,
>(
    input: &[[i64; IW]; IH],
    kernel: &[[i64; KW]; KH],
    bias: i64,
    padding: &Conv2dPadding,
    scale: i64,
) -> Conv2dFixedPointWitness<OH, OW> {
    let expected_oh = compute_conv2d_output_dim(IH, padding.top, padding.bottom, KH);
    let expected_ow = compute_conv2d_output_dim(IW, padding.left, padding.right, KW);
    assert_eq!(OH, expected_oh, "OH must match convolution output height");
    assert_eq!(OW, expected_ow, "OW must match convolution output width");

    let mut raw = [[0i64; OW]; OH];
    let mut quotient = [[0i64; OW]; OH];
    let mut remainder = [[0i64; OW]; OH];
    let mut output = [[0i64; OW]; OH];

    for oy in 0..OH {
        for ox in 0..OW {
            let mut acc = 0i64;
            for ky in 0..KH {
                for kx in 0..KW {
                    let iy = oy as isize + ky as isize - padding.top as isize;
                    let ix = ox as isize + kx as isize - padding.left as isize;
                    if 0 <= iy && iy < IH as isize && 0 <= ix && ix < IW as isize {
                        acc += kernel[ky][kx] * input[iy as usize][ix as usize];
                    }
                }
            }
            let (q, r) = rescale_with_remainder(acc, scale);
            raw[oy][ox] = acc;
            quotient[oy][ox] = q;
            remainder[oy][ox] = r;
            output[oy][ox] = q + bias;
        }
    }

    Conv2dFixedPointWitness {
        raw,
        quotient,
        remainder,
        output,
    }
}
