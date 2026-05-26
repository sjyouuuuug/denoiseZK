use denoise::{
    activations::clipped_relu::ClippedReluLookupTable,
    denoise_fixed_point_conv::{assert_flat_image_padding_zero, assert_kernel_padding_zero},
    fixed_point::rescale_with_remainder,
    layers::conv2d::{
        apply_conv2d_fixed_point_with_witness, compute_conv2d_output_dim, Conv2dPadding,
        Conv2dRealShape, FixedConv2dParams,
    },
};

#[test]
fn conv2d_fixed_point_tests() {
    conv2d_no_padding_1x1_kernel_matches_input_scale();
}

#[test]
fn conv2d_no_padding_1x1_kernel_matches_input_scale() {
    let padding = Conv2dPadding {
        top: 0,
        bottom: 0,
        left: 0,
        right: 0,
    };
    let input = [[16, -8], [4, 0]];
    let kernel = [[16]];
    let w =
        apply_conv2d_fixed_point_with_witness::<2, 2, 1, 1, 2, 2>(&input, &kernel, 0, &padding, 16);
    assert_eq!(w.output, input);
}

#[test]
fn conv2d_same_padding_3x3_small_hand_computed() {
    let input = [[16, 0], [0, 0]];
    let kernel = [[16, 16, 16], [16, 16, 16], [16, 16, 16]];
    let w = apply_conv2d_fixed_point_with_witness::<2, 2, 3, 3, 2, 2>(
        &input,
        &kernel,
        0,
        &Conv2dPadding::same_3x3(),
        16,
    );
    assert_eq!(w.raw, [[256, 256], [256, 256]]);
    assert_eq!(w.output, [[16, 16], [16, 16]]);
}

#[test]
fn conv2d_same_padding_5x5_3x3_matches_reference() {
    const IH: usize = 5;
    const IW: usize = 5;
    const KH: usize = 3;
    const KW: usize = 3;
    const OH: usize = 5;
    const OW: usize = 5;
    let input = [
        [16, -8, 4, 0, 12],
        [-4, 8, -12, 16, 0],
        [4, 0, 12, -8, 8],
        [0, -16, 8, 4, -4],
        [12, 4, 0, -8, 16],
    ];
    let kernel = [[4, -2, 0], [2, 8, 2], [0, -2, 4]];
    let bias = 1;
    let padding = Conv2dPadding::same_3x3();
    let witness = apply_conv2d_fixed_point_with_witness::<IH, IW, KH, KW, OH, OW>(
        &input, &kernel, bias, &padding, 16,
    );

    let mut expected = [[0i64; OW]; OH];
    for oy in 0..OH {
        for ox in 0..OW {
            let mut raw = 0i64;
            for ky in 0..KH {
                for kx in 0..KW {
                    let iy = oy as isize + ky as isize - padding.top as isize;
                    let ix = ox as isize + kx as isize - padding.left as isize;
                    if 0 <= iy && iy < IH as isize && 0 <= ix && ix < IW as isize {
                        raw += kernel[ky][kx] * input[iy as usize][ix as usize];
                    }
                }
            }
            let (q, r) = rescale_with_remainder(raw, 16);
            assert!((0..16).contains(&r));
            expected[oy][ox] = q + bias;
        }
    }

    assert_eq!(witness.output, expected);
}

#[test]
fn conv2d_padding_out_of_bounds_uses_zero() {
    let input = [[16]];
    let kernel = [[16, 16, 16], [16, 16, 16], [16, 16, 16]];
    let w = apply_conv2d_fixed_point_with_witness::<1, 1, 3, 3, 1, 1>(
        &input,
        &kernel,
        0,
        &Conv2dPadding::same_3x3(),
        16,
    );
    assert_eq!(w.raw, [[256]]);
    assert_eq!(w.output, [[16]]);
}

#[test]
#[should_panic(expected = "invalid conv shape")]
fn conv2d_output_dim_rejects_invalid_shape() {
    let _ = compute_conv2d_output_dim(2, 0, 0, 5);
}

#[test]
fn conv2d_fixed_point_floor_handles_negative_raw() {
    let padding = Conv2dPadding {
        top: 0,
        bottom: 0,
        left: 0,
        right: 0,
    };
    let input = [[-7]];
    let kernel = [[1]];
    let w =
        apply_conv2d_fixed_point_with_witness::<1, 1, 1, 1, 1, 1>(&input, &kernel, 0, &padding, 4);
    assert_eq!(w.raw, [[-7]]);
    assert_eq!(w.quotient, [[-2]]);
    assert_eq!(w.remainder, [[1]]);
}

#[test]
#[should_panic(expected = "kernel padding")]
fn conv2d_kernel_padding_zero_check_rejects_nonzero_padding() {
    let kernel = [[4, 1, 0], [0, 0, 0], [0, 0, 0]];
    let shape = Conv2dRealShape::new(2, 2, 1, 1, 2, 2);
    assert_kernel_padding_zero::<3, 3>(&kernel, &shape);
}

#[test]
#[should_panic(expected = "spatial padding")]
fn conv2d_flat_image_padding_zero_check_rejects_nonzero_padding() {
    let flat = [1, 2, 7, 0, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    let shape = Conv2dRealShape::new(2, 2, 1, 1, 2, 2);
    assert_flat_image_padding_zero::<4, 4>(&flat, &shape, "image");
}

#[test]
fn conv2d_kernel_flatten_layout_is_row_major() {
    let params = FixedConv2dParams::<2, 2, 2, 2, 1, 1>::new(
        [[1, 2], [3, 4]],
        5,
        Conv2dPadding {
            top: 0,
            bottom: 0,
            left: 0,
            right: 0,
        },
    );
    assert_eq!(FixedConv2dParams::<2, 2, 2, 2, 1, 1>::block_len(), 5);
    assert_eq!(params.flatten_i64(), vec![1, 2, 3, 4, 5]);
}

#[test]
#[should_panic(expected = "outside")]
fn conv2d_clipped_relu_rejects_out_of_range() {
    let table = ClippedReluLookupTable::new(-4, 4, 2);
    let w = apply_conv2d_fixed_point_with_witness::<1, 1, 1, 1, 1, 1>(
        &[[16]],
        &[[16]],
        0,
        &Conv2dPadding {
            top: 0,
            bottom: 0,
            left: 0,
            right: 0,
        },
        16,
    );
    assert!(
        table.contains(w.output[0][0]),
        "conv preactivation {} is outside clipped ReLU range",
        w.output[0][0]
    );
}
