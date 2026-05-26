pub mod fixed_point;
pub mod gadget;
pub mod params;

pub use fixed_point::{apply_conv2d_fixed_point_with_witness, Conv2dFixedPointWitness};
pub use gadget::{
    synthesize_fixed_point_conv2d_clipped_relu_single_channel,
    synthesize_fixed_point_conv2d_single_channel,
};
pub use params::{
    compute_conv2d_output_dim, is_real_input_coord, is_real_kernel_coord, is_real_output_coord,
    Conv2dPadding, Conv2dRealShape, FixedConv2dParams,
};
