use ff::PrimeField;
use nova_snark::frontend::{num::AllocatedNum, ConstraintSystem, SynthesisError};

use crate::{
    activations::clipped_relu::{clipped_relu_lookup, ClippedReluLookupTable},
    fixed_point::enforce_floor_rescale,
};

use super::{
    fixed_point::Conv2dFixedPointWitness,
    params::{compute_conv2d_output_dim, Conv2dPadding},
};

pub fn synthesize_fixed_point_conv2d_single_channel<
    CS,
    F,
    const IH: usize,
    const IW: usize,
    const KH: usize,
    const KW: usize,
    const OH: usize,
    const OW: usize,
>(
    cs: &mut CS,
    input: &[AllocatedNum<F>],
    kernel: &[AllocatedNum<F>],
    bias: &AllocatedNum<F>,
    padding: &Conv2dPadding,
    scale: i64,
    quotient_min: i64,
    quotient_max: i64,
    witness: &Conv2dFixedPointWitness<OH, OW>,
    prefix: &str,
) -> Result<Vec<AllocatedNum<F>>, SynthesisError>
where
    CS: ConstraintSystem<F>,
    F: PrimeField,
{
    assert_eq!(input.len(), IH * IW, "input must be IH*IW row-major");
    assert_eq!(kernel.len(), KH * KW, "kernel must be KH*KW row-major");
    let expected_oh = compute_conv2d_output_dim(IH, padding.top, padding.bottom, KH);
    let expected_ow = compute_conv2d_output_dim(IW, padding.left, padding.right, KW);
    assert_eq!(OH, expected_oh, "OH must match convolution output height");
    assert_eq!(OW, expected_ow, "OW must match convolution output width");

    let mut out = Vec::with_capacity(OH * OW);
    for oy in 0..OH {
        for ox in 0..OW {
            let mut products = Vec::with_capacity(KH * KW);
            for ky in 0..KH {
                for kx in 0..KW {
                    let iy = oy as isize + ky as isize - padding.top as isize;
                    let ix = ox as isize + kx as isize - padding.left as isize;
                    if 0 <= iy && iy < IH as isize && 0 <= ix && ix < IW as isize {
                        let product = kernel[ky * KW + kx].mul(
                            cs.namespace(|| {
                                format!("{prefix}_mul_oy_{oy}_ox_{ox}_ky_{ky}_kx_{kx}")
                            }),
                            &input[iy as usize * IW + ix as usize],
                        )?;
                        products.push(product);
                    }
                }
            }

            let raw = AllocatedNum::alloc(
                cs.namespace(|| format!("{prefix}_raw_oy_{oy}_ox_{ox}")),
                || {
                    Ok(crate::clipped_relu::field_from_i64::<F>(
                        witness.raw[oy][ox],
                    ))
                },
            )?;
            cs.enforce(
                || format!("{prefix}_raw_check_oy_{oy}_ox_{ox}"),
                |lc| {
                    products
                        .iter()
                        .fold(lc, |acc, product| acc + product.get_variable())
                },
                |lc| lc + CS::one(),
                |lc| lc + raw.get_variable(),
            );

            let q = enforce_floor_rescale(
                &mut cs.namespace(|| format!("{prefix}_rescale_oy_{oy}_ox_{ox}")),
                &raw,
                witness.quotient[oy][ox],
                witness.remainder[oy][ox],
                scale,
                quotient_min,
                quotient_max,
                &format!("{prefix}_rescale_oy_{oy}_ox_{ox}"),
            )?;
            let biased = q.add(
                cs.namespace(|| format!("{prefix}_add_bias_oy_{oy}_ox_{ox}")),
                bias,
            )?;
            let output = AllocatedNum::alloc(
                cs.namespace(|| format!("{prefix}_output_oy_{oy}_ox_{ox}")),
                || {
                    Ok(crate::clipped_relu::field_from_i64::<F>(
                        witness.output[oy][ox],
                    ))
                },
            )?;
            cs.enforce(
                || format!("{prefix}_output_check_oy_{oy}_ox_{ox}"),
                |lc| lc + biased.get_variable(),
                |lc| lc + CS::one(),
                |lc| lc + output.get_variable(),
            );
            out.push(output);
        }
    }

    Ok(out)
}

pub fn synthesize_fixed_point_conv2d_clipped_relu_single_channel<
    CS,
    F,
    const IH: usize,
    const IW: usize,
    const KH: usize,
    const KW: usize,
    const OH: usize,
    const OW: usize,
>(
    cs: &mut CS,
    input: &[AllocatedNum<F>],
    kernel: &[AllocatedNum<F>],
    bias: &AllocatedNum<F>,
    padding: &Conv2dPadding,
    scale: i64,
    quotient_min: i64,
    quotient_max: i64,
    witness: &Conv2dFixedPointWitness<OH, OW>,
    table: &ClippedReluLookupTable,
    activation_values: &[[i64; OW]; OH],
    prefix: &str,
) -> Result<Vec<AllocatedNum<F>>, SynthesisError>
where
    CS: ConstraintSystem<F>,
    F: PrimeField,
{
    let pre = synthesize_fixed_point_conv2d_single_channel::<CS, F, IH, IW, KH, KW, OH, OW>(
        cs,
        input,
        kernel,
        bias,
        padding,
        scale,
        quotient_min,
        quotient_max,
        witness,
        &format!("{prefix}_conv"),
    )?;
    let mut act = Vec::with_capacity(OH * OW);
    for oy in 0..OH {
        for ox in 0..OW {
            let idx = oy * OW + ox;
            act.push(clipped_relu_lookup(
                &mut cs.namespace(|| format!("{prefix}_relu_oy_{oy}_ox_{ox}")),
                &pre[idx],
                witness.output[oy][ox],
                crate::clipped_relu::field_from_i64::<F>(activation_values[oy][ox]),
                table,
                &format!("{prefix}_relu_oy_{oy}_ox_{ox}"),
            )?);
        }
    }
    Ok(act)
}
