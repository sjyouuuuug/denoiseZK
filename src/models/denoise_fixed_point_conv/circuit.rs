use ff::PrimeField;
use nova_snark::{
    frontend::{num::AllocatedNum, ConstraintSystem, SynthesisError},
    traits::{circuit::StepCircuit, Group},
};

use crate::{
    commitment::{synthesize_toy_hash_block, TOY_HASH_BASE_U64},
    fixed_point::{enforce_floor_rescale, enforce_signed_range_bits, FixedPointConfig},
    layers::conv2d::{
        is_real_input_coord, is_real_kernel_coord, is_real_output_coord,
        synthesize_fixed_point_conv2d_clipped_relu_single_channel, Conv2dRealShape,
    },
    models::denoise_fixed_point_time_embedding::time_embedding::synthesize_time_embedding_lookup,
    models::denoise_update::{synthesize_denoise_update, DenoiseUpdateMode},
    padding::enforce_zero,
    public_state::{
        enforce_equal_if, synthesize_is_equal_to_constant, CommittedDenoiseStateLayout,
        PublicStateLayout,
    },
};

use super::{params::FixedDenoiseConvStepParams, trace::FixedDenoiseConvIteration};

#[derive(Clone, Debug)]
pub struct PublicFixedPointDenoiseConvCircuit<
    G: Group,
    const N: usize,
    const IH: usize,
    const IW: usize,
    const TE: usize,
    const KH: usize,
    const KW: usize,
    const OH: usize,
    const OW: usize,
> {
    pub num_iters_per_step: usize,
    pub total_iters: usize,
    pub config: FixedPointConfig,
    pub clipped_relu_table: crate::clipped_relu::ClippedReluLookupTable,
    pub padding: crate::layers::conv2d::Conv2dPadding,
    pub time_table_values: Vec<[i64; TE]>,
    pub real_shape: Conv2dRealShape,
    pub te_real: usize,
    pub bind_output: bool,
    pub commit_params: bool,
    pub update_mode: DenoiseUpdateMode,
    pub param_hash_witnesses: Vec<Vec<G::Scalar>>,
    pub seq: Vec<FixedDenoiseConvIteration<G::Scalar, N, IH, IW, TE, KH, KW, OH, OW>>,
}

fn block_len<const TE: usize, const KH: usize, const KW: usize>() -> usize {
    FixedDenoiseConvStepParams::<TE, KH, KW>::block_len()
}

fn alloc_zero<CS, F>(
    cs: &mut CS,
    name: impl FnOnce() -> String,
) -> Result<AllocatedNum<F>, SynthesisError>
where
    CS: ConstraintSystem<F>,
    F: PrimeField,
{
    let zero = AllocatedNum::alloc(cs.namespace(name), || Ok(F::ZERO))?;
    cs.enforce(
        || "enforce_zero",
        |lc| lc + zero.get_variable(),
        |lc| lc + CS::one(),
        |lc| lc,
    );
    Ok(zero)
}

impl<
        G: Group,
        const N: usize,
        const IH: usize,
        const IW: usize,
        const TE: usize,
        const KH: usize,
        const KW: usize,
        const OH: usize,
        const OW: usize,
    > StepCircuit<G::Scalar>
    for PublicFixedPointDenoiseConvCircuit<G, N, IH, IW, TE, KH, KW, OH, OW>
{
    fn arity(&self) -> usize {
        let p = block_len::<TE, KH, KW>();
        if self.commit_params {
            CommittedDenoiseStateLayout::new(N, true, self.total_iters, p, TE).state_len()
        } else {
            PublicStateLayout::new(N, self.bind_output, self.total_iters, p, TE).state_len()
        }
    }

    fn synthesize<CS: ConstraintSystem<G::Scalar>>(
        &self,
        cs: &mut CS,
        z: &[AllocatedNum<G::Scalar>],
    ) -> Result<Vec<AllocatedNum<G::Scalar>>, SynthesisError> {
        assert_eq!(N, IH * IW, "N must equal IH*IW");
        assert_eq!(N, OH * OW, "N must equal OH*OW");
        assert_eq!(OH, IH, "denoise conv backend currently requires OH == IH");
        assert_eq!(OW, IW, "denoise conv backend currently requires OW == IW");
        self.real_shape.assert_fits::<IH, IW, KH, KW, OH, OW>();
        assert!(self.te_real <= TE, "te_real must be <= TE");
        assert_eq!(
            self.seq.len(),
            self.num_iters_per_step,
            "witness length must equal num_iters_per_step"
        );
        if self.commit_params {
            assert_eq!(
                self.param_hash_witnesses.len(),
                self.num_iters_per_step,
                "hash witness chunks must match num_iters_per_step"
            );
        }

        let p = block_len::<TE, KH, KW>();
        let (
            layout_len,
            params_base,
            table_range,
            mut x_i,
            expected_y,
            mut h_var,
            c_var,
            mut t_var,
        ) = if self.commit_params {
            let layout = CommittedDenoiseStateLayout::new(N, true, self.total_iters, p, TE);
            layout.assert_state_len(z.len());
            let y = z[layout.y_range().expect("output range is present")].to_vec();
            for row in 0..IH {
                for col in 0..IW {
                    if !is_real_input_coord(row, col, &self.real_shape) {
                        let idx = row * IW + col;
                        enforce_zero(
                            &mut cs.namespace(|| {
                                format!(
                                    "conv_commit_expected_y_padding_row_{row}_col_{col}_idx_{idx}"
                                )
                            }),
                            &y[idx],
                            || {
                                format!(
                                    "conv_commit_expected_y_padding_row_{row}_col_{col}_idx_{idx}"
                                )
                            },
                        )?;
                    }
                }
            }
            (
                layout.state_len(),
                layout.params_base(),
                layout.time_table_range(),
                z[layout.x_range()].to_vec(),
                Some(y),
                Some(z[layout.h_index()].clone()),
                Some(z[layout.c_index()].clone()),
                z[layout.t_index()].clone(),
            )
        } else {
            let layout = PublicStateLayout::new(N, self.bind_output, self.total_iters, p, TE);
            layout.assert_state_len(z.len());
            let expected_y = if self.bind_output {
                let y = z[layout.y_range().expect("output range is present")].to_vec();
                for row in 0..IH {
                    for col in 0..IW {
                        if !is_real_input_coord(row, col, &self.real_shape) {
                            let idx = row * IW + col;
                            enforce_zero(
                                &mut cs.namespace(|| {
                                    format!("conv_expected_y_padding_row_{row}_col_{col}_idx_{idx}")
                                }),
                                &y[idx],
                                || format!("conv_expected_y_padding_row_{row}_col_{col}_idx_{idx}"),
                            )?;
                        }
                    }
                }
                Some(y)
            } else {
                None
            };
            (
                layout.state_len(),
                layout.params_base(),
                layout.time_table_range(),
                z[layout.x_range()].to_vec(),
                expected_y,
                None,
                None,
                z[layout.t_index()].clone(),
            )
        };
        let table_flat_vars = &z[table_range.clone()];

        for local_step in 0..self.num_iters_per_step {
            for row in 0..IH {
                for col in 0..IW {
                    let j = row * IW + col;
                    let value = &x_i[j];
                    enforce_signed_range_bits(
                        &mut cs.namespace(|| {
                            format!("conv_x_range_local_{local_step}_row_{row}_col_{col}")
                        }),
                        value,
                        self.seq[local_step].x_i_int[j],
                        self.config.value_min,
                        self.config.value_max,
                        &format!("conv_x_range_local_{local_step}_row_{row}_col_{col}"),
                    )?;
                    if !is_real_input_coord(row, col, &self.real_shape) {
                        enforce_zero(
                            &mut cs.namespace(|| {
                                format!("conv_x_padding_local_{local_step}_row_{row}_col_{col}")
                            }),
                            value,
                            || format!("conv_x_padding_local_{local_step}_row_{row}_col_{col}"),
                        )?;
                    }
                }
            }

            let time_emb = synthesize_time_embedding_lookup(
                &mut cs.namespace(|| format!("conv_time_embedding_lookup_local_{local_step}")),
                &t_var,
                self.seq[local_step].t_int,
                table_flat_vars,
                &self.time_table_values,
                &format!("conv_time_embedding_lookup_local_{local_step}"),
            )?;
            for (j, value) in time_emb.iter().enumerate().skip(self.te_real) {
                enforce_zero(
                    &mut cs.namespace(|| {
                        format!("conv_time_embedding_padding_local_{local_step}_coord_{j}")
                    }),
                    value,
                    || format!("conv_time_embedding_padding_local_{local_step}_coord_{j}"),
                )?;
            }

            let base = params_base + local_step * p;
            if self.commit_params {
                let witnesses = &self.param_hash_witnesses[local_step];
                assert_eq!(
                    witnesses.len(),
                    p,
                    "hash witness length must match parameter block length"
                );
                let h_next = synthesize_toy_hash_block(
                    &mut cs.namespace(|| format!("conv_param_hash_local_{local_step}")),
                    h_var.as_ref().expect("hash accumulator present"),
                    &z[base..base + p],
                    G::Scalar::from(TOY_HASH_BASE_U64),
                    witnesses,
                    &format!("conv_param_hash_local_{local_step}"),
                )?;
                h_var = Some(h_next);
            }

            let kernel_base = base;
            let conv_bias_idx = kernel_base + KH * KW;
            let time_w_base = conv_bias_idx + 1;
            let time_b_idx = time_w_base + TE;
            let alpha_idx = time_b_idx + 1;
            let beta_idx = alpha_idx + 1;

            for ky in 0..KH {
                for kx in 0..KW {
                    if !is_real_kernel_coord(ky, kx, &self.real_shape) {
                        enforce_zero(
                            &mut cs.namespace(|| {
                                format!("conv_kernel_padding_local_{local_step}_ky_{ky}_kx_{kx}")
                            }),
                            &z[kernel_base + ky * KW + kx],
                            || format!("conv_kernel_padding_local_{local_step}_ky_{ky}_kx_{kx}"),
                        )?;
                    }
                }
            }
            for j in self.te_real..TE {
                enforce_zero(
                    &mut cs
                        .namespace(|| format!("conv_time_w_padding_local_{local_step}_coord_{j}")),
                    &z[time_w_base + j],
                    || format!("conv_time_w_padding_local_{local_step}_coord_{j}"),
                )?;
            }

            let mut time_products = Vec::with_capacity(TE);
            for j in 0..TE {
                time_products.push(z[time_w_base + j].mul(
                    cs.namespace(|| format!("conv_time_mul_local_{local_step}_coord_{j}")),
                    &time_emb[j],
                )?);
            }
            let time_raw = AllocatedNum::alloc(
                cs.namespace(|| format!("conv_time_raw_local_{local_step}")),
                || {
                    Ok(crate::clipped_relu::field_from_i64(
                        self.seq[local_step].time_raw_int,
                    ))
                },
            )?;
            cs.enforce(
                || format!("conv_time_raw_check_local_{local_step}"),
                |lc| {
                    time_products
                        .iter()
                        .fold(lc, |acc, product| acc + product.get_variable())
                },
                |lc| lc + CS::one(),
                |lc| lc + time_raw.get_variable(),
            );
            let time_q = enforce_floor_rescale(
                &mut cs.namespace(|| format!("conv_time_rescale_local_{local_step}")),
                &time_raw,
                self.seq[local_step].time_quotient_int,
                self.seq[local_step].time_remainder_int,
                self.config.scale,
                self.config.quotient_min,
                self.config.quotient_max,
                &format!("conv_time_rescale_local_{local_step}"),
            )?;
            let time_bias = time_q.add(
                cs.namespace(|| format!("conv_time_add_bias_local_{local_step}")),
                &z[time_b_idx],
            )?;
            let effective_bias = z[conv_bias_idx].add(
                cs.namespace(|| format!("conv_effective_bias_local_{local_step}")),
                &time_bias,
            )?;

            let mut activation_values = [[0i64; OW]; OH];
            for oy in 0..OH {
                for ox in 0..OW {
                    activation_values[oy][ox] = self.seq[local_step].epsilon_int[oy * OW + ox];
                }
            }
            let epsilon = synthesize_fixed_point_conv2d_clipped_relu_single_channel::<
                CS,
                G::Scalar,
                IH,
                IW,
                KH,
                KW,
                OH,
                OW,
            >(
                cs,
                &x_i,
                &z[kernel_base..kernel_base + KH * KW],
                &effective_bias,
                &self.padding,
                self.config.scale,
                self.config.quotient_min,
                self.config.quotient_max,
                &self.seq[local_step].conv_witness,
                &self.clipped_relu_table,
                &activation_values,
                &format!("conv_layer_local_{local_step}"),
            )?;
            for row in 0..OH {
                for col in 0..OW {
                    if !is_real_output_coord(row, col, &self.real_shape) {
                        let j = row * OW + col;
                        enforce_zero(
                            &mut cs.namespace(|| {
                                format!(
                                    "conv_epsilon_padding_local_{local_step}_row_{row}_col_{col}"
                                )
                            }),
                            &epsilon[j],
                            || {
                                format!(
                                    "conv_epsilon_padding_local_{local_step}_row_{row}_col_{col}"
                                )
                            },
                        )?;
                    }
                }
            }

            let x_next = synthesize_denoise_update::<_, G::Scalar, N>(
                &mut cs.namespace(|| format!("conv_denoise_update_local_{local_step}")),
                &x_i,
                &epsilon,
                &z[alpha_idx],
                &z[beta_idx],
                self.config.scale,
                self.config.quotient_min,
                self.config.quotient_max,
                self.update_mode,
                &self.seq[local_step].update_witness,
                &format!("conv_denoise_update_local_{local_step}"),
            )?;
            for row in 0..IH {
                for col in 0..IW {
                    let j = row * IW + col;
                    enforce_signed_range_bits(
                        &mut cs.namespace(|| {
                            format!("conv_x_next_range_local_{local_step}_coord_{j}")
                        }),
                        &x_next[j],
                        self.seq[local_step].x_i_plus_1_int[j],
                        self.config.value_min,
                        self.config.value_max,
                        &format!("conv_x_next_range_local_{local_step}_coord_{j}"),
                    )?;
                    if !is_real_input_coord(row, col, &self.real_shape) {
                        enforce_zero(
                            &mut cs.namespace(|| {
                                format!("conv_x_next_padding_local_{local_step}_coord_{j}")
                            }),
                            &x_next[j],
                            || format!("conv_x_next_padding_local_{local_step}_coord_{j}"),
                        )?;
                    }
                }
            }

            let t_next = AllocatedNum::alloc(
                cs.namespace(|| format!("conv_t_next_local_{local_step}")),
                || {
                    Ok(crate::clipped_relu::field_from_i64::<G::Scalar>(
                        self.seq[local_step].t_int + 1,
                    ))
                },
            )?;
            cs.enforce(
                || format!("conv_t_increment_local_{local_step}"),
                |lc| lc + t_var.get_variable() + CS::one(),
                |lc| lc + CS::one(),
                |lc| lc + t_next.get_variable(),
            );

            if self.commit_params {
                let is_final = synthesize_is_equal_to_constant(
                    &mut cs.namespace(|| format!("conv_is_final_local_{local_step}")),
                    &t_next,
                    self.seq[local_step].t_int + 1,
                    crate::clipped_relu::field_from_i64::<G::Scalar>(self.total_iters as i64),
                    self.total_iters as i64,
                    &format!("conv_is_final_local_{local_step}"),
                )?;
                enforce_equal_if(
                    &mut cs.namespace(|| format!("conv_final_commitment_eq_local_{local_step}")),
                    &is_final,
                    h_var.as_ref().expect("hash accumulator present"),
                    c_var.as_ref().expect("commitment present"),
                    &format!("conv_final_commitment_eq_local_{local_step}"),
                )?;
                let y = expected_y.as_ref().expect("expected output present");
                for j in 0..N {
                    enforce_equal_if(
                        &mut cs.namespace(|| {
                            format!("conv_final_output_eq_local_{local_step}_coord_{j}")
                        }),
                        &is_final,
                        &x_next[j],
                        &y[j],
                        &format!("conv_final_output_eq_local_{local_step}_coord_{j}"),
                    )?;
                }
            }
            t_var = t_next;
            x_i = x_next;
        }

        let mut z_out = Vec::with_capacity(layout_len);
        z_out.extend(x_i);
        if let Some(y) = expected_y {
            z_out.extend(y);
        }
        if self.commit_params {
            z_out.push(h_var.expect("hash accumulator present"));
            z_out.push(c_var.expect("commitment present"));
        }
        z_out.push(t_var);

        let shift_blocks = self.num_iters_per_step;
        let remaining_blocks = self.total_iters.saturating_sub(shift_blocks);
        for block_idx in 0..remaining_blocks {
            let src_base = params_base + (block_idx + shift_blocks) * p;
            for offset in 0..p {
                z_out.push(z[src_base + offset].clone());
            }
        }
        for pad_block in 0..shift_blocks {
            for offset in 0..p {
                z_out.push(alloc_zero(cs, || {
                    format!("conv_pad_zero_block_{pad_block}_offset_{offset}")
                })?);
            }
        }
        for idx in table_range {
            z_out.push(z[idx].clone());
        }

        Ok(z_out)
    }
}
