use ff::PrimeField;
use nova_snark::{
    frontend::{num::AllocatedNum, ConstraintSystem, SynthesisError},
    traits::{circuit::StepCircuit, Group},
};

use crate::{
    clipped_relu::{clipped_relu_lookup, field_from_i64, ClippedReluLookupTable},
    commitment::{synthesize_toy_hash_block, TOY_HASH_BASE_U64},
    fixed_point::{enforce_floor_rescale, enforce_signed_range_bits, FixedPointConfig},
    models::denoise_update::{synthesize_denoise_update, DenoiseUpdateMode},
    padding::{enforce_zero, enforce_zero_padding_matrix_flat},
    public_state::{
        enforce_equal_if, synthesize_is_equal_to_constant, CommittedDenoiseStateLayout,
        PublicStateLayout,
    },
};

use super::{
    params::FixedDenoiseTimeEmbeddingStepParams, time_embedding::synthesize_time_embedding_lookup,
    trace::FixedDenoiseTimeEmbeddingIteration,
};

#[derive(Clone, Debug)]
pub struct PublicFixedPointDenoiseTimeEmbeddingCircuit<
    G: Group,
    const N: usize,
    const TE: usize,
    const IN: usize,
    const H: usize,
> {
    pub num_iters_per_step: usize,
    pub total_iters: usize,
    pub config: FixedPointConfig,
    pub clipped_relu_table: ClippedReluLookupTable,
    pub time_table_values: Vec<[i64; TE]>,
    pub n_real: usize,
    pub te_real: usize,
    pub in_real: usize,
    pub h_real: usize,
    pub bind_public_output: bool,
    pub commit_params: bool,
    pub update_mode: DenoiseUpdateMode,
    pub param_hash_witnesses: Vec<Vec<G::Scalar>>,
    pub seq: Vec<FixedDenoiseTimeEmbeddingIteration<G::Scalar, N, TE, IN, H>>,
}

fn param_block_len<const N: usize, const TE: usize, const IN: usize, const H: usize>() -> usize {
    FixedDenoiseTimeEmbeddingStepParams::<N, TE, IN, H>::block_len()
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

impl<G: Group, const N: usize, const TE: usize, const IN: usize, const H: usize>
    StepCircuit<G::Scalar> for PublicFixedPointDenoiseTimeEmbeddingCircuit<G, N, TE, IN, H>
{
    fn arity(&self) -> usize {
        let p = param_block_len::<N, TE, IN, H>();
        if self.commit_params {
            CommittedDenoiseStateLayout::new(N, true, self.total_iters, p, TE).state_len()
        } else {
            PublicStateLayout::new(N, self.bind_public_output, self.total_iters, p, TE).state_len()
        }
    }

    fn synthesize<CS: ConstraintSystem<G::Scalar>>(
        &self,
        cs: &mut CS,
        z: &[AllocatedNum<G::Scalar>],
    ) -> Result<Vec<AllocatedNum<G::Scalar>>, SynthesisError> {
        assert_eq!(IN, N + TE, "IN must equal N + TE");
        assert_eq!(
            self.in_real,
            self.n_real + self.te_real,
            "in_real must equal n_real + te_real"
        );
        assert!(self.n_real <= N, "n_real must be <= N");
        assert!(self.te_real <= TE, "te_real must be <= TE");
        assert!(self.in_real <= IN, "in_real must be <= IN");
        assert!(self.h_real <= H, "h_real must be <= H");
        let p = param_block_len::<N, TE, IN, H>();
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
            for (j, value) in y.iter().enumerate().skip(self.n_real) {
                enforce_zero(
                    &mut cs.namespace(|| format!("te_commit_expected_y_padding_coord_{j}")),
                    value,
                    || format!("te_commit_expected_y_padding_coord_{j}"),
                )?;
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
            let layout =
                PublicStateLayout::new(N, self.bind_public_output, self.total_iters, p, TE);
            layout.assert_state_len(z.len());
            let expected_y = if self.bind_public_output {
                let y = z[layout.y_range().expect("output range is present")].to_vec();
                for (j, value) in y.iter().enumerate().skip(self.n_real) {
                    enforce_zero(
                        &mut cs.namespace(|| format!("te_expected_y_padding_coord_{j}")),
                        value,
                        || format!("te_expected_y_padding_coord_{j}"),
                    )?;
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
            for (j, value) in x_i.iter().enumerate() {
                enforce_signed_range_bits(
                    &mut cs.namespace(|| format!("te_x_i_range_local_{local_step}_coord_{j}")),
                    value,
                    self.seq[local_step].x_i_int[j],
                    self.config.value_min,
                    self.config.value_max,
                    &format!("te_x_i_range_local_{local_step}_coord_{j}"),
                )?;
                if j >= self.n_real {
                    enforce_zero(
                        &mut cs
                            .namespace(|| format!("te_x_i_padding_local_{local_step}_coord_{j}")),
                        value,
                        || format!("te_x_i_padding_local_{local_step}_coord_{j}"),
                    )?;
                }
            }

            let time_emb = synthesize_time_embedding_lookup(
                &mut cs.namespace(|| format!("time_embedding_lookup_local_{local_step}")),
                &t_var,
                self.seq[local_step].t_int,
                table_flat_vars,
                &self.time_table_values,
                &format!("time_embedding_lookup_local_{local_step}"),
            )?;
            for (j, value) in time_emb.iter().enumerate().skip(self.te_real) {
                enforce_zero(
                    &mut cs
                        .namespace(|| format!("te_time_emb_padding_local_{local_step}_coord_{j}")),
                    value,
                    || format!("te_time_emb_padding_local_{local_step}_coord_{j}"),
                )?;
            }

            let mut mlp_input = Vec::with_capacity(IN);
            mlp_input.extend(x_i.iter().cloned());
            mlp_input.extend(time_emb);
            assert_eq!(mlp_input.len(), IN);
            for (j, value) in mlp_input.iter().enumerate() {
                let is_padded_x = j < N && j >= self.n_real;
                let is_padded_embedding = j >= N + self.te_real;
                if is_padded_x || is_padded_embedding {
                    enforce_zero(
                        &mut cs.namespace(|| {
                            format!("te_mlp_input_padding_local_{local_step}_coord_{j}")
                        }),
                        value,
                        || format!("te_mlp_input_padding_local_{local_step}_coord_{j}"),
                    )?;
                }
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
                    &mut cs.namespace(|| format!("te_param_hash_local_{local_step}")),
                    h_var.as_ref().expect("hash accumulator present"),
                    &z[base..base + p],
                    G::Scalar::from(TOY_HASH_BASE_U64),
                    witnesses,
                    &format!("te_param_hash_local_{local_step}"),
                )?;
                h_var = Some(h_next);
            }

            let w1_base = base;
            let b1_base = w1_base + H * IN;
            let w2_base = b1_base + H;
            let b2_base = w2_base + N * H;
            let alpha_idx = b2_base + N;
            let beta_idx = alpha_idx + 1;
            for r in 0..H {
                for c in 0..IN {
                    let is_real_x_weight = r < self.h_real && c < self.n_real;
                    let is_real_embedding_weight =
                        r < self.h_real && c >= N && c < N + self.te_real;
                    if !is_real_x_weight && !is_real_embedding_weight {
                        enforce_zero(
                            &mut cs.namespace(|| {
                                format!("te_w1_padding_local_{local_step}_row_{r}_col_{c}")
                            }),
                            &z[w1_base + r * IN + c],
                            || format!("te_w1_padding_local_{local_step}_row_{r}_col_{c}"),
                        )?;
                    }
                }
            }
            for r in self.h_real..H {
                enforce_zero(
                    &mut cs.namespace(|| format!("te_b1_padding_local_{local_step}_row_{r}")),
                    &z[b1_base + r],
                    || format!("te_b1_padding_local_{local_step}_row_{r}"),
                )?;
            }
            enforce_zero_padding_matrix_flat(
                &mut cs.namespace(|| format!("te_w2_padding_local_{local_step}")),
                &z[w2_base..b2_base],
                self.n_real,
                self.h_real,
                N,
                H,
                &format!("te_w2_padding_local_{local_step}"),
            )?;
            for r in self.n_real..N {
                enforce_zero(
                    &mut cs.namespace(|| format!("te_b2_padding_local_{local_step}_row_{r}")),
                    &z[b2_base + r],
                    || format!("te_b2_padding_local_{local_step}_row_{r}"),
                )?;
            }

            let mut hidden_act = Vec::with_capacity(H);
            for r in 0..H {
                let mut acc = z[w1_base + r * IN].mul(
                    cs.namespace(|| format!("te_w1_times_input_local_{local_step}_row_{r}_col_0")),
                    &mlp_input[0],
                )?;
                for c in 1..IN {
                    let product = z[w1_base + r * IN + c].mul(
                        cs.namespace(|| {
                            format!("te_w1_times_input_local_{local_step}_row_{r}_col_{c}")
                        }),
                        &mlp_input[c],
                    )?;
                    acc = acc.add(
                        cs.namespace(|| {
                            format!("te_hidden_raw_acc_local_{local_step}_row_{r}_col_{c}")
                        }),
                        &product,
                    )?;
                }

                let hidden_raw = AllocatedNum::alloc(
                    cs.namespace(|| format!("te_hidden_raw_local_{local_step}_row_{r}")),
                    || Ok(self.seq[local_step].hidden_raw[r]),
                )?;
                cs.enforce(
                    || format!("te_hidden_raw_check_local_{local_step}_row_{r}"),
                    |lc| lc + acc.get_variable(),
                    |lc| lc + CS::one(),
                    |lc| lc + hidden_raw.get_variable(),
                );

                let hidden_q = enforce_floor_rescale(
                    &mut cs.namespace(|| format!("te_hidden_rescale_local_{local_step}_row_{r}")),
                    &hidden_raw,
                    self.seq[local_step].hidden_quotient_int[r],
                    self.seq[local_step].hidden_remainder_int[r],
                    self.config.scale,
                    self.config.quotient_min,
                    self.config.quotient_max,
                    &format!("te_hidden_rescale_local_{local_step}_row_{r}"),
                )?;
                let hidden_affine = hidden_q.add(
                    cs.namespace(|| format!("te_hidden_add_bias_local_{local_step}_row_{r}")),
                    &z[b1_base + r],
                )?;
                let act = clipped_relu_lookup(
                    &mut cs.namespace(|| {
                        format!("te_hidden_clipped_relu_lookup_local_{local_step}_row_{r}")
                    }),
                    &hidden_affine,
                    self.seq[local_step].hidden_affine_int[r],
                    self.seq[local_step].hidden_act[r],
                    &self.clipped_relu_table,
                    &format!("te_hidden_act_local_{local_step}_row_{r}"),
                )?;
                if r >= self.h_real {
                    enforce_zero(
                        &mut cs.namespace(|| {
                            format!("te_hidden_affine_padding_local_{local_step}_row_{r}")
                        }),
                        &hidden_affine,
                        || format!("te_hidden_affine_padding_local_{local_step}_row_{r}"),
                    )?;
                    enforce_zero(
                        &mut cs.namespace(|| {
                            format!("te_hidden_act_padding_local_{local_step}_row_{r}")
                        }),
                        &act,
                        || format!("te_hidden_act_padding_local_{local_step}_row_{r}"),
                    )?;
                }
                hidden_act.push(act);
            }

            let mut epsilon = Vec::with_capacity(N);
            for r in 0..N {
                let mut acc = z[w2_base + r * H].mul(
                    cs.namespace(|| format!("te_w2_times_h_local_{local_step}_row_{r}_col_0")),
                    &hidden_act[0],
                )?;
                for c in 1..H {
                    let product = z[w2_base + r * H + c].mul(
                        cs.namespace(|| {
                            format!("te_w2_times_h_local_{local_step}_row_{r}_col_{c}")
                        }),
                        &hidden_act[c],
                    )?;
                    acc = acc.add(
                        cs.namespace(|| {
                            format!("te_output_raw_acc_local_{local_step}_row_{r}_col_{c}")
                        }),
                        &product,
                    )?;
                }

                let output_raw = AllocatedNum::alloc(
                    cs.namespace(|| format!("te_output_raw_local_{local_step}_row_{r}")),
                    || Ok(self.seq[local_step].output_raw[r]),
                )?;
                cs.enforce(
                    || format!("te_output_raw_check_local_{local_step}_row_{r}"),
                    |lc| lc + acc.get_variable(),
                    |lc| lc + CS::one(),
                    |lc| lc + output_raw.get_variable(),
                );
                let output_q = enforce_floor_rescale(
                    &mut cs.namespace(|| format!("te_output_rescale_local_{local_step}_row_{r}")),
                    &output_raw,
                    self.seq[local_step].output_quotient_int[r],
                    self.seq[local_step].output_remainder_int[r],
                    self.config.scale,
                    self.config.quotient_min,
                    self.config.quotient_max,
                    &format!("te_output_rescale_local_{local_step}_row_{r}"),
                )?;
                let out = output_q.add(
                    cs.namespace(|| format!("te_output_add_bias_local_{local_step}_row_{r}")),
                    &z[b2_base + r],
                )?;
                enforce_signed_range_bits(
                    &mut cs.namespace(|| format!("te_epsilon_range_local_{local_step}_coord_{r}")),
                    &out,
                    self.seq[local_step].epsilon_int[r],
                    self.config.value_min,
                    self.config.value_max,
                    &format!("te_epsilon_range_local_{local_step}_coord_{r}"),
                )?;
                if r >= self.n_real {
                    enforce_zero(
                        &mut cs.namespace(|| {
                            format!("te_epsilon_padding_local_{local_step}_coord_{r}")
                        }),
                        &out,
                        || format!("te_epsilon_padding_local_{local_step}_coord_{r}"),
                    )?;
                }
                epsilon.push(out);
            }

            let x_next = synthesize_denoise_update::<_, G::Scalar, N>(
                &mut cs.namespace(|| format!("te_denoise_update_local_{local_step}")),
                &x_i,
                &epsilon,
                &z[alpha_idx],
                &z[beta_idx],
                self.config.scale,
                self.config.quotient_min,
                self.config.quotient_max,
                self.update_mode,
                &self.seq[local_step].update_witness,
                &format!("te_denoise_update_local_{local_step}"),
            )?;
            for j in 0..N {
                enforce_signed_range_bits(
                    &mut cs.namespace(|| format!("te_x_next_range_local_{local_step}_coord_{j}")),
                    &x_next[j],
                    self.seq[local_step].x_i_plus_1_int[j],
                    self.config.value_min,
                    self.config.value_max,
                    &format!("te_x_next_range_local_{local_step}_coord_{j}"),
                )?;
                if j >= self.n_real {
                    enforce_zero(
                        &mut cs.namespace(|| {
                            format!("te_x_next_padding_local_{local_step}_coord_{j}")
                        }),
                        &x_next[j],
                        || format!("te_x_next_padding_local_{local_step}_coord_{j}"),
                    )?;
                }
            }

            let t_next = AllocatedNum::alloc(
                cs.namespace(|| format!("te_t_next_local_{local_step}")),
                || Ok(field_from_i64::<G::Scalar>(self.seq[local_step].t_int + 1)),
            )?;
            cs.enforce(
                || format!("te_t_increment_local_{local_step}"),
                |lc| lc + t_var.get_variable() + CS::one(),
                |lc| lc + CS::one(),
                |lc| lc + t_next.get_variable(),
            );

            if self.commit_params {
                let is_final = synthesize_is_equal_to_constant(
                    &mut cs.namespace(|| format!("te_is_final_local_{local_step}")),
                    &t_next,
                    self.seq[local_step].t_int + 1,
                    field_from_i64::<G::Scalar>(self.total_iters as i64),
                    self.total_iters as i64,
                    &format!("te_is_final_local_{local_step}"),
                )?;
                enforce_equal_if(
                    &mut cs.namespace(|| format!("te_final_commitment_eq_local_{local_step}")),
                    &is_final,
                    h_var.as_ref().expect("hash accumulator present"),
                    c_var.as_ref().expect("commitment present"),
                    &format!("te_final_commitment_eq_local_{local_step}"),
                )?;
                let y = expected_y.as_ref().expect("expected output present");
                for j in 0..N {
                    enforce_equal_if(
                        &mut cs.namespace(|| {
                            format!("te_final_output_eq_local_{local_step}_coord_{j}")
                        }),
                        &is_final,
                        &x_next[j],
                        &y[j],
                        &format!("te_final_output_eq_local_{local_step}_coord_{j}"),
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
            for t in 0..p {
                z_out.push(z[src_base + t].clone());
            }
        }
        for pad_block in 0..shift_blocks {
            for t in 0..p {
                z_out.push(alloc_zero(cs, || {
                    format!("te_pad_zero_block_{pad_block}_offset_{t}")
                })?);
            }
        }
        for idx in table_range {
            z_out.push(z[idx].clone());
        }

        Ok(z_out)
    }
}
