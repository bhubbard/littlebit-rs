//! Meta OPT causal language model architecture.

use std::collections::HashMap;

use candle_core::{DType, Device, Tensor, Var};
use rand_chacha::ChaCha8Rng;

use littlebit_core::{BiasMode, LayerShape, LittleBitConfig, TrainMode};

use crate::config::ModelConfig;
use crate::error::Result;
use crate::layer::{
    compute_attention, AttentionParams, Embedding, KvCache, LayerNorm, LinearLayer, LinearLoadSpec,
};
use crate::model::{CausalLmOutput, HiddenStatesMode};
use crate::patch::{patch_linear_layer, PatchPolicy, PatchSummary};

const OPT_POSITION_OFFSET: usize = 2;

#[derive(Debug, Clone)]
struct OptAttention {
    q_proj: LinearLayer,
    k_proj: LinearLayer,
    v_proj: LinearLayer,
    out_proj: LinearLayer,
    num_heads: usize,
    head_dim: usize,
}

impl OptAttention {
    fn forward(
        &self,
        hidden_states: &Tensor,
        layer_idx: usize,
        cache: Option<&mut KvCache>,
    ) -> Result<Tensor> {
        let (batch, seq_len, _) = hidden_states.dims3()?;
        let q = self
            .q_proj
            .forward(hidden_states)?
            .reshape((batch, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = self
            .k_proj
            .forward(hidden_states)?
            .reshape((batch, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = self
            .v_proj
            .forward(hidden_states)?
            .reshape((batch, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        let attn_out = compute_attention(
            &q,
            &k,
            &v,
            layer_idx,
            cache,
            AttentionParams {
                scale: 1.0 / (self.head_dim as f64).sqrt(),
                softcap: None,
                sliding_window: None,
            },
        )?;
        self.out_proj.forward(&attn_out)
    }
}

#[derive(Debug, Clone)]
struct OptDecoderLayer {
    self_attn: OptAttention,
    self_attn_layer_norm: LayerNorm,
    fc1: LinearLayer,
    fc2: LinearLayer,
    final_layer_norm: LayerNorm,
    do_layer_norm_before: bool,
}

impl OptDecoderLayer {
    fn forward(
        &self,
        hidden_states: &Tensor,
        layer_idx: usize,
        cache: Option<&mut KvCache>,
    ) -> Result<Tensor> {
        let residual = hidden_states.clone();
        let x_attn_in = if self.do_layer_norm_before {
            self.self_attn_layer_norm.forward(hidden_states)?
        } else {
            hidden_states.clone()
        };
        let attn = self.self_attn.forward(&x_attn_in, layer_idx, cache)?;
        let mut hidden = residual.add(&attn)?;
        if !self.do_layer_norm_before {
            hidden = self.self_attn_layer_norm.forward(&hidden)?;
        }

        let residual = hidden.clone();
        let x_ff_in = if self.do_layer_norm_before {
            self.final_layer_norm.forward(&hidden)?
        } else {
            hidden
        };
        let ff = self.fc2.forward(&self.fc1.forward(&x_ff_in)?.relu()?)?;
        let mut out = residual.add(&ff)?;
        if !self.do_layer_norm_before {
            out = self.final_layer_norm.forward(&out)?;
        }
        Ok(out)
    }
}

/// Meta OPT causal language model (`model_type = "opt"`).
#[derive(Debug, Clone)]
pub struct OptModel {
    config: ModelConfig,
    embed_tokens: Embedding,
    embed_positions: Embedding,
    project_in: Option<LinearLayer>,
    project_out: Option<LinearLayer>,
    layers: Vec<OptDecoderLayer>,
    final_layer_norm: Option<LayerNorm>,
    lm_head: Option<LinearLayer>,
}

impl OptModel {
    /// Load OPT weights from a Hugging Face or LittleBit state dictionary.
    ///
    /// # Errors
    /// Returns [`crate::ModelError::MissingWeight`] if expected checkpoint keys are missing.
    pub fn from_tensors(
        config: ModelConfig,
        tensors: &HashMap<String, Tensor>,
        spec: LinearLoadSpec<'_>,
    ) -> Result<Self> {
        let embed_dim = config.word_embed_proj_dim.unwrap_or(config.hidden_size);
        let embed_tokens = Embedding::from_tensors(
            "model.decoder.embed_tokens",
            tensors,
            None,
            spec.dtype,
            spec.device,
        )?;
        let embed_positions = Embedding::from_tensors(
            "model.decoder.embed_positions",
            tensors,
            None,
            spec.dtype,
            spec.device,
        )?;

        let unit_spec = LinearLoadSpec {
            ratio_factor: 1.0,
            ..spec
        };
        let kv_factor = spec.littlebit.map_or(1.0, |c| c.kv_factor);
        let kv_spec = LinearLoadSpec {
            ratio_factor: kv_factor,
            ..spec
        };

        let project_in = if embed_dim != config.hidden_size
            || tensors.contains_key("model.decoder.project_in.weight")
            || tensors.contains_key("model.decoder.project_in.U")
            || tensors.contains_key("model.decoder.project_in.U_packed")
        {
            Some(LinearLayer::from_tensors(
                "model.decoder.project_in",
                tensors,
                LayerShape::new(embed_dim, config.hidden_size, BiasMode::NoBias),
                unit_spec,
            )?)
        } else {
            None
        };

        let project_out = if embed_dim != config.hidden_size
            || tensors.contains_key("model.decoder.project_out.weight")
            || tensors.contains_key("model.decoder.project_out.U")
            || tensors.contains_key("model.decoder.project_out.U_packed")
        {
            Some(LinearLayer::from_tensors(
                "model.decoder.project_out",
                tensors,
                LayerShape::new(config.hidden_size, embed_dim, BiasMode::NoBias),
                unit_spec,
            )?)
        } else {
            None
        };

        let bias = BiasMode::from(config.enable_bias.unwrap_or(true));
        let affine = config.layer_norm_elementwise_affine.unwrap_or(true);
        let do_ln_before = config.do_layer_norm_before.unwrap_or(true);
        let ffn_dim = config.effective_intermediate_size();
        let num_heads = config.num_attention_heads;
        let head_dim = config.hidden_size / num_heads.max(1);

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let p = format!("model.decoder.layers.{i}");
            let self_attn = OptAttention {
                q_proj: LinearLayer::from_tensors(
                    &format!("{p}.self_attn.q_proj"),
                    tensors,
                    LayerShape::new(config.hidden_size, config.hidden_size, bias),
                    unit_spec,
                )?,
                k_proj: LinearLayer::from_tensors(
                    &format!("{p}.self_attn.k_proj"),
                    tensors,
                    LayerShape::new(config.hidden_size, config.hidden_size, bias),
                    kv_spec,
                )?,
                v_proj: LinearLayer::from_tensors(
                    &format!("{p}.self_attn.v_proj"),
                    tensors,
                    LayerShape::new(config.hidden_size, config.hidden_size, bias),
                    kv_spec,
                )?,
                out_proj: LinearLayer::from_tensors(
                    &format!("{p}.self_attn.out_proj"),
                    tensors,
                    LayerShape::new(config.hidden_size, config.hidden_size, bias),
                    unit_spec,
                )?,
                num_heads,
                head_dim,
            };
            let self_attn_layer_norm = LayerNorm::from_tensors(
                &format!("{p}.self_attn_layer_norm"),
                tensors,
                config.hidden_size,
                1e-5,
                affine,
                spec.dtype,
                spec.device,
            )?;
            let fc1 = LinearLayer::from_tensors(
                &format!("{p}.fc1"),
                tensors,
                LayerShape::new(config.hidden_size, ffn_dim, bias),
                unit_spec,
            )?;
            let fc2 = LinearLayer::from_tensors(
                &format!("{p}.fc2"),
                tensors,
                LayerShape::new(ffn_dim, config.hidden_size, bias),
                unit_spec,
            )?;
            let final_layer_norm = LayerNorm::from_tensors(
                &format!("{p}.final_layer_norm"),
                tensors,
                config.hidden_size,
                1e-5,
                affine,
                spec.dtype,
                spec.device,
            )?;
            layers.push(OptDecoderLayer {
                self_attn,
                self_attn_layer_norm,
                fc1,
                fc2,
                final_layer_norm,
                do_layer_norm_before: do_ln_before,
            });
        }

        let final_layer_norm = if tensors.contains_key("model.decoder.final_layer_norm.weight") {
            Some(LayerNorm::from_tensors(
                "model.decoder.final_layer_norm",
                tensors,
                config.hidden_size,
                1e-5,
                affine,
                spec.dtype,
                spec.device,
            )?)
        } else {
            None
        };

        let lm_head = if !config.tie_word_embeddings && tensors.contains_key("lm_head.weight") {
            Some(LinearLayer::from_tensors(
                "lm_head",
                tensors,
                LayerShape::new(embed_dim, config.vocab_size, BiasMode::NoBias),
                LinearLoadSpec {
                    littlebit: None,
                    ratio_factor: 1.0,
                    ..spec
                },
            )?)
        } else {
            None
        };

        Ok(Self {
            config,
            embed_tokens,
            embed_positions,
            project_in,
            project_out,
            layers,
            final_layer_norm,
            lm_head,
        })
    }

    /// Initialize a random OPT model for testing.
    ///
    /// # Errors
    /// Returns an error if tensor creation fails.
    pub fn new_random(
        config: ModelConfig,
        dtype: DType,
        device: &Device,
        rng: &mut ChaCha8Rng,
    ) -> Result<Self> {
        let embed_dim = config.word_embed_proj_dim.unwrap_or(config.hidden_size);
        let embed_tokens =
            Embedding::new_random(config.vocab_size, embed_dim, None, dtype, device, rng)?;
        let embed_positions = Embedding::new_random(
            config.max_position_embeddings + OPT_POSITION_OFFSET,
            config.hidden_size,
            None,
            dtype,
            device,
            rng,
        )?;
        let bias = BiasMode::from(config.enable_bias.unwrap_or(true));
        let affine = config.layer_norm_elementwise_affine.unwrap_or(true);
        let do_ln_before = config.do_layer_norm_before.unwrap_or(true);
        let ffn_dim = config.effective_intermediate_size();
        let num_heads = config.num_attention_heads;
        let head_dim = config.hidden_size / num_heads.max(1);

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for _ in 0..config.num_hidden_layers {
            layers.push(OptDecoderLayer {
                self_attn: OptAttention {
                    q_proj: LinearLayer::new_random(
                        LayerShape::new(config.hidden_size, config.hidden_size, bias),
                        dtype,
                        device,
                        rng,
                    )?,
                    k_proj: LinearLayer::new_random(
                        LayerShape::new(config.hidden_size, config.hidden_size, bias),
                        dtype,
                        device,
                        rng,
                    )?,
                    v_proj: LinearLayer::new_random(
                        LayerShape::new(config.hidden_size, config.hidden_size, bias),
                        dtype,
                        device,
                        rng,
                    )?,
                    out_proj: LinearLayer::new_random(
                        LayerShape::new(config.hidden_size, config.hidden_size, bias),
                        dtype,
                        device,
                        rng,
                    )?,
                    num_heads,
                    head_dim,
                },
                self_attn_layer_norm: LayerNorm::new_init(
                    config.hidden_size,
                    1e-5,
                    affine,
                    dtype,
                    device,
                )?,
                fc1: LinearLayer::new_random(
                    LayerShape::new(config.hidden_size, ffn_dim, bias),
                    dtype,
                    device,
                    rng,
                )?,
                fc2: LinearLayer::new_random(
                    LayerShape::new(ffn_dim, config.hidden_size, bias),
                    dtype,
                    device,
                    rng,
                )?,
                final_layer_norm: LayerNorm::new_init(
                    config.hidden_size,
                    1e-5,
                    affine,
                    dtype,
                    device,
                )?,
                do_layer_norm_before: do_ln_before,
            });
        }
        let final_layer_norm = if do_ln_before {
            Some(LayerNorm::new_init(
                config.hidden_size,
                1e-5,
                affine,
                dtype,
                device,
            )?)
        } else {
            None
        };
        let lm_head = if config.tie_word_embeddings {
            None
        } else {
            Some(LinearLayer::new_random(
                LayerShape::new(embed_dim, config.vocab_size, BiasMode::NoBias),
                dtype,
                device,
                rng,
            )?)
        };

        Ok(Self {
            config,
            embed_tokens,
            embed_positions,
            project_in: None,
            project_out: None,
            layers,
            final_layer_norm,
            lm_head,
        })
    }

    /// Forward pass returning logits `(batch, seq_len, vocab_size)` and optional hidden states.
    ///
    /// # Errors
    /// Returns an error if layer forward passes fail.
    pub fn forward(
        &self,
        input_ids: &Tensor,
        mut cache: Option<&mut KvCache>,
        hidden_mode: HiddenStatesMode,
    ) -> Result<CausalLmOutput> {
        let (batch, seq_len) = input_ids.dims2()?;
        let past_len = cache.as_ref().map_or(0, |c| c.past_len(0));

        let mut tok_emb = self.embed_tokens.forward(input_ids)?;
        if let Some(proj_in) = &self.project_in {
            tok_emb = proj_in.forward(&tok_emb)?;
        }

        let pos_ids: Vec<u32> = (0..seq_len)
            .map(|p| (past_len + p + OPT_POSITION_OFFSET) as u32)
            .collect();
        let pos_t = Tensor::from_vec(pos_ids, (1, seq_len), input_ids.device())?
            .broadcast_as((batch, seq_len))?;
        let pos_emb = self.embed_positions.forward(&pos_t)?;
        let mut hidden = tok_emb.add(&pos_emb)?;

        let collect = hidden_mode == HiddenStatesMode::All;
        let mut all_hidden = if collect {
            Vec::with_capacity(self.layers.len() + 1)
        } else {
            Vec::new()
        };

        for (idx, layer) in self.layers.iter().enumerate() {
            if collect {
                all_hidden.push(hidden.clone());
            }
            hidden = layer.forward(&hidden, idx, cache.as_deref_mut())?;
        }

        if let Some(norm) = &self.final_layer_norm {
            hidden = norm.forward(&hidden)?;
        }
        if let Some(proj_out) = &self.project_out {
            hidden = proj_out.forward(&hidden)?;
        }
        if collect {
            all_hidden.push(hidden.clone());
        }

        let logits = match &self.lm_head {
            Some(head) => head.forward(&hidden)?,
            None => self.embed_tokens.forward_lm_head(&hidden)?,
        };
        Ok(CausalLmOutput {
            logits,
            hidden_states: all_hidden,
        })
    }

    pub(crate) fn apply_littlebit_patch(
        &mut self,
        lb_cfg: &LittleBitConfig,
        policy: &PatchPolicy,
        train_mode: TrainMode,
        base_seed: u64,
        summary: &mut PatchSummary,
    ) -> Result<()> {
        if let Some(pin) = &mut self.project_in {
            patch_linear_layer(
                "model.decoder.project_in",
                pin,
                lb_cfg,
                policy,
                train_mode,
                base_seed,
                summary,
            )?;
        }
        if let Some(pout) = &mut self.project_out {
            patch_linear_layer(
                "model.decoder.project_out",
                pout,
                lb_cfg,
                policy,
                train_mode,
                base_seed,
                summary,
            )?;
        }
        for (i, layer) in self.layers.iter_mut().enumerate() {
            let p = format!("model.decoder.layers.{i}");
            patch_linear_layer(
                &format!("{p}.self_attn.q_proj"),
                &mut layer.self_attn.q_proj,
                lb_cfg,
                policy,
                train_mode,
                base_seed,
                summary,
            )?;
            patch_linear_layer(
                &format!("{p}.self_attn.k_proj"),
                &mut layer.self_attn.k_proj,
                lb_cfg,
                policy,
                train_mode,
                base_seed,
                summary,
            )?;
            patch_linear_layer(
                &format!("{p}.self_attn.v_proj"),
                &mut layer.self_attn.v_proj,
                lb_cfg,
                policy,
                train_mode,
                base_seed,
                summary,
            )?;
            patch_linear_layer(
                &format!("{p}.self_attn.out_proj"),
                &mut layer.self_attn.out_proj,
                lb_cfg,
                policy,
                train_mode,
                base_seed,
                summary,
            )?;
            patch_linear_layer(
                &format!("{p}.fc1"),
                &mut layer.fc1,
                lb_cfg,
                policy,
                train_mode,
                base_seed,
                summary,
            )?;
            patch_linear_layer(
                &format!("{p}.fc2"),
                &mut layer.fc2,
                lb_cfg,
                policy,
                train_mode,
                base_seed,
                summary,
            )?;
        }
        if let Some(head) = &mut self.lm_head {
            patch_linear_layer(
                "lm_head", head, lb_cfg, policy, train_mode, base_seed, summary,
            )?;
        }
        Ok(())
    }

    pub(crate) fn convert_to_packed(&mut self) -> Result<()> {
        if let Some(pin) = &mut self.project_in {
            pin.convert_to_packed()?;
        }
        if let Some(pout) = &mut self.project_out {
            pout.convert_to_packed()?;
        }
        for layer in &mut self.layers {
            layer.self_attn.q_proj.convert_to_packed()?;
            layer.self_attn.k_proj.convert_to_packed()?;
            layer.self_attn.v_proj.convert_to_packed()?;
            layer.self_attn.out_proj.convert_to_packed()?;
            layer.fc1.convert_to_packed()?;
            layer.fc2.convert_to_packed()?;
        }
        if let Some(head) = &mut self.lm_head {
            head.convert_to_packed()?;
        }
        Ok(())
    }

    pub(crate) fn make_non_quantized_trainable(&mut self) -> Result<()> {
        self.embed_tokens.make_trainable()?;
        self.embed_positions.make_trainable()?;
        if let Some(pin) = &mut self.project_in {
            pin.make_trainable()?;
        }
        if let Some(pout) = &mut self.project_out {
            pout.make_trainable()?;
        }
        for layer in &mut self.layers {
            layer.self_attn_layer_norm.make_trainable()?;
            layer.final_layer_norm.make_trainable()?;
            layer.self_attn.q_proj.make_trainable()?;
            layer.self_attn.k_proj.make_trainable()?;
            layer.self_attn.v_proj.make_trainable()?;
            layer.self_attn.out_proj.make_trainable()?;
            layer.fc1.make_trainable()?;
            layer.fc2.make_trainable()?;
        }
        if let Some(fln) = &mut self.final_layer_norm {
            fln.make_trainable()?;
        }
        if let Some(head) = &mut self.lm_head {
            head.make_trainable()?;
        }
        Ok(())
    }

    pub(crate) fn freeze_matching(&mut self, exclude: &[&str]) {
        let matches = |name: &str| exclude.iter().any(|k| name.contains(k));
        if matches("model.decoder.embed_tokens") {
            self.embed_tokens.freeze();
        }
        if matches("model.decoder.embed_positions") {
            self.embed_positions.freeze();
        }
        if let Some(fln) = &mut self.final_layer_norm {
            if matches("model.decoder.final_layer_norm") {
                fln.freeze();
            }
        }
        if let Some(head) = &mut self.lm_head {
            if matches("lm_head") {
                head.freeze();
            }
        }
        for (i, layer) in self.layers.iter_mut().enumerate() {
            let p = format!("model.decoder.layers.{i}");
            if matches(&format!("{p}.self_attn_layer_norm")) {
                layer.self_attn_layer_norm.freeze();
            }
            if matches(&format!("{p}.final_layer_norm")) {
                layer.final_layer_norm.freeze();
            }
            if matches(&format!("{p}.self_attn")) {
                layer.self_attn.q_proj.freeze();
                layer.self_attn.k_proj.freeze();
                layer.self_attn.v_proj.freeze();
                layer.self_attn.out_proj.freeze();
            }
            if matches(&format!("{p}.fc1")) {
                layer.fc1.freeze();
            }
            if matches(&format!("{p}.fc2")) {
                layer.fc2.freeze();
            }
        }
    }

    pub(crate) fn trainable_vars(&self) -> Vec<Var> {
        let mut vars = self.embed_tokens.trainable_vars();
        vars.extend(self.embed_positions.trainable_vars());
        if let Some(pin) = &self.project_in {
            vars.extend(pin.trainable_vars());
        }
        if let Some(pout) = &self.project_out {
            vars.extend(pout.trainable_vars());
        }
        for layer in &self.layers {
            vars.extend(layer.self_attn.q_proj.trainable_vars());
            vars.extend(layer.self_attn.k_proj.trainable_vars());
            vars.extend(layer.self_attn.v_proj.trainable_vars());
            vars.extend(layer.self_attn.out_proj.trainable_vars());
            vars.extend(layer.self_attn_layer_norm.trainable_vars());
            vars.extend(layer.fc1.trainable_vars());
            vars.extend(layer.fc2.trainable_vars());
            vars.extend(layer.final_layer_norm.trainable_vars());
        }
        if let Some(fln) = &self.final_layer_norm {
            vars.extend(fln.trainable_vars());
        }
        if let Some(head) = &self.lm_head {
            vars.extend(head.trainable_vars());
        }
        vars
    }

    pub(crate) fn total_params(&self) -> usize {
        let mut count = self.embed_tokens.weight.elem_count()
            + self.embed_positions.weight.elem_count()
            + self.project_in.as_ref().map_or(0, LinearLayer::param_count)
            + self
                .project_out
                .as_ref()
                .map_or(0, LinearLayer::param_count)
            + self
                .final_layer_norm
                .as_ref()
                .map_or(0, LayerNorm::param_count);
        for layer in &self.layers {
            count += layer.self_attn.q_proj.param_count()
                + layer.self_attn.k_proj.param_count()
                + layer.self_attn.v_proj.param_count()
                + layer.self_attn.out_proj.param_count()
                + layer.self_attn_layer_norm.param_count()
                + layer.fc1.param_count()
                + layer.fc2.param_count()
                + layer.final_layer_norm.param_count();
        }
        if let Some(head) = &self.lm_head {
            count += head.param_count();
        }
        count
    }

    pub(crate) fn pack_state_dict(&self) -> Result<HashMap<String, Tensor>> {
        let mut out = HashMap::new();
        self.embed_tokens
            .pack_state_dict("model.decoder.embed_tokens", &mut out);
        self.embed_positions
            .pack_state_dict("model.decoder.embed_positions", &mut out);
        if let Some(pin) = &self.project_in {
            out.extend(pin.pack_state_dict("model.decoder.project_in")?);
        }
        if let Some(pout) = &self.project_out {
            out.extend(pout.pack_state_dict("model.decoder.project_out")?);
        }
        for (i, layer) in self.layers.iter().enumerate() {
            let p = format!("model.decoder.layers.{i}");
            out.extend(
                layer
                    .self_attn
                    .q_proj
                    .pack_state_dict(&format!("{p}.self_attn.q_proj"))?,
            );
            out.extend(
                layer
                    .self_attn
                    .k_proj
                    .pack_state_dict(&format!("{p}.self_attn.k_proj"))?,
            );
            out.extend(
                layer
                    .self_attn
                    .v_proj
                    .pack_state_dict(&format!("{p}.self_attn.v_proj"))?,
            );
            out.extend(
                layer
                    .self_attn
                    .out_proj
                    .pack_state_dict(&format!("{p}.self_attn.out_proj"))?,
            );
            layer
                .self_attn_layer_norm
                .pack_state_dict(&format!("{p}.self_attn_layer_norm"), &mut out);
            out.extend(layer.fc1.pack_state_dict(&format!("{p}.fc1"))?);
            out.extend(layer.fc2.pack_state_dict(&format!("{p}.fc2"))?);
            layer
                .final_layer_norm
                .pack_state_dict(&format!("{p}.final_layer_norm"), &mut out);
        }
        if let Some(fln) = &self.final_layer_norm {
            fln.pack_state_dict("model.decoder.final_layer_norm", &mut out);
        }
        if let Some(head) = &self.lm_head {
            out.extend(head.pack_state_dict("lm_head")?);
        }
        Ok(out)
    }

    pub(crate) fn config(&self) -> &ModelConfig {
        &self.config
    }
}
