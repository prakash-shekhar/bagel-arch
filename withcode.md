## INPUT STAGE

### 1\. Text Conditioning

  - **Tokenization** (modeling/qwen2/tokenization\_qwen2.py): Qwen2 tokenizer with special tokens for multimodal content
    ```python
    # Copyright 2024 The Qwen Team and The HuggingFace Inc. team.
    # SPDX-License-Identifier: Apache-2.0

    """Tokenization classes for Qwen2."""

    import json
    import os
    import unicodedata
    from functools import lru_cache
    from typing import Optional, Tuple
    ```
  - **Text Embedding** (modeling/bagel/bagel.py:150-152): Standard LLM token embeddings via embed\_tokens
    ```python
            self.latent_channel = config.vae_config.z_channels
            self.patch_latent_dim = self.latent_patch_size ** 2 * self.latent_channel
            self.time_embedder = TimestepEmbedder(self.hidden_size)
    ```
  - **Special Tokens** (data/data\_utils.py:27-41): \<|img\_start|\>, \<|img\_end|\>, \<|vid\_start|\>, \<|vid\_end|\> for multimodal sequences
    ```python
    def get_new_tokens(tokenizer):
        all_special_tokens = []
        for k, v in tokenizer.special_tokens_map.items():
            if isinstance(v, str):
                all_special_tokens.append(v)
            elif isinstance(v, list):
                all_special_tokens += v

        new_tokens = []

        if '<|im_start|>' not in all_special_tokens:
            new_tokens.append('<|im_start|>')

        if '<|im_end|>' not in all_special_tokens:
            new_tokens.append('<|im_end|>')
    ```

### 2\. Image Understanding Processing

  - **VIT Encoder** (modeling/siglip/modeling\_siglip.py): SigLIP vision transformer for semantic features
    ```python
    # Copyright 2024 The HuggingFace Inc. team.
    # SPDX-License-Identifier: Apache-2.0

    """PyTorch Siglip model."""

    import math
    import warnings
    from dataclasses import dataclass
    from typing import Any, Optional, Tuple, Union
    ```
  - **Patch Extraction** (data/data\_utils.py:43-50): Converts images to patch tokens via patchify()
    ```python
    def patchify(imgs, patch_size):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x
    ```
  - **Feature Projection** (modeling/bagel/modeling\_utils.py:113-124): MLPconnector maps VIT features to LLM hidden size
    ```python
    class MLPconnector(nn.Module):
        def __init__(self, in_dim: int, out_dim: int, hidden_act: str):
            super().__init__()
            self.activation_fn = ACT2FN[hidden_act]
            self.fc1 = nn.Linear(in_dim, out_dim)
            self.fc2 = nn.Linear(out_dim, out_dim)

        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            hidden_states = self.fc1(hidden_states)
            hidden_states = self.activation_fn(hidden_states)
            hidden_states = self.fc2(hidden_states)
            return hidden_states
    ```
  - **Position Encoding** (modeling/bagel/bagel.py:176-177): 2D sinusoidal embeddings via vit\_pos\_embed
    ```python
            self.connector = MLPconnector(self.vit_hidden_size, self.hidden_size, config.connector_act)
            self.vit_pos_embed = PositionEmbedding(self.vit_max_num_patch_per_side, self.hidden_size)
    ```

### 3\. Image Generation Processing

  - **VAE Encoder** (modeling/autoencoder.py): FLUX VAE encodes images to 16-channel latent space
    ```python
    class Encoder(nn.Module):
        def __init__(self, in_channels: int, downsample: int, ch: int, ch_mult: list[int], num_res_blocks: int, z_channels: int):
            super().__init__()
            self.conv_in = nn.Conv2d(in_channels, ch, kernel_size=3, padding=1)

            downs = []
            now_ch = ch
            for i_block, mult in enumerate(ch_mult):
                out_ch = ch * mult
                for _ in range(num_res_blocks):
                    downs.append(ResBlock(now_ch, out_ch))
                    now_ch = out_ch
                if i_block < len(ch_mult) - 1:
                    downs.append(Downsample(now_ch))
            self.downs = nn.Sequential(*downs)

            self.mid = nn.Sequential(ResBlock(now_ch, now_ch), AttnBlock(now_ch), ResBlock(now_ch, now_ch))
            self.norm_out = nn.GroupNorm(num_groups=32, num_channels=now_ch, eps=1e-6, affine=True)
            self.conv_out = nn.Conv2d(now_ch, 2 * z_channels, kernel_size=3, padding=1)

        def forward(self, x: Tensor) -> Tensor:
            # x = x * 2 - 1 # to -1, 1
            h = self.conv_in(x)
            h = self.downs(h)
            h = self.mid(h)
            h = self.norm_out(h)
            h = swish(h)
            h = self.conv_out(h)
            return h
    ```
  - **Latent Patching** (modeling/bagel/bagel.py:182-187): Converts 2×2 latent patches to sequence tokens
    ```python
        def unpack_latent_patches(self, packed_latent, image_shapes):
            patch_height = image_shapes[0] // self.latent_downsample
            patch_width = image_shapes[1] // self.latent_downsample
            unpacked_latent = packed_latent.reshape(
                packed_latent.shape[0], patch_height, patch_width, self.latent_patch_size, self.latent_patch_size, self.latent_channel
            )
            unpacked_latent = rearrange(unpacked_latent, 'b h w p1 p2 c -> b (h p1) (w p2) c')
            unpacked_latent = unpacked_latent.permute(0, 3, 1, 2).contiguous()
            return unpacked_latent
    ```
  - **Timestep Embedding** (modeling/bagel/modeling\_utils.py:74-110): TimestepEmbedder with sinusoidal encoding + MLP
    ```python
    class TimestepEmbedder(nn.Module):
        """
        Embeds scalar timestep into a vector embedding.
        """
        def __init__(self, hidden_size, frequency_embedding_size=256):
            super().__init__()
            self.mlp = nn.Sequential(
                nn.Linear(frequency_embedding_size, hidden_size, bias=True),
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size, bias=True),
            )
            self.frequency_embedding_size = frequency_embedding_size

        def timestep_embedding(self, t, dim, max_period=10000):
            if max_period is None:
                max_period = 10000
            # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
            half = dim // 2
            freqs = torch.exp(
                -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
            ).to(device=t.device)
            args = t[:, None].float() * freqs[None]
            embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
            if dim % 2:
                embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
            return embedding

        def forward(self, t):
            t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
            t_emb = self.mlp(t_freq)
            return t_emb
    ```
  - **Noise Injection** (modeling/bagel/bagel.py:189-192): Flow matching with configurable timestep\_shift
    ```python
        def apply_flow_matching(self, latent, noise, timesteps):
            return latent * timesteps + noise * (1 - timesteps)

        def get_prediction_target(self, latent, noise, timesteps, model_output, cfg_renorm_type):
            target = noise * timesteps - latent * (1 - timesteps)
            return target
    ```

### 4\. Multimodal Sequence Assembly

  - **Packed Sequences** (modeling/bagel/bagel.py:136-148): Interleaved text, VIT, and VAE tokens
    ```python
    def forward(
        self,
        sequence_length: int,
        packed_text_ids: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        sample_lens: List[int],
        packed_position_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        packed_vit_ids: Optional[torch.LongTensor] = None,
        packed_vit_indexes: Optional[torch.LongTensor] = None,
        packed_vae_ids: Optional[torch.LongTensor] = None,
        packed_vae_indexes: Optional[torch.LongTensor] = None,
        past_key_values: Optional[NaiveCache] = None,
        key_values_lens: Optional[torch.Tensor] = None,
        packed_key_value_indexes: Optional[torch.LongTensor] = None,
        update_past_key_values: bool = True,
        is_causal: bool = True,
        mode: str = "und",
        packed_und_token_indexes: Optional[torch.LongTensor] = None,
        packed_gen_token_indexes: Optional[torch.LongTensor] = None,
        cfg_image_input: Optional[torch.LongTensor] = None,
        cfg_text_input: Optional[torch.LongTensor] = None,
        cfg_image_mask: Optional[torch.LongTensor] = None,
        cfg_text_mask: Optional[torch.LongTensor] = None,
        cfg_image_indexes: Optional[torch.LongTensor] = None,
        cfg_text_indexes: Optional[torch.LongTensor] = None,
        cfg_text_position_ids: Optional[torch.LongTensor] = None,
        cfg_image_position_ids: Optional[torch.LongTensor] = None,
        cfg_attention_mask: Optional[torch.LongTensor] = None,
        cfg_image_past_key_values: Optional[NaiveCache] = None,
        cfg_text_past_key_values: Optional[NaiveCache] = None,
        cfg_image_key_values_lens: Optional[torch.LongTensor] = None,
        cfg_text_key_values_lens: Optional[torch.LongTensor] = None,
        timesteps: Optional[torch.LongTensor] = None,
        do_sample: bool = True,
        temperature: float = 1.0,
        llm_max_new_tokens: int = 100,
        gen_image_size: Optional[Tuple[int]] = None,
        cfg_text_scale: float = 1.0,
        cfg_img_scale: float = 1.0,
        cfg_interval: int = 3,
        timestep_shift: float = 1.0,
        num_timesteps: int = 20,
        cfg_renorm_min: float = 0.0,
        cfg_renorm_type: str = "none",
        cfg_alpha: float = 0.3,
    ) -> torch.Tensor:
    ```
  - **Index Tracking** (modeling/bagel/bagel.py:198-215): Separate indexes for understanding vs generation losses
    ```python
            packed_und_token_indexes = torch.cat([packed_text_indexes, packed_vit_indexes], dim=0)
            packed_gen_token_indexes = packed_vae_indexes
    ```
  - **Position IDs** (data/data\_utils.py:53-69): Extrapolation/interpolation modes for variable sequence lengths
    ```python
    def get_flattened_position_ids_extrapolate(image_height, image_width, vit_patch_size, vit_max_num_patch_per_side):
        assert image_height % vit_patch_size == 0 and image_width % vit_patch_size == 0
        patch_height = image_height // vit_patch_size
        patch_width = image_width // vit_patch_size

        base_pos_embed = torch.arange(
            vit_max_num_patch_per_side ** 2, dtype=torch.long
        ).reshape(vit_max_num_patch_per_side, vit_max_num_patch_per_side)
        pos_embed = base_pos_embed[:patch_height, :patch_width]
        return pos_embed.flatten()

    def get_flattened_position_ids_interpolate(image_height, image_width, vit_patch_size, vit_max_num_patch_per_side):
        assert image_height % vit_patch_size == 0 and image_width % vit_patch_size == 0
        patch_height = image_height // vit_patch_size
        patch_width = image_width // vit_patch_size

        base_pos_embed = torch.arange(
            vit_max_num_patch_per_side ** 2, dtype=torch.long
        ).reshape(vit_max_num_patch_per_side, vit_max_num_patch_per_side)
    ```

## MIXTURE-OF-TOKENS PROCESSING

### 1\. MoT Architecture (modeling/bagel/qwen2\_navit.py:684-814)

  - **Dual Parameter Sets**: Separate weights for understanding (\_moe\_und) vs generation (\_moe\_gen)
    ```python
    class Qwen2DecoderLayer(nn.Module):
        def __init__(self, config: Qwen2Config, layer_idx: int):
            super().__init__()
            self.hidden_size = config.hidden_size
            self.self_attn = Qwen2Attention(config, layer_idx=layer_idx)
            self.mlp_und = Qwen2MLP(config)
            self.mlp_gen = Qwen2MLP(config)
            self.input_layernorm_und = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.input_layernorm_gen = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.post_attention_layernorm_und = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.post_attention_layernorm_gen = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = False,
            packed_und_token_indexes=None,
            packed_gen_token_indexes=None,
            **kwargs,
        ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
            if packed_und_token_indexes.numel() > 0:
                residual_und = hidden_states.index_select(dim=0, index=packed_und_token_indexes)
                hidden_states_und = self.input_layernorm_und(residual_und)
            else:
                hidden_states_und = None
                residual_und = None

            if packed_gen_token_indexes.numel() > 0:
                residual_gen = hidden_states.index_select(dim=0, index=packed_gen_token_indexes)
                hidden_states_gen = self.input_layernorm_gen(residual_gen)
            else:
                hidden_states_gen = None
                residual_gen = None
    ```
  - **Token-Type Routing** (qwen2\_navit.py:419-426): Different projections based on token modality
    ```python
        # [batch_size, seq_len, hidden_size]
        if packed_und_token_indexes.numel() > 0:
            query_states_und = self.q_proj_moe_und(hidden_states.index_select(dim=0, index=packed_und_token_indexes))
            key_states_und = self.k_proj_moe_und(hidden_states.index_select(dim=0, index=packed_und_token_indexes))
            value_states_und = self.v_proj_moe_und(hidden_states.index_select(dim=0, index=packed_und_token_indexes))
        if packed_gen_token_indexes.numel() > 0:
            query_states_gen = self.q_proj_moe_gen(hidden_states.index_select(dim=0, index=packed_gen_token_indexes))
            key_states_gen = self.k_proj_moe_gen(hidden_states.index_select(dim=0, index=packed_gen_token_indexes))
            value_states_gen = self.v_proj_moe_gen(hidden_states.index_select(dim=0, index=packed_gen_token_indexes))
    ```
  - **Shared Backbone**: Core transformer layers shared across modalities
    ```python
    class Qwen2Model(Qwen2PreTrainedModel):
        def __init__(self, config: Qwen2Config):
            super().__init__(config)
            self.padding_idx = config.pad_token_id
            self.vocab_size = config.vocab_size

            self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
            self.layers = nn.ModuleList(
                [Qwen2DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
            )
            self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

            self.gradient_checkpointing = False
            # Initialize weights and apply final processing
            self.post_init()
    ```

### 2\. MoT Attention (modeling/bagel/qwen2\_navit.py:378-598)

  - **Separate Projections**: q\_proj\_moe\_gen, k\_proj\_moe\_gen for generation tokens
    ```python
    class Qwen2Attention(nn.Module):
        """
        Multi-headed attention from 'Attention Is All You Need' paper.
        """

        def __init__(self, config: Qwen2Config, layer_idx: Optional[int] = None):
            super().__init__()
            self.config = config
            self.layer_idx = layer_idx
            self.hidden_size = config.hidden_size
            self.num_heads = config.num_attention_heads
            self.head_dim = self.hidden_size // self.num_heads
            self.num_key_value_heads = config.num_key_value_heads
            self.num_key_value_groups = self.num_heads // self.num_key_value_heads
            self.max_position_embeddings = config.max_position_embeddings
            self.rope_theta = config.rope_theta
            self.is_causal = True

            self.q_proj_moe_und = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
            self.k_proj_moe_und = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
            self.v_proj_moe_und = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
            self.q_proj_moe_gen = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
            self.k_proj_moe_gen = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
            self.v_proj_moe_gen = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
            self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
            self._init_rope()
    ```
  - **Dual QK Normalization** (qwen2\_navit.py:382-390): Independent normalization per token type
    ```python
            self.q_norm_und = Qwen2RMSNorm(self.num_heads * self.head_dim, eps=config.rms_norm_eps)
            self.k_norm_und = Qwen2RMSNorm(self.num_key_value_heads * self.head_dim, eps=config.rms_norm_eps)
            self.q_norm_gen = Qwen2RMSNorm(self.num_heads * self.head_dim, eps=config.rms_norm_eps)
            self.k_norm_gen = Qwen2RMSNorm(self.num_key_value_heads * self.head_dim, eps=config.rms_norm_eps)
    ```
  - **Flash Attention**: Optimized attention computation via flash\_attn\_func
    ```python
    from flash_attn import flash_attn_varlen_func
    ```

### 3\. MoT MLP (modeling/bagel/qwen2\_navit.py:728-744)

  - **Parameter Selection**: Choose MLP weights based on token indexes
    ```python
            if packed_und_token_indexes.numel() > 0:
                hidden_states_und = self.post_attention_layernorm_und(hidden_states_und)
                hidden_states_und = self.mlp_und(hidden_states_und)
                hidden_states.index_copy_(0, packed_und_token_indexes, residual_und + hidden_states_und)
            if packed_gen_token_indexes.numel() > 0:
                hidden_states_gen = self.post_attention_layernorm_gen(hidden_states_gen)
                hidden_states_gen = self.mlp_gen(hidden_states_gen)
                hidden_states.index_copy_(0, packed_gen_token_indexes, residual_gen + hidden_states_gen)
    ```
  - **Efficient Routing**: Minimal overhead for token-type switching
    ```python
            if packed_und_token_indexes.numel() > 0:
                residual_und = hidden_states.index_select(dim=0, index=packed_und_token_indexes)
                hidden_states_und = self.input_layernorm_und(residual_und)
            else:
                hidden_states_und = None
                residual_und = None

            if packed_gen_token_indexes.numel() > 0:
                residual_gen = hidden_states.index_select(dim=0, index=packed_gen_token_indexes)
                hidden_states_gen = self.input_layernorm_gen(residual_gen)
            else:
                hidden_states_gen = None
                residual_gen = None
    ```
  - **Shared Activations**: Common GELU/SwiGLU activations across paths
    ```python
    class Qwen2MLP(nn.Module):
        def __init__(self, config: Qwen2Config):
            super().__init__()
            self.config = config
            self.hidden_size = config.hidden_size
            self.intermediate_size = config.intermediate_size
            self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
            self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
            self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
            self.act_fn = ACT2FN[config.hidden_act]
    ```

## UNIFIED TRANSFORMER PROCESSING

### 1\. Packed Attention (modeling/bagel/qwen2\_navit.py:233-376)

  - **Variable Sequences**: Handles different sequence lengths in single batch
    ```python
        def _prepare_inputs_for_qwen2_attention(
            self,
            packed_query_sequence: torch.Tensor,
            query_lens: torch.Tensor,
            packed_query_position_ids: torch.Tensor,
            past_key_values: Optional[NaiveCache] = None,
            key_values_lens: Optional[torch.Tensor] = None,
            packed_key_value_indexes: Optional[torch.Tensor] = None,
            update_past_key_values=True,
            is_causal=True,
            packed_und_token_indexes=None,
            packed_gen_token_indexes=None,
        ):
            input_bs = query_lens.shape[0]
            max_query_len = query_lens.max().item()
            if past_key_values is None:
                max_past_len = 0
            else:
                max_past_len = past_key_values.max_len
            cu_seqlens_q = torch.cat(
                [torch.zeros(1, dtype=torch.long, device=query_lens.device), query_lens.cumsum(dim=0)]
            )
            cu_seqlens_kv = torch.cat(
                [torch.zeros(1, dtype=torch.long, device=key_values_lens.device), key_values_lens.cumsum(dim=0)]
            )

            q_starts = cu_seqlens_q[:-1]
            q_ends = cu_seqlens_q[1:]
            kv_starts = cu_seqlens_kv[:-1]
            kv_ends = cu_seqlens_kv[1:]

            # [batch_size, seq_len, hidden_size]
            if packed_und_token_indexes.numel() > 0:
                query_states_und = self.q_proj_moe_und(packed_query_sequence.index_select(dim=0, index=packed_und_token_indexes))
                key_states_und = self.k_proj_moe_und(packed_query_sequence.index_select(dim=0, index=packed_und_token_indexes))
                value_states_und = self.v_proj_moe_und(packed_query_sequence.index_select(dim=0, index=packed_und_token_indexes))
                query_states_und = self.q_norm_und(query_states_und)
                key_states_und = self.k_norm_und(key_states_und)
                query_states_und = query_states_und.view(-1, self.num_heads, self.head_dim)
                key_states_und = key_states_und.view(-1, self.num_key_value_heads, self.head_dim)
                value_states_und = value_states_und.view(-1, self.num_key_value_heads, self.head_dim)
            if packed_gen_token_indexes.numel() > 0:
                query_states_gen = self.q_proj_moe_gen(packed_query_sequence.index_select(dim=0, index=packed_gen_token_indexes))
                key_states_gen = self.k_proj_moe_gen(packed_query_sequence.index_select(dim=0, index=packed_gen_token_indexes))
                value_states_gen = self.v_proj_moe_gen(packed_query_sequence.index_select(dim=0, index=packed_gen_token_indexes))
                query_states_gen = self.q_norm_gen(query_states_gen)
                key_states_gen = self.k_norm_gen(key_states_gen)
                query_states_gen = query_states_gen.view(-1, self.num_heads, self.head_dim)
                key_states_gen = key_states_gen.view(-1, self.num_key_value_heads, self.head_dim)
                value_states_gen = value_states_gen.view(-1, self.num_key_value_heads, self.head_dim)

            if packed_und_token_indexes.numel() > 0:
                cos_und, sin_und = self.rotary_emb_und(value_states_und, position_ids.index_select(dim=0, index=packed_und_token_indexes))
                query_states_und, key_states_und = apply_rotary_pos_emb(query_states_und, key_states_und, cos_und, sin_und)
            else:
                query_states_und, key_states_und, value_states_und = None, None, None
            if packed_gen_token_indexes.numel() > 0:
                cos_gen, sin_gen = self.rotary_emb_gen(value_states_gen, packed_query_position_ids.index_select(dim=0, index=packed_gen_token_indexes))
                query_states_gen, key_states_gen = apply_rotary_pos_emb(query_states_gen, key_states_gen, cos_gen, sin_gen)
            else:
                query_states_gen, key_states_gen, value_states_gen = None, None, None

            if past_key_values is not None:
                if packed_key_value_indexes.numel() > 0:
                    past_key_values.update(key_states_und, value_states_und, packed_key_value_indexes, self.layer_idx, 'und')
                    past_key_values.update(key_states_gen, value_states_gen, packed_key_value_indexes, self.layer_idx, 'gen')
                key_states_und, value_states_und = past_key_values.get_key_value(self.layer_idx, 'und')
                key_states_gen, value_states_gen = past_key_values.get_key_value(self.layer_idx, 'gen')
    ```
  - **Cumulative Lengths**: Efficient attention masking via cu\_seqlens
    ```python
            cu_seqlens_q = torch.cat(
                [torch.zeros(1, dtype=torch.long, device=query_lens.device), query_lens.cumsum(dim=0)]
            )
            cu_seqlens_kv = torch.cat(
                [torch.zeros(1, dtype=torch.long, device=key_values_lens.device), key_values_lens.cumsum(dim=0)]
            )
    ```
  - **Memory Optimization**: Reduces padding overhead for efficiency
    ```python
            input_bs = query_lens.shape[0]
            max_query_len = query_lens.max().item()
            if past_key_values is None:
                max_past_len = 0
            else:
                max_past_len = past_key_values.max_len
    ```

### 2\. Layer Processing (modeling/bagel/qwen2\_navit.py:751-814)

  - **RMSNorm**: Pre-normalization for stability
    ```python
    class Qwen2RMSNorm(nn.Module):
        def __init__(self, hidden_size, eps=1e-6):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.eps = eps

        def forward(self, hidden_states):
            input_dtype = hidden_states.dtype
            hidden_states = hidden_states.to(torch.float32)
            variance = hidden_states.pow(2).mean(-1, keepdim=True)
            hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
            return self.weight.to(input_dtype) * hidden_states.to(input_dtype)
    ```
  - **Residual Connections**: Standard transformer residuals
    ```python
            if packed_und_token_indexes.numel() > 0:
                hidden_states_und = self.post_attention_layernorm_und(hidden_states_und)
                hidden_states_und = self.mlp_und(hidden_states_und)
                hidden_states.index_copy_(0, packed_und_token_indexes, residual_und + hidden_states_und)
            if packed_gen_token_indexes.numel() > 0:
                hidden_states_gen = self.post_attention_layernorm_gen(hidden_states_gen)
                hidden_states_gen = self.mlp_gen(hidden_states_gen)
                hidden_states.index_copy_(0, packed_gen_token_indexes, residual_gen + hidden_states_gen)
    ```
  - **Gradient Checkpointing**: Memory-efficient training via checkpoint\_wrapper
    ```python
    class Qwen2Model(Qwen2PreTrainedModel):
        def __init__(self, config: Qwen2Config):
            super().__init__(config)
            self.padding_idx = config.pad_token_id
            self.vocab_size = config.vocab_size

            self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
            self.layers = nn.ModuleList(
                [Qwen2DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
            )
            self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

            self.gradient_checkpointing = False
            # Initialize weights and apply final processing
            self.post_init()
    ```

### 3\. Position Encoding Integration

  - **RoPE for Text**: Rotary position encoding for language tokens
    ```python
            self.rotary_emb_und = Qwen2RotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
                rope_scaling=config.rope_scaling,
            )
            self.rotary_emb_gen = Qwen2RotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
                rope_scaling=config.rope_scaling,
            )
    ```
  - **2D Embeddings**: Spatial position encoding for image patches
    ```python
    def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
        grid_h = np.arange(grid_size, dtype=np.float32)
        grid_w = np.arange(grid_size, dtype=np.float32)
        grid = np.meshgrid(grid_w, grid_h)  # here w goes first
        grid = np.stack(grid, axis=0)

        grid = grid.reshape([2, 1, grid_size, grid_size])
        pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
        if cls_token and extra_tokens > 0:
            pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
        return pos_embed
    ```
  - **Flexible Positioning** (data/data\_utils.py:53-95): Extrapolation for high-resolution images
    ```python
    def get_flattened_position_ids_extrapolate(image_height, image_width, vit_patch_size, vit_max_num_patch_per_side):
        assert image_height % vit_patch_size == 0 and image_width % vit_patch_size == 0
        patch_height = image_height // vit_patch_size
        patch_width = image_width // vit_patch_size

        base_pos_embed = torch.arange(
            vit_max_num_patch_per_side ** 2, dtype=torch.long
        ).reshape(vit_max_num_patch_per_side, vit_max_num_patch_per_side)
        pos_embed = base_pos_embed[:patch_height, :patch_width]
        return pos_embed.flatten()

    def get_flattened_position_ids_interpolate(image_height, image_width, vit_patch_size, vit_max_num_patch_per_side):
        assert image_height % vit_patch_size == 0 and image_width % vit_patch_size == 0
        patch_height = image_height // vit_patch_size
        patch_width = image_width // vit_patch_size

        base_pos_embed = torch.arange(
            vit_max_num_patch_per_side ** 2, dtype=torch.long
        ).reshape(vit_max_num_patch_per_side, vit_max_num_patch_per_side)

        # Interpolate the position embeddings
        pos_embed = F.interpolate(
            base_pos_embed.float().unsqueeze(0).unsqueeze(0),
            size=(patch_height, patch_width),
            mode='bicubic',
            align_corners=False
        ).squeeze().long()
        return pos_embed.flatten()


    def pil_img2rgb(img):
        return img.convert('RGB')
    ```

## OUTPUT STAGE

### 1\. Visual Language Modeling (modeling/bagel/bagel.py:224-227)

  - **LM Head**: Standard language modeling head for text generation
    ```python
            if self.config.visual_gen:
                if self.config.use_moe:
                    lm_head_weight = self.language_model.lm_head.weight.clone()
                    lm_head_bias = self.language_model.lm_head.bias.clone() if self.language_model.lm_head.bias is not None else None
                else:
                    lm_head_weight = self.language_model.lm_head.weight
                    lm_head_bias = self.language_model.lm_head.bias
                self.language_model.lm_head = nn.Linear(
                    self.hidden_size, self.config.llm_config.vocab_size + self.tokenizer.vocab_size, bias=False
                )
                self.language_model.lm_head.weight.data[:self.config.llm_config.vocab_size, :] = lm_head_weight
                if lm_head_bias is not None:
                    self.language_model.lm_head.bias.data[:self.config.llm_config.vocab_size] = lm_head_bias
    ```
  - **Cross-Entropy Loss**: Token-level prediction loss for understanding tasks
    ```python
            loss = None
            if labels is not None:
                # move labels to correct device to enable model parallelism
                labels = labels.to(logits.device)
                if self.config.problem_type is None:
                    if self.num_labels == 1:
                        self.config.problem_type = "regression"
                    elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                        self.config.problem_type = "single_label_classification"
                    else:
                        self.config.problem_type = "multi_label_classification"

                if self.config.problem_type == "regression":
                    loss_fct = MSELoss()
                    if self.num_labels == 1:
                        loss = loss_fct(logits.squeeze(), labels.squeeze())
                    else:
                        loss = loss_fct(logits, labels)
                elif self.config.problem_type == "single_label_classification":
                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                elif self.config.problem_type == "multi_label_classification":
                    loss_fct = BCEWithLogitsLoss()
                    loss = loss_fct(logits, labels)
    ```
  - **Vocabulary Projection**: Maps hidden states to token logits
    ```python
                self.language_model.lm_head = nn.Linear(
                    self.hidden_size, self.config.llm_config.vocab_size + self.tokenizer.vocab_size, bias=False
                )
                self.language_model.lm_head.weight.data[:self.config.llm_config.vocab_size, :] = lm_head_weight
                if lm_head_bias is not None:
                    self.language_model.lm_head.bias.data[:self.config.llm_config.vocab_size] = lm_head_bias
    ```

### 2\. Image Generation (modeling/bagel/bagel.py:217-221)

  - **VAE Projection** (bagel.py:76): llm2vae linear layer maps hidden dim to latent patches
    ```python
            self.time_embedder = TimestepEmbedder(self.hidden_size)
            self.vae2llm = nn.Linear(self.patch_latent_dim, self.hidden_size)
            self.llm2vae = nn.Linear(self.hidden_size, self.patch_latent_dim)
            self.latent_pos_embed = PositionEmbedding(self.max_latent_size, self.hidden_size)
    ```
  - **Flow Matching Target**: noise - clean\_latent velocity prediction
    ```python
        def get_prediction_target(self, latent, noise, timesteps, model_output, cfg_renorm_type):
            target = noise * timesteps - latent * (1 - timesteps)
            if cfg_renorm_type == "global":
                target_norm = torch.norm(target, dim=-1)
                model_output_norm = torch.norm(model_output, dim=-1)
                model_output = model_output * (target_norm / (model_output_norm + 1e-6)).unsqueeze(-1)
            elif cfg_renorm_type == "channel":
                target_norm = torch.norm(target, dim=0)
                model_output_norm = torch.norm(model_output, dim=0)
                model_output = model_output * (target_norm / (model_output_norm + 1e-6))
            elif cfg_renorm_type == "text_channel":
                target_norm = torch.norm(target[:, :self.tokenizer.vocab_size], dim=0)
                model_output_norm = torch.norm(model_output[:, :self.tokenizer.vocab_size], dim=0)
                model_output[:, :self.tokenizer.vocab_size] = model_output[:, :self.tokenizer.vocab_size] * (target_norm / (model_output_norm + 1e-6))
            return target, model_output
    ```
  - **MSE Loss**: Pixel-level reconstruction loss in latent space
    ```python
            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
    ```

### 3\. Classifier-Free Guidance (modeling/bagel/bagel.py:643-865)

  - **Dual CFG**: Text conditioning (cfg\_text\_scale) + image conditioning (cfg\_image\_scale)
    ```python
    def generate_image(
        self,
        image_shapes: Tuple[int],
        gen_context: dict,
        cfg_text_precontext: Optional[dict] = None,
        cfg_img_precontext: Optional[dict] = None,
        cfg_text_scale: float = 1.0,
        cfg_img_scale: float = 1.0,
        cfg_interval: int = 3,
        timestep_shift: float = 1.0,
        num_timesteps: int = 20,
        cfg_renorm_min: float = 0.0,
        cfg_renorm_type: str = "none",
    ) -> Image.Image:
    ```
  - **CFG Renormalization** (bagel.py:776-794): Multiple renorm strategies (global, channel, text\_channel)
    ```python
            if cfg_renorm_type == "global":
                target_norm = torch.norm(target, dim=-1)
                model_output_norm = torch.norm(model_output, dim=-1)
                model_output = model_output * (target_norm / (model_output_norm + 1e-6)).unsqueeze(-1)
            elif cfg_renorm_type == "channel":
                target_norm = torch.norm(target, dim=0)
                model_output_norm = torch.norm(model_output, dim=0)
                model_output = model_output * (target_norm / (model_output_norm + 1e-6))
            elif cfg_renorm_type == "text_channel":
                target_norm = torch.norm(target[:, :self.tokenizer.vocab_size], dim=0)
                model_output_norm = torch.norm(model_output[:, :self.tokenizer.vocab_size], dim=0)
                model_output[:, :self.tokenizer.vocab_size] = model_output[:, :self.tokenizer.vocab_size] * (target_norm / (model_output_norm + 1e-6))
            return target, model_output
    ```
  - **Timestep Scheduling**: Euler method integration for denoising
    ```python
            for i in tqdm(range(num_timesteps)):
                t = torch.full((latent.shape[0],), i, device=latent.device, dtype=torch.long)
                
                # Predict the noise
                model_output = self.get_model_output(
                    latent=latent, 
                    timesteps=t, 
                    gen_context=gen_context, 
                    cfg_text_context=cfg_text_precontext, 
                    cfg_img_context=cfg_img_precontext,
                    cfg_text_scale=cfg_text_scale,
                    cfg_img_scale=cfg_img_scale,
                    cfg_interval=cfg_interval,
                    cfg_renorm_min=cfg_renorm_min,
                    cfg_renorm_type=cfg_renorm_type,
                )

                target, model_output = self.get_prediction_target(latent, noise, t, model_output, cfg_renorm_type)

                # Euler method
                dt = (1 - timestep_shift) / num_timesteps
                latent = latent + model_output * dt
    ```

### 4\. VAE Decoding

  - **Unpacking** (modeling/bagel/bagel.py:274-282): Reshape token sequence back to spatial latents
    ```python
        def unpack_latent_patches(self, packed_latent, image_shapes):
            patch_height = image_shapes[0] // self.latent_downsample
            patch_width = image_shapes[1] // self.latent_downsample
            unpacked_latent = packed_latent.reshape(
                packed_latent.shape[0], patch_height, patch_width, self.latent_patch_size, self.latent_patch_size, self.latent_channel
            )
            unpacked_latent = rearrange(unpacked_latent, 'b h w p1 p2 c -> b (h p1) (w p2) c')
            unpacked_latent = unpacked_latent.permute(0, 3, 1, 2).contiguous()
            return unpacked_latent
    ```
  - **VAE Decoder** (modeling/autoencoder.py): Latent → RGB image reconstruction
    ```python
    class Decoder(nn.Module):
        def __init__(self, out_channels: int, downsample: int, ch: int, ch_mult: list[int], num_res_blocks: int, z_channels: int):
            super().__init__()
            self.conv_in = nn.Conv2d(z_channels, ch_mult[-1] * ch, kernel_size=3, padding=1)
            self.mid = nn.Sequential(ResBlock(ch_mult[-1] * ch, ch_mult[-1] * ch), AttnBlock(ch_mult[-1] * ch), ResBlock(ch_mult[-1] * ch, ch_mult[-1] * ch))

            ups = []
            now_ch = ch_mult[-1] * ch
            for i_block, mult in reversed(list(enumerate(ch_mult))):
                out_ch = ch * mult
                for _ in range(num_res_blocks):
                    ups.append(ResBlock(now_ch, out_ch))
                    now_ch = out_ch
                if i_block > 0:
                    ups.append(Upsample(now_ch))
            self.ups = nn.Sequential(*ups)

            self.norm_out = nn.GroupNorm(num_groups=32, num_channels=now_ch, eps=1e-6, affine=True)
            self.conv_out = nn.Conv2d(now_ch, out_channels, kernel_size=3, padding=1)

        def forward(self, x: Tensor) -> Tensor:
            h = self.conv_in(x)
            h = self.mid(h)
            h = self.ups(h)
            h = self.norm_out(h)
            h = swish(h)
            h = self.conv_out(h)
            return h


    class AutoEncoder(nn.Module):
        def __init__(self, params: AutoEncoderParams):
            super().__init__()
            self.encoder = Encoder(
                in_channels=params.in_channels,
                downsample=params.downsample,
                ch=params.ch,
                ch_mult=params.ch_mult,
                num_res_blocks=params.num_res_blocks,
                z_channels=params.z_channels,
            )
            self.decoder = Decoder(
                out_channels=params.out_ch,
                downsample=params.downsample,
                ch=params.ch,
                ch_mult=params.ch_mult,
                num_res_blocks=params.num_res_blocks,
                z_channels=params.z_channels,
            )
            self.scale_factor = params.scale_factor
            self.shift_factor = params.shift_factor

        def encode(self, x: Tensor) -> Tensor:
            # x = x * 2 - 1 # to -1, 1
            z = self.encoder(x)
            z = self.scale_factor * (z - self.shift_factor)
            return z

        def decode(self, z: Tensor) -> Tensor:
            z = z / self.scale_factor + self.shift_factor
            return self.decoder(z)

        def forward(self, x: Tensor) -> Tensor:
            return self.decode(self.encode(x))
    ```
  - **Post-Processing**: Clamp and normalize output images
    ```python
            image = (image * 0.5 + 0.5).clamp(0, 1)
            image = Image.fromarray((255 * rearrange(image, 'c h w -> h w c')).cpu().numpy().astype(np.uint8))
            return image
    ```

## INFERENCE PIPELINE

### 1\. Context Management (inferencer.py:31-96)

  - **KV Cache**: NaiveCache for incremental generation
    ```python
    class InterleaveInferencer:
        def __init__(self, model, vae_model, tokenizer, vae_transform, vit_transform, new_token_ids):
            self.model = model
            self.vae_model = vae_model
            self.tokenizer = tokenizer
            self.vae_transform = vae_transform
            self.vit_transform = vit_transform
            self.new_token_ids = new_token_ids
            
        def init_gen_context(self): 
            gen_context = {
                'kv_lens': [0],
                'ropes': [0],
                'past_key_values': NaiveCache(self.model.config.llm_config.num_hidden_layers),
                'cfg_text_kv_lens': [0],
                'cfg_text_ropes': [0],
                'cfg_text_past_key_values': NaiveCache(self.model.config.llm_config.num_hidden_layers),
                'cfg_img_kv_lens': [0],
                'cfg_img_ropes': [0],
                'cfg_img_past_key_values': NaiveCache(self.model.config.llm_config.num_hidden_layers),
            }
            return gen_context

        def init_text_context(self):
            text_context = {
                'kv_lens': [0],
                'ropes': [0],
                'past_key_values': NaiveCache(self.model.config.llm_config.num_hidden_layers),
            }
            return text_context
    ```
  - **Context Updates**: Efficient context extension for long sequences
    ```python
        def _update_context(self, context, new_kv_len, new_rope):
            new_kv_lens = deepcopy(context['kv_lens'])
            new_kv_lens[0] = new_kv_len
            new_ropes = deepcopy(context['ropes'])
            new_ropes[0] = new_rope
            return new_kv_lens, new_ropes
    ```
  - **Memory Management**: Optimized cache handling for large contexts
    ```python
                'past_key_values': NaiveCache(self.model.config.llm_config.num_hidden_layers),
                'cfg_text_kv_lens': [0],
                'cfg_text_ropes': [0],
                'cfg_text_past_key_values': NaiveCache(self.model.config.llm_config.num_hidden_layers),
                'cfg_img_kv_lens': [0],
                'cfg_img_ropes': [0],
                'cfg_img_past_key_values': NaiveCache(self.model.config.llm_config.num_hidden_layers),
    ```

### 2\. Multimodal Generation (inferencer.py:206-283)

  - **Interleaved Processing**: Mixed text/image input handling
    ```python
        def interleave_inference(self, input_list: List[Union[Image.Image, str]], **kargs) -> List[Union[Image.Image, str]]:
            output_list = []
            
            # init context
            gen_context = self.init_gen_context()
            text_context = self.init_text_context()
            cfg_text_context = self.init_text_context()
            cfg_img_context = self.init_text_context()

            cfg_text_scale = kargs.pop('cfg_text_scale', 1.0)
            cfg_img_scale = kargs.pop('cfg_img_scale', 1.0)
            cfg_interval = kargs.pop('cfg_interval', 3)
            timestep_shift = kargs.pop('timestep_shift', 1.0)
            num_timesteps = kargs.pop('num_timesteps', 20)
            cfg_renorm_min = kargs.pop('cfg_renorm_min', 0.0)
            cfg_renorm_type = kargs.pop('cfg_renorm_type', "none")
            
            for input_item in input_list:
                if isinstance(input_item, Image.Image):
                    # understand image
                    text = self.understand_image(
                        image=input_item, 
                        text_context=text_context, 
                        **kargs
                    )
                    output_list.append(text)

                elif isinstance(input_item, str):
                    # generate image
                    gen_text = self.add_text_to_context(
                        text=input_item, 
                        gen_context=gen_context, 
                        cfg_text_context=cfg_text_context, 
                        cfg_img_context=cfg_img_context
                    )
                    output_list.append(gen_text)

                    image_shapes = kargs.pop('image_shapes', (512, 512))
                    img = self.gen_image(
                        image_shapes, 
                        gen_context, 
                        cfg_text_precontext=cfg_text_context, 
                        cfg_img_precontext=cfg_img_context,
                        cfg_text_scale=cfg_text_scale, 
                        cfg_img_scale=cfg_img_scale, 
                        cfg_interval=cfg_interval, 
                        timestep_shift=timestep_shift, 
                        num_timesteps=num_timesteps,
                        cfg_renorm_min=cfg_renorm_min,
                        cfg_renorm_type=cfg_renorm_type,
                    )
                    output_list.append(img)
            return output_list
    ```
  - **Modality Switching**: Dynamic routing between understanding and generation
    ```python
        def __call__(
            self, 
            image: Optional[Image.Image] = None, 
            text: Optional[str] = None, 
            **kargs
        ) -> Dict[str, Any]:
            output_dict = {'image': None, 'text': None}

            if image is None and text is None:
                print('Please provide at least one input: either an image or text.')
                return output_dict

            input_list = []
            if image is not None:
                input_list.append(image)
            if text is not None:
                input_list.append(text)

            output_list = self.interleave_inference(input_list, **kargs)

            for i in output_list:
                if isinstance(i, Image.Image):
                    output_dict['image'] = i
                elif isinstance(i, str):
                    output_dict['text'] = i
            return output_dict
    ```
  - **Chain-of-Thought**: Optional thinking mode with \<think\> tags
    ```python
    VLM_THINK_SYSTEM_PROMPT = '''You should first think about the reasoning process in the mind and then provide the user with the answer. 
    The reasoning process is enclosed within <think> </think> tags, i.e. <think> reasoning process here </think> answer here'''

    GEN_THINK_SYSTEM_PROMPT = '''You should first think about the planning process in the mind and then generate the image. 
    The planning process is enclosed within <think> </think> tags, i.e. <think> planning process here </think> image here'''
    ```

### 3\. Generation Control (inferencer.py:99-169)

  - **Sampling Strategies**: Configurable temperature, top-p, top-k
    ```python
            # decode
            generation_input = self.model.prepare_start_tokens(newlens, new_rope, new_token_ids)
            for k, v in generation_input.items():
                if torch.is_tensor(v):
                    generation_input[k] = v.to(device)
            with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                unpacked_latent = self.model.generate_text(
                    past_key_values=past_key_values,
                    max_length=max_length,
                    do_sample=do_sample,
                    temperature=temperature,
                    end_token_id=new_token_ids['eos_token_id'],
                    pad_token_id=self.tokenizer.pad_token_id,
                    **generation_input,
                )
    ```
  - **Guidance Control**: Dynamic CFG scaling throughout generation
    ```python
                model_output = self.get_model_output(
                    latent=latent, 
                    timesteps=t, 
                    gen_context=gen_context, 
                    cfg_text_context=cfg_text_precontext, 
                    cfg_img_context=cfg_img_precontext,
                    cfg_text_scale=cfg_text_scale,
                    cfg_img_scale=cfg_img_scale,
                    cfg_interval=cfg_interval,
                    cfg_renorm_min=cfg_renorm_min,
                    cfg_renorm_type=cfg_renorm_type,
                )
    ```
  - **Quality Control**: Multiple checkpoints and validation steps
    ```python
    torch._dynamo.config.cache_size_limit = 512
    torch._dynamo.config.accumulated_cache_size_limit = 4096
    # flex_attention = torch.compile(flex_attention) # , dynamic=True, mode='max-autotune'
    flex_attention = torch.compile(flex_attention)
    ```
