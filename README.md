## bagel-arch

## INPUT STAGE

### 1. Text Conditioning

- **Tokenization** (`modeling/qwen2/tokenization_qwen2.py`): Qwen2 tokenizer with special tokens for multimodal content  
- **Text Embedding** (`modeling/bagel/bagel.py:150-152`): Standard LLM token embeddings via `embed_tokens`  
- **Special Tokens** (`data/data_utils.py:27-41`): `<|img_start|>`, `<|img_end|>`, `<|vid_start|>`, `<|vid_end|>` for multimodal sequences  

### 2. Image Understanding Processing

- **VIT Encoder** (`modeling/siglip/modeling_siglip.py`): SigLIP vision transformer for semantic features  
- **Patch Extraction** (`data/data_utils.py:43-50`): Converts images to patch tokens via `patchify()`  
- **Feature Projection** (`modeling/bagel/modeling_utils.py:113-124`): `MLPconnector` maps VIT features to LLM hidden size  
- **Position Encoding** (`modeling/bagel/bagel.py:176-177`): 2D sinusoidal embeddings via `vit_pos_embed`  

### 3. Image Generation Processing

- **VAE Encoder** (`modeling/autoencoder.py`): FLUX VAE encodes images to 16-channel latent space  
- **Latent Patching** (`modeling/bagel/bagel.py:182-187`): Converts 2×2 latent patches to sequence tokens  
- **Timestep Embedding** (`modeling/bagel/modeling_utils.py:74-110`): `TimestepEmbedder` with sinusoidal encoding + MLP  
- **Noise Injection** (`modeling/bagel/bagel.py:189-192`): Flow matching with configurable `timestep_shift`  

### 4. Multimodal Sequence Assembly

- **Packed Sequences** (`modeling/bagel/bagel.py:136-148`): Interleaved text, VIT, and VAE tokens  
- **Index Tracking** (`modeling/bagel/bagel.py:198-215`): Separate indexes for understanding vs generation losses  
- **Position IDs** (`data/data_utils.py:53-69`): Extrapolation/interpolation modes for variable sequence lengths  

## MIXTURE-OF-TOKENS PROCESSING

### 1. MoT Architecture (`modeling/bagel/qwen2_navit.py:684-814`)

- **Dual Parameter Sets**: Separate weights for understanding (`_moe_und`) vs generation (`_moe_gen`)  
- **Token-Type Routing** (`qwen2_navit.py:419-426`): Different projections based on token modality  
- **Shared Backbone**: Core transformer layers shared across modalities  

### 2. MoT Attention (`modeling/bagel/qwen2_navit.py:378-598`)

- **Separate Projections**: `q_proj_moe_gen`, `k_proj_moe_gen` for generation tokens  
- **Dual QK Normalization** (`qwen2_navit.py:382-390`): Independent normalization per token type  
- **Flash Attention**: Optimized attention computation via `flash_attn_func`  

### 3. MoT MLP (`modeling/bagel/qwen2_navit.py:728-744`)

- **Parameter Selection**: Choose MLP weights based on token indexes  
- **Efficient Routing**: Minimal overhead for token-type switching  
- **Shared Activations**: Common GELU/SwiGLU activations across paths  

## UNIFIED TRANSFORMER PROCESSING

### 1. Packed Attention (`modeling/bagel/qwen2_navit.py:233-376`)

- **Variable Sequences**: Handles different sequence lengths in single batch  
- **Cumulative Lengths**: Efficient attention masking via `cu_seqlens`  
- **Memory Optimization**: Reduces padding overhead for efficiency  

### 2. Layer Processing (`modeling/bagel/qwen2_navit.py:751-814`)

- **RMSNorm**: Pre-normalization for stability  
- **Residual Connections**: Standard transformer residuals  
- **Gradient Checkpointing**: Memory-efficient training via `checkpoint_wrapper`  

### 3. Position Encoding Integration

- **RoPE for Text**: Rotary position encoding for language tokens  
- **2D Embeddings**: Spatial position encoding for image patches  
- **Flexible Positioning** (`data/data_utils.py:53-95`): Extrapolation for high-resolution images  

## OUTPUT STAGE

### 1. Visual Language Modeling (`modeling/bagel/bagel.py:224-227`)

- **LM Head**: Standard language modeling head for text generation  
- **Cross-Entropy Loss**: Token-level prediction loss for understanding tasks  
- **Vocabulary Projection**: Maps hidden states to token logits  

### 2. Image Generation (`modeling/bagel/bagel.py:217-221`)

- **VAE Projection** (`bagel.py:76`): `llm2vae` linear layer maps hidden dim to latent patches  
- **Flow Matching Target**: `noise - clean_latent` velocity prediction  
- **MSE Loss**: Pixel-level reconstruction loss in latent space  

### 3. Classifier-Free Guidance (`modeling/bagel/bagel.py:643-865`)

- **Dual CFG**: Text conditioning (`cfg_text_scale`) + image conditioning (`cfg_image_scale`)  
- **CFG Renormalization** (`bagel.py:776-794`): Multiple renorm strategies (global, channel, text_channel)  
- **Timestep Scheduling**: Euler method integration for denoising  

### 4. VAE Decoding

- **Unpacking** (`modeling/bagel/bagel.py:274-282`): Reshape token sequence back to spatial latents  
- **VAE Decoder** (`modeling/autoencoder.py`): Latent → RGB image reconstruction  
- **Post-Processing**: Clamp and normalize output images  

## INFERENCE PIPELINE

### 1. Context Management (`inferencer.py:31-96`)

- **KV Cache**: `NaiveCache` for incremental generation  
- **Context Updates**: Efficient context extension for long sequences  
- **Memory Management**: Optimized cache handling for large contexts  

### 2. Multimodal Generation (`inferencer.py:206-283`)

- **Interleaved Processing**: Mixed text/image input handling  
- **Modality Switching**: Dynamic routing between understanding and generation  
- **Chain-of-Thought**: Optional thinking mode with `<think>` tags  

### 3. Generation Control (`inferencer.py:99-169`)

- **Sampling Strategies**: Configurable temperature, top-p, top-k  
- **Guidance Control**: Dynamic CFG scaling throughout generation  
- **Quality Control**: Multiple checkpoints and validation steps  
