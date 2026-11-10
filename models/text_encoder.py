from torch import nn
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer


def _open_clip_to_hf_text_model(clip_model) -> CLIPTextModel:
    """Convert an open_clip QuickGELU text tower into a Hugging Face CLIPTextModel."""
    vocab_size, width = clip_model.token_embedding.weight.shape
    context_length = clip_model.positional_embedding.shape[0]
    layers = len(clip_model.transformer.resblocks)
    heads = clip_model.transformer.resblocks[0].attn.num_heads
    mlp_width = clip_model.transformer.resblocks[0].mlp.c_fc.weight.shape[0]
    hf_model = CLIPTextModel(
        CLIPTextConfig(
            vocab_size=vocab_size,
            hidden_size=width,
            intermediate_size=mlp_width,
            num_hidden_layers=layers,
            num_attention_heads=heads,
            max_position_embeddings=context_length,
            hidden_act="quick_gelu",
            bos_token_id=min(49406, vocab_size - 2),
            eos_token_id=min(49407, vocab_size - 1),
        )
    )
    root = "text_model." if "text_model.embeddings.token_embedding.weight" in hf_model.state_dict() else ""
    state = {
        f"{root}embeddings.token_embedding.weight": clip_model.token_embedding.weight.data,
        f"{root}embeddings.position_embedding.weight": clip_model.positional_embedding.data,
        f"{root}final_layer_norm.weight": clip_model.ln_final.weight.data,
        f"{root}final_layer_norm.bias": clip_model.ln_final.bias.data,
    }
    for i, block in enumerate(clip_model.transformer.resblocks):
        prefix = f"{root}encoder.layers.{i}"
        q_w, k_w, v_w = block.attn.in_proj_weight.data.chunk(3, dim=0)
        q_b, k_b, v_b = block.attn.in_proj_bias.data.chunk(3, dim=0)
        state[f"{prefix}.self_attn.q_proj.weight"] = q_w
        state[f"{prefix}.self_attn.q_proj.bias"] = q_b
        state[f"{prefix}.self_attn.k_proj.weight"] = k_w
        state[f"{prefix}.self_attn.k_proj.bias"] = k_b
        state[f"{prefix}.self_attn.v_proj.weight"] = v_w
        state[f"{prefix}.self_attn.v_proj.bias"] = v_b
        state[f"{prefix}.self_attn.out_proj.weight"] = block.attn.out_proj.weight.data
        state[f"{prefix}.self_attn.out_proj.bias"] = block.attn.out_proj.bias.data
        state[f"{prefix}.layer_norm1.weight"] = block.ln_1.weight.data
        state[f"{prefix}.layer_norm1.bias"] = block.ln_1.bias.data
        state[f"{prefix}.mlp.fc1.weight"] = block.mlp.c_fc.weight.data
        state[f"{prefix}.mlp.fc1.bias"] = block.mlp.c_fc.bias.data
        state[f"{prefix}.mlp.fc2.weight"] = block.mlp.c_proj.weight.data
        state[f"{prefix}.mlp.fc2.bias"] = block.mlp.c_proj.bias.data
        state[f"{prefix}.layer_norm2.weight"] = block.ln_2.weight.data
        state[f"{prefix}.layer_norm2.bias"] = block.ln_2.bias.data
    hf_model.load_state_dict(state, strict=True)
    return hf_model


def _load_rn50_text(model_name: str, *, revision: str | None = None) -> CLIPTextModel:
    import open_clip

    if revision is not None:
        url = open_clip.pretrained.get_pretrained_url(model_name, "openai")
        if not url or revision not in url:
            raise ValueError(f"Expected RN50 revision {revision} in pretrained URL {url!r}")
    return _open_clip_to_hf_text_model(open_clip.create_model(model_name, pretrained="openai"))


class TextEncoder(nn.Module):
    """CLIP text transformer: list of strings -> token features and a key-padding mask.

    config: build a randomly initialised text model from this config (offline tests only); the
    tokenizer always comes from tokenizer_name or model_name.
    """

    def __init__(self, model_name="openai/clip-vit-large-patch14-336", config: CLIPTextConfig | None = None,
                 *, revision: str | None = None, tokenizer_name: str | None = None,
                 tokenizer_revision: str | None = None):
        super().__init__()
        options = {"revision": revision} if revision else {}
        if config is not None:
            self.model = CLIPTextModel(config)
        elif model_name == "RN50-quickgelu":
            self.model = _load_rn50_text(model_name, revision=revision)
        else:
            self.model = CLIPTextModel.from_pretrained(model_name, **options)
        tok_options = {"revision": tokenizer_revision} if tokenizer_revision else options
        self.tokenizer = CLIPTokenizer.from_pretrained(tokenizer_name or model_name, **tok_options)
        self.output_dim = self.model.config.hidden_size  # 768 for ViT-L/14-336, 512 for RN50

    def forward(self, texts):
        """texts: list[str] -> (tokens (B, L, D_t), key_padding_mask (B, L), True = padding)."""
        if isinstance(texts, str):
            texts = [texts]
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True,
                                max_length=self.model.config.max_position_embeddings).to(self.model.device)
        outputs = self.model(**inputs)
        return outputs.last_hidden_state, ~inputs["attention_mask"].bool()  # (B, L, D_t), (B, L)
