from torch import nn
from transformers import (
    AutoTokenizer,
    CLIPTextConfig,
    CLIPTextModel,
    CLIPTokenizer,
    GemmaTokenizer,
    SiglipTextConfig,
    SiglipTextModel,
)


class TextEncoder(nn.Module):
    """Text transformer (CLIP or SigLIP2): list of strings -> token features and a key-padding mask.

    config: build a randomly initialised text model from this config (offline tests only).
    """

    def __init__(self, model_name="openai/clip-vit-large-patch14-336",
                 config: CLIPTextConfig | SiglipTextConfig | None = None, *, revision: str | None = None):
        super().__init__()
        options = {"revision": revision} if revision else {}
        is_siglip = isinstance(config, SiglipTextConfig) or model_name.startswith("google/siglip")
        self.backbone = "siglip2" if is_siglip else "clip"
        if is_siglip:
            self.model = SiglipTextModel(config) if config is not None else SiglipTextModel.from_pretrained(model_name, **options)
            # SigLIP's pooled projection head is not used by token cross-attention.
            self.model.head.requires_grad_(False)
            self.tokenizer = (GemmaTokenizer(padding_side="right", add_bos_token=False, add_eos_token=True)
                              if config is not None else AutoTokenizer.from_pretrained(model_name, **options))
            if hasattr(self.tokenizer, "padding_side"):
                self.tokenizer.padding_side = "right"
        else:
            self.model = CLIPTextModel(config) if config is not None else CLIPTextModel.from_pretrained(model_name, **options)
            self.tokenizer = CLIPTokenizer.from_pretrained(model_name, **options)
        self.output_dim = self.model.config.hidden_size  # 768 for ViT-L/14-336 and SigLIP2 Base/16-384

    def forward(self, texts):
        """texts: list[str] -> (tokens (B, L, D_t), key_padding_mask (B, L), True = padding)."""
        if isinstance(texts, str):
            texts = [texts]
        if self.backbone == "siglip2":
            inputs = self.tokenizer(texts, return_tensors="pt", padding="max_length", truncation=True,
                                    max_length=self.model.config.max_position_embeddings,
                                    return_attention_mask=True).to(self.model.device)
            outputs = self.model(input_ids=inputs["input_ids"])
            padding = ~inputs["attention_mask"].bool()
            if padding.all(dim=-1).any():
                padding = padding.clone()
                padding[padding.all(dim=-1), 0] = False
            return outputs.last_hidden_state, padding  # (B, L, D_t), (B, L)
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True,
                                max_length=self.model.config.max_position_embeddings).to(self.model.device)
        outputs = self.model(**inputs)
        return outputs.last_hidden_state, ~inputs["attention_mask"].bool()  # (B, L, D_t), (B, L)
