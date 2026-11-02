from torch import nn
from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer


class TextEncoder(nn.Module):
    """CLIP text transformer: list of strings -> token features and a key-padding mask.

    config: build a randomly initialised text model from this config (offline tests only); the
    tokenizer always comes from model_name.
    """

    def __init__(self, model_name="openai/clip-vit-large-patch14-336", config: CLIPTextConfig | None = None):
        super().__init__()
        self.model = CLIPTextModel(config) if config is not None else CLIPTextModel.from_pretrained(model_name)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.output_dim = self.model.config.hidden_size  # 768 for ViT-L/14-336

    def forward(self, texts):
        """texts: list[str] -> (tokens (B, L, D_t), key_padding_mask (B, L), True = padding)."""
        if isinstance(texts, str):
            texts = [texts]
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True,
                                max_length=self.model.config.max_position_embeddings).to(self.model.device)
        outputs = self.model(**inputs)
        return outputs.last_hidden_state, ~inputs["attention_mask"].bool()  # (B, L, D_t), (B, L)
