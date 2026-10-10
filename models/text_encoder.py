from torch import nn
from transformers import CLIPTextModel, CLIPTokenizer


class TextEncoder(nn.Module):
    def __init__(self, model_name="openai/clip-vit-large-patch14-336"):
        super().__init__()
        self.model = CLIPTextModel.from_pretrained(model_name)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.output_dim = self.model.config.hidden_size  # 768

    def forward(self, texts, return_mask=False):
        if isinstance(texts, str):
            texts = [texts]
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.model.config.max_position_embeddings,
        ).to(self.model.device)
        outputs = self.model(**inputs)
        if return_mask:
            # True indicates padded token positions to be ignored by MultiheadAttention
            key_padding_mask = ~inputs["attention_mask"].bool()
            return outputs.last_hidden_state, key_padding_mask
        return outputs.last_hidden_state  # (B, L, D)
