from collections import defaultdict

from torch import nn
from transformers import (CLIPVisionConfig, CLIPVisionModel, Siglip2ImageProcessor,
                          Siglip2VisionConfig, Siglip2VisionModel)

BASE_NAFLEX = "google/siglip2-base-patch16-naflex"
SO400M_NAFLEX = "google/siglip2-so400m-patch16-naflex"
CLIP_BASELINE = "openai/clip-vit-large-patch14-336"
VISION_MODELS = (BASE_NAFLEX, SO400M_NAFLEX, CLIP_BASELINE)


def make_naflex_processor(config):
    """Raw RGB -> rescale once -> SigLIP normalize once -> NaFlex patches."""
    processor = Siglip2ImageProcessor(**config["image_processor"])
    if (not processor.do_resize or not processor.do_rescale or not processor.do_normalize
            or abs(processor.rescale_factor - 1 / 255) > 1e-12
            or list(processor.image_mean) != [0.5] * 3
            or list(processor.image_std) != [0.5] * 3):
        raise ValueError("Official NaFlex requires rescale=1/255 and RGB mean/std=[0.5,0.5,0.5]")
    return processor


class ImageEncoder(nn.Module):
    def __init__(
        self, model_name=CLIP_BASELINE, pretrained=True, config=None,
        feature_layers=(4, 7, 9), max_num_patches=512, image_processor=None,
    ):
        super().__init__()
        self.is_naflex = (model_name in (BASE_NAFLEX, SO400M_NAFLEX)
                          or (config or {}).get("model_type") == "siglip2_vision_model")
        self.feature_layers = tuple(feature_layers)
        if self.is_naflex:
            if max_num_patches <= 0:
                raise ValueError("max_num_patches must be positive")
            config = (Siglip2VisionConfig.from_pretrained(model_name) if config is None
                      else Siglip2VisionConfig(**config))
            config.vision_use_head = False
            self.vision_encoder = (Siglip2VisionModel.from_pretrained(model_name, config=config)
                                   if pretrained else Siglip2VisionModel(config))
            self.patch_size = config.patch_size
            if image_processor is None:
                image_processor = Siglip2ImageProcessor.from_pretrained(model_name).to_dict()
            self.processor_config = dict(image_processor)
            self.processor_config.update(max_num_patches=max_num_patches, patch_size=self.patch_size)
            self.make_processor()  # reject CLIP statistics or disabled/double rescaling
        elif pretrained:
            self.vision_encoder = CLIPVisionModel.from_pretrained(model_name)
        else:
            config = (CLIPVisionConfig.from_pretrained(model_name) if config is None
                      else CLIPVisionConfig(**config))
            self.vision_encoder = CLIPVisionModel(config)

        self.out_channels = self.vision_encoder.config.hidden_size
        if len(self.feature_layers) != 3 or not all(0 < i <= self.vision_encoder.config.num_hidden_layers for i in self.feature_layers):
            raise ValueError("Exactly three valid hidden-state layer indices are required")

    def make_processor(self):
        if not self.is_naflex:
            raise ValueError("CLIP uses its existing dataset preprocessing")
        return make_naflex_processor({"image_processor": self.processor_config})

    def spatial_groups(self, features, inputs):
        """Unpack valid NaFlex tokens to rectangular CNN inputs, grouping equal grids."""
        groups = defaultdict(list)
        for i, grid in enumerate(inputs["spatial_shapes"].tolist()):
            groups[tuple(grid)].append(i)
        for (h, w), indices in groups.items():
            maps = tuple(tokens[indices, :h * w].reshape(len(indices), h, w, self.out_channels)
                         .permute(0, 3, 1, 2).contiguous() for tokens in features)
            yield indices, maps

    def forward(self, x):
        if self.is_naflex:
            grids = x["spatial_shapes"].tolist()
            padding = x["pixel_attention_mask"]
            for i, (h, w) in enumerate(grids):
                n = h * w
                if h <= 0 or w <= 0 or n > padding.shape[1]:
                    raise ValueError("Invalid NaFlex spatial grid")
                if not bool(padding[i, :n].all()) or bool(padding[i, n:].any()):
                    raise ValueError("NaFlex valid patches must be a contiguous prefix matching spatial_shapes")
            outputs = self.vision_encoder(**x, output_hidden_states=True)
            return tuple(outputs.hidden_states[i] for i in self.feature_layers) + (outputs.last_hidden_state,)
        outputs = self.vision_encoder(x, output_hidden_states=True)
        hidden_states = outputs.hidden_states

        # intermediate layers + final bottleneck
        enc_feat1, enc_feat2, enc_feat3 = (hidden_states[i] for i in self.feature_layers)
        bottleneck = outputs.last_hidden_state  # (B, 1 + N, D_v)

        def reshape_feat(feat):
            B, _, D = feat.shape
            feat = feat[:, 1:, :]  # (B, N, D_v) drop [CLS] token
            num_patches = feat.shape[1]
            grid_size = int(num_patches**0.5)
            if grid_size * grid_size != num_patches:
                raise ValueError(
                    f"Expected square patch grid, got {num_patches} patches"
                )
            return (
                feat.view(B, grid_size, grid_size, D)
                .permute(0, 3, 1, 2)
                .contiguous()
            )  # (B, D_v, H_p, W_p)

        return tuple(map(reshape_feat, [enc_feat1, enc_feat2, enc_feat3])) + (
            reshape_feat(bottleneck),
        )
