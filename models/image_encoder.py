"""Fixed SigLIP 2 Base NaFlex vision encoder for this experiment branch."""
from collections import defaultdict

from torch import nn
from transformers import Siglip2ImageProcessor, Siglip2VisionConfig, Siglip2VisionModel

VISION_MODEL = "google/siglip2-base-patch16-naflex"


def make_naflex_processor(config):
    processor = Siglip2ImageProcessor(**config["image_processor"])
    if (not processor.do_resize or not processor.do_rescale or not processor.do_normalize
            or abs(processor.rescale_factor - 1 / 255) > 1e-12
            or list(processor.image_mean) != [0.5] * 3
            or list(processor.image_std) != [0.5] * 3):
        raise ValueError("NaFlex requires rescale=1/255 and RGB mean/std=[0.5,0.5,0.5]")
    return processor


class ImageEncoder(nn.Module):
    def __init__(self, pretrained=True, config=None, feature_layers=(4, 7, 9),
                 max_num_patches=512, image_processor=None):
        super().__init__()
        if max_num_patches <= 0:
            raise ValueError("max_num_patches must be positive")
        config = (Siglip2VisionConfig.from_pretrained(VISION_MODEL) if config is None
                  else Siglip2VisionConfig(**config))
        config.vision_use_head = False
        self.vision_encoder = (Siglip2VisionModel.from_pretrained(VISION_MODEL, config=config)
                               if pretrained else Siglip2VisionModel(config))
        self.out_channels = config.hidden_size
        self.patch_size = config.patch_size
        self.feature_layers = tuple(feature_layers)
        if len(self.feature_layers) != 3 or not all(0 < i <= config.num_hidden_layers for i in self.feature_layers):
            raise ValueError("Exactly three valid feature layers are required")
        if image_processor is None:
            image_processor = Siglip2ImageProcessor.from_pretrained(VISION_MODEL).to_dict()
        self.processor_config = dict(image_processor)
        self.processor_config.update(max_num_patches=max_num_patches, patch_size=self.patch_size)
        self.make_processor()

    def make_processor(self):
        return make_naflex_processor({"image_processor": self.processor_config})

    def forward(self, inputs):
        padding = inputs["pixel_attention_mask"]
        for i, (h, w) in enumerate(inputs["spatial_shapes"].tolist()):
            n = h * w
            if h <= 0 or w <= 0 or n > padding.shape[1]:
                raise ValueError("Invalid NaFlex spatial grid")
            if not bool(padding[i, :n].all()) or bool(padding[i, n:].any()):
                raise ValueError("Valid patches must be a contiguous prefix matching spatial_shapes")
        outputs = self.vision_encoder(**inputs, output_hidden_states=True)
        return tuple(outputs.hidden_states[i] for i in self.feature_layers) + (outputs.last_hidden_state,)

    def spatial_groups(self, features, inputs):
        """Drop padding and group equal rectangular grids for the CNN decoder."""
        groups = defaultdict(list)
        for i, grid in enumerate(inputs["spatial_shapes"].tolist()):
            groups[tuple(grid)].append(i)
        for (h, w), indices in groups.items():
            maps = tuple(tokens[indices, :h * w].reshape(len(indices), h, w, self.out_channels)
                         .permute(0, 3, 1, 2).contiguous() for tokens in features)
            yield indices, maps
