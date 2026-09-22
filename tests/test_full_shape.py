import unittest
from unittest.mock import patch
import torch
from torch import nn
import models.model as model_module
from models.text_encoder import TextFeatures

class ShapeImage(nn.Module):
    out_channels = 1024
    def forward(self, image):
        return tuple(torch.empty(image.shape[0], 1024, 24, 24, device=image.device) for _ in range(4))

class ShapeText(nn.Module):
    output_dim = 768
    def forward(self, texts, return_features=False):
        tokens = torch.empty(len(texts), 77, 768, device='meta')
        if not return_features:
            return tokens
        mask = torch.ones(len(texts), 77, dtype=torch.bool, device='meta')
        return TextFeatures(tokens, mask, mask, mask)

class FullShapeTests(unittest.TestCase):
    def test_production_decoder_channels_and_default_resolution(self):
        with torch.device('meta'), patch.object(model_module, 'ImageEncoder', ShapeImage), patch.object(model_module, 'TextEncoder', ShapeText):
            model = model_module.GroundingModel()
            output = model(torch.empty(1, 3, 336, 336), ['Q: color? A: red'])
        self.assertEqual(output.shape, (1, 1, 384, 384))
        # Same 16x patch-grid output as the original; train/eval resize to masks.
