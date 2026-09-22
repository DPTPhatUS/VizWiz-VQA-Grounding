import unittest
from unittest.mock import patch
import torch
from torch import nn

import models.model as model_module

class TinyImage(nn.Module):
    out_channels = 16
    def forward(self, image):
        x = image.mean(1, keepdim=True).expand(-1, 16, -1, -1)
        x = torch.nn.functional.adaptive_avg_pool2d(x, (2, 2))
        return x, x, x, x

class TinyText(nn.Module):
    output_dim = 8
    def forward(self, texts, return_features=False):
        tokens = torch.arange(32, dtype=torch.float32).view(1, 4, 8).expand(len(texts), -1, -1) / 32
        if not return_features:
            return tokens
        from models.text_encoder import TextFeatures
        valid = torch.tensor([[True, True, True, False]]).expand(len(texts), -1)
        question = torch.tensor([[True, True, False, False]]).expand(len(texts), -1)
        answer = torch.tensor([[False, False, True, False]]).expand(len(texts), -1)
        return TextFeatures(tokens, valid, question, answer)

class TinyDecoder(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.head = nn.Conv2d(in_channels, 1, 1)
    def forward(self, x, a, b, c):
        return self.head(x + a + b + c)

class ModelTests(unittest.TestCase):
    def make_model(self, **kwargs):
        with patch.object(model_module, 'ImageEncoder', TinyImage), patch.object(model_module, 'TextEncoder', TinyText), patch.object(model_module, 'UNetDecoder', TinyDecoder):
            return model_module.GroundingModel(n_heads=2, **kwargs)

    def test_checkpoint_roundtrip_and_finite_backward(self):
        model = self.make_model()
        image = torch.rand(2, 3, 16, 16, requires_grad=True)
        output = model(image, ['Q: color? A: red'] * 2)
        self.assertEqual(output.shape[:2], (2, 1))
        self.assertTrue(torch.isfinite(output).all())
        output.square().mean().backward()
        self.assertTrue(torch.isfinite(image.grad).all())
        self.assertGreater(model.detail_refiner.output.weight.grad.abs().sum().item(), 0)
        clone = self.make_model()
        clone.load_state_dict(model.state_dict(), strict=True)
        model.eval(); clone.eval()
        torch.testing.assert_close(model(image, ['x'] * 2), clone(image, ['x'] * 2))

if __name__ == '__main__':
    unittest.main()
