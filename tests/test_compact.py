import importlib.util
import unittest
import torch

class CompactTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('models.experiment'), 'compact decoder missing')
        from models.experiment import CompactDecoder
        return CompactDecoder

    def test_all_feature_levels_receive_gradients(self):
        from models.text_encoder import TextFeatures
        decoder = self.module()(in_channels=16, text_dim=8, width=16)
        features = [torch.randn(2, 16, 2, 2, requires_grad=True) for _ in range(4)]
        tokens = torch.randn(2, 4, 8)
        mask = torch.ones(2, 4, dtype=torch.bool)
        text = TextFeatures(tokens, mask, mask, mask)
        output = decoder(*features, text)
        self.assertEqual(output.shape, (2, 1, 32, 32))
        output.square().mean().backward()
        for feature in features:
            self.assertGreater(feature.grad.abs().sum().item(), 0)
        self.assertTrue(torch.isfinite(output).all())

    def test_full_resolution_shape_and_parameter_budget(self):
        from models.text_encoder import TextFeatures
        decoder_type = self.module()
        with torch.device('meta'):
            decoder = decoder_type()
            features = [torch.empty(1, 1024, 24, 24) for _ in range(4)]
            tokens = torch.empty(1, 77, 768)
            mask = torch.ones(1, 77, dtype=torch.bool)
            output = decoder(*features, TextFeatures(tokens, mask, mask, mask))
        self.assertEqual(output.shape, (1, 1, 384, 384))
        self.assertLess(sum(p.numel() for p in decoder.parameters()), 1500000)
