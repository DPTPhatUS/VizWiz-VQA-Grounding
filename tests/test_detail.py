import importlib.util
import unittest
import torch

class DetailTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('models.experiment'), 'detail refinement missing')
        from models.experiment import DetailRefiner
        return DetailRefiner()

    def test_identity_initialization_preserves_logits_for_odd_image_sizes(self):
        module = self.module()
        image = torch.randn(2, 3, 35, 29, requires_grad=True)
        logits = torch.randn(2, 1, 32, 32, requires_grad=True)
        refined = module(image, logits)
        torch.testing.assert_close(refined, logits)
        refined.square().mean().backward()
        self.assertGreater(module.output.weight.grad.abs().sum().item(), 0)
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_rgb_receives_gradient_after_output_head_learns(self):
        module = self.module()
        with torch.no_grad():
            module.output.weight.fill_(0.1)
        image = torch.randn(2, 3, 35, 29, requires_grad=True)
        logits = torch.randn(2, 1, 32, 32)
        module(image, logits).square().mean().backward()
        self.assertGreater(image.grad.abs().sum().item(), 0)
        self.assertLess(sum(p.numel() for p in module.parameters()), 200000)
