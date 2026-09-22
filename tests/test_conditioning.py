import importlib.util
import unittest
import torch

class ConditioningTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('models.experiment'), 'conditioning module missing')
        from models import experiment
        return experiment

    def features(self, tokens=None):
        from models.text_encoder import TextFeatures
        tokens = torch.tensor([[[1., 2.], [3., 4.], [999., 999.]]]) if tokens is None else tokens
        return TextFeatures(tokens, torch.tensor([[True, True, False]]), torch.tensor([[True, False, False]]), torch.tensor([[False, True, False]]))

    def test_pooling_ignores_padding_and_handles_empty_masks(self):
        e = self.module()
        features = self.features()
        torch.testing.assert_close(e.masked_mean(features.tokens, features.valid_mask), torch.tensor([[2., 3.]]))
        torch.testing.assert_close(e.masked_mean(features.tokens, torch.zeros_like(features.valid_mask)), torch.zeros(1, 2))

    def test_film_starts_as_identity_and_can_learn_conditioning(self):
        e = self.module()
        film = e.SkipConditioner(2, 4, reduction=3)
        skips = [torch.randn(1, 4, 2, 2, requires_grad=True) for _ in range(3)]
        output = film(skips, self.features())
        for actual, expected in zip(output, skips):
            torch.testing.assert_close(actual, expected)
        sum(x.square().mean() for x in output).backward()
        self.assertGreater(sum(p.grad.abs().sum().item() for p in film.parameters() if p.grad is not None), 0)
        with torch.no_grad():
            film.reduction.weight.fill_(0.1)
            film.reduction.bias.fill_(0.1)
            for head in film.affine:
                head.weight.fill_(0.1)
        changed = film(skips, self.features(self.features().tokens + 1))
        reference = film(skips, self.features())
        self.assertFalse(torch.allclose(changed[0], reference[0]))

    def test_conditioning_parameter_budget(self):
        e = self.module()
        module = e.SkipConditioner(768, 1024)
        self.assertLess(sum(p.numel() for p in module.parameters()), 1000000)
