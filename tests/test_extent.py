import json
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image
from research import variant
from research.data import GroundingDataset
from research.engine import training_parser, check_arguments
from research.checkpoint import save_checkpoint, load_checkpoint, initialize_weights
from research.model import ResearchGrounder


def args_for(*options):
    return training_parser().parse_args(['--tiny', '--image-size', '28', '--device', 'cpu', *options])


class ExtentTests(unittest.TestCase):
    # Missing location/extent supervision or question conditioning must fail these tests.
    def test_two_tokens_receive_gradients_and_ignore_answers_and_padding(self):
        model = variant.build_model(args_for()).eval()
        batch = {'image': torch.rand(1, 3, 28, 28), 'text': ['Q: x'],
                 'question_text': ['Q: x'], 'mask': torch.zeros(1, 1, 28, 28)}
        output = model(batch)
        self.assertIn('location_logits', output)
        altered = dict(batch, text=['Q: x A: secret'], answer_text=['secret'])
        torch.testing.assert_close(output['logits'], model(altered)['logits'])
        doubled = {'image': batch['image'].repeat(2, 1, 1, 1),
                   'text': ['Q: x', 'Q: a much longer question']}
        torch.testing.assert_close(output['logits'], model(doubled)['logits'][:1], atol=2e-5, rtol=2e-5)
        changed = model(dict(batch, question_text=['Q: y']))['logits']
        self.assertGreater((changed-output['logits']).abs().max().item(), 1e-6)
        loss, metrics = variant.build_objective(args_for(), model, 'cpu')(model, batch)
        loss.backward()
        for grad in model.query_tokens.grad:
            self.assertGreater(grad.abs().sum().item(), 0)
        self.assertTrue(torch.isfinite(loss))
        self.assertIn('area', metrics)

    def test_evidence_width_changes_checkpoint_identity(self):
        narrow = variant.build_model(args_for('--evidence-width', '8'))
        wide = variant.build_model(args_for('--evidence-width', '16'))
        self.assertLess(sum(p.numel() for p in narrow.parameters()), sum(p.numel() for p in wide.parameters()))
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'narrow.pt'
            save_checkpoint(path, narrow, None, None, 1, 0, {})
            with self.assertRaisesRegex(ValueError, 'configuration'):
                load_checkpoint(path, wide)
        with self.assertRaises(ValueError):
            check_arguments(args_for('--evidence-width', '3'))

    def test_paired_loss_matches_equal_original_batch_shards(self):
        from research.extent import ExtentObjective
        torch.manual_seed(9)
        model = variant.build_model(args_for()).eval()
        objective = ExtentObjective(dice_weight=.3, pair_delta_weight=.5, pair_consistency_weight=.7)
        for same in (False, True):
            with self.subTest(same=same):
                masks = torch.zeros(2, 1, 28, 28)
                masks[1] = 1
                paired_masks = torch.zeros_like(masks)
                if not same:
                    paired_masks[0] = 1
                batch = {'image': torch.rand(2, 3, 28, 28), 'mask': masks,
                         'text': ['Q: small?', 'Q: entire region?'],
                         'paired_text': ['Q: where is the evidence?', 'unused'],
                         'paired_mask': paired_masks, 'pair_available': torch.tensor([True, False]),
                         'pair_same': torch.tensor([same, False])}
                combined, metrics = objective(model, batch)
                shards = [{key: value[i:i+1] for key, value in batch.items()} for i in range(2)]
                results = [objective(model, shard) for shard in shards]
                torch.testing.assert_close(combined, (results[0][0]+results[1][0])/2, atol=2e-6, rtol=2e-6)
                for key in metrics:
                    torch.testing.assert_close(metrics[key], (results[0][1][key]+results[1][1][key])/2,
                                               atol=2e-6, rtol=2e-6)

    def test_support_envelope_empty_and_rectangular(self):
        from research.extent import support_envelope, area_loss
        target = torch.zeros(2, 1, 5, 6)
        target[0, 0, 1, 2] = target[0, 0, 3, 4] = 1
        envelope = support_envelope(target)
        expected = torch.zeros_like(target); expected[0, 0, 1:4, 2:5] = 1
        torch.testing.assert_close(envelope, expected)
        logits = torch.zeros_like(target, requires_grad=True)
        empty_loss = area_loss(logits[1:], target[1:])
        self.assertAlmostEqual(empty_loss.item(), .25)
        empty_loss.backward()
        self.assertTrue((logits.grad[1] > 0).all())

    def test_signed_pair_delta_distinguishes_addition_and_removal(self):
        from research.extent import paired_losses
        first = torch.tensor([[[[-10., 10.]]]], requires_grad=True)
        second = -first.detach().clone().requires_grad_()
        mask1 = torch.tensor([[[[0., 1.]]]])
        mask2 = 1-mask1
        correct, same = paired_losses(first, second, mask1, mask2, torch.tensor([False]))
        reversed_loss, _ = paired_losses(second, first, mask1, mask2, torch.tensor([False]))
        self.assertLess(correct.item(), 1e-6)
        self.assertGreater(reversed_loss.item(), 3.9)
        self.assertEqual(same.item(), 0)
        _, same = paired_losses(first, second, mask1, mask1, torch.tensor([True]))
        self.assertGreater(same.item(), .99)
        with self.assertRaisesRegex(ValueError, 'same'):
            paired_losses(first, second, mask1, mask2, torch.tensor([True]))

    def test_rejects_privileged_text_and_unannotated_pair_loss(self):
        for options in [('--text-mode', 'answer'), ('--text-mode', 'dropout'),
                        ('--pair-delta-weight', '1'), ('--pair-consistency-weight', '1'),
                        ('--area-weight', '-1'), ('--support-weight', 'nan')]:
            with self.subTest(options=options), self.assertRaises(ValueError):
                check_arguments(args_for(*options))

    def test_strict_variant_checkpoint_and_controls_initialization(self):
        model = variant.build_model(args_for())
        self.assertTrue(hasattr(model, 'coarse'))
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'control.pt'
            coarse = ResearchGrounder(tiny=True)
            save_checkpoint(path, coarse, None, None, 1, 0, {})
            initialize_weights(path, model)
            torch.testing.assert_close(model.coarse.text_proj.weight, coarse.text_proj.weight)
            with self.assertRaises(ValueError):
                load_checkpoint(path, model)
            save_checkpoint(path, model, None, None, 1, 0, {})
            clone = variant.build_model(args_for()); load_checkpoint(path, clone)
            batch = {'image': torch.rand(1, 3, 28, 28), 'text': ['Q: x']}
            torch.testing.assert_close(model(batch)['logits'], clone(batch)['logits'])

    def test_same_pairs_validate_original_masks_before_resize(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root/'train').mkdir(); (root/'binary_masks_png/train').mkdir(parents=True)
            Image.new('RGB', (40, 40)).save(root/'train/x.jpg')
            Image.new('L', (40, 40)).save(root/'binary_masks_png/train/x.png')
            (root/'train_grounding.json').write_text(json.dumps({'x.jpg': {'question': 'x'}}))
            Image.new('L', (40, 40)).save(root/'a.png')
            changed = Image.new('L', (40, 40)); changed.putpixel((20, 20), 255); changed.save(root/'b.png')
            manifest = root/'pairs.json'
            manifest.write_text(json.dumps([{'filename': 'x.jpg', 'question1': 'x', 'question2': 'y',
                'mask1': 'a.png', 'mask2': 'b.png', 'relation': 'same'}]))
            with self.assertRaisesRegex(ValueError, 'same'):
                GroundingDataset(root, 'train', image_size=2, pairs=manifest)[0]

if __name__ == '__main__':
    unittest.main()
