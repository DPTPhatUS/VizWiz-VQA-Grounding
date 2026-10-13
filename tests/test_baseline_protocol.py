"""Offline protocol checks; no CLIP download or real dataset."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from PIL import Image
import torch
from torch import nn
from dataset import VizWizGroundingDataset, collate_eval_samples
from metrics import compute_iou_per_sample
import eval as evaluator

class FixedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(4.0))
    def forward(self, image, text):
        return self.bias.expand(len(text), 1, 4, 4)

class BaselineProtocolTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / 'val').mkdir()
        (self.root / 'binary_masks_png' / 'val').mkdir(parents=True)
        self.sizes = {'one.jpg': (8, 6), 'two.jpg': (13, 9)}
        records = {}
        for name, size in self.sizes.items():
            Image.new('RGB', size, 'white').save(self.root / 'val' / name)
            Image.new('L', size, 255).save(self.root / 'binary_masks_png' / 'val' / Path(name).with_suffix('.png'))
            records[name] = {'question': 'What color?', 'most_common_answer': 'white'}
        (self.root / 'val_grounding.json').write_text(json.dumps(records))
    def tearDown(self):
        self.tmp.cleanup()
    def dataset(self, **kwargs):
        return VizWizGroundingDataset(self.root / 'val_grounding.json', self.root / 'val',
                                     self.root / 'binary_masks_png' / 'val', is_test=True, **kwargs)
    def test_native_collation_and_question_control(self):
        qa = self.dataset(return_original_mask=True)
        batch = collate_eval_samples([qa[0], qa[1]])
        self.assertEqual(tuple(batch['image'].shape), (2, 3, 336, 336))
        self.assertEqual([tuple(m.shape) for m in batch['original_mask']], [(1, 6, 8), (1, 9, 13)])
        self.assertIn(' A: white', qa[0]['text'])
        self.assertEqual(self.dataset(text_mode='question')[0]['text'], 'Q: What color?')
    def test_default_qa_text_preserves_previous_coercion(self):
        qa = self.dataset()
        qa.data['one.jpg']['most_common_answer'] = None
        self.assertEqual(qa[0]['text'], 'Q: What color? A: None')
        q = self.dataset(text_mode='question')
        q.data['one.jpg']['most_common_answer'] = None
        self.assertEqual(q[0]['text'], 'Q: What color?')
    def test_missing_native_mask_fails(self):
        (self.root / 'binary_masks_png' / 'val' / 'one.png').unlink()
        with self.assertRaises(FileNotFoundError):
            self.dataset(return_original_mask=True)[0]
    def test_empty_union_and_legacy(self):
        empty = torch.zeros(2, 1, 2, 2)
        self.assertTrue(torch.equal(compute_iou_per_sample(empty, empty), torch.ones(2)))
        self.assertTrue(torch.equal(compute_iou_per_sample(empty, empty, legacy=True), torch.zeros(2)))
    def test_native_and_legacy_evaluation_with_raw_and_full_weights(self):
        for full in (False, True):
            checkpoint = self.root / f'weights-{full}.pt'
            state = FixedModel().state_dict()
            torch.save({'model_state_dict': state, 'epoch': 10} if full else state, checkpoint)
            for resolution in ('original', 'resized'):
                out = self.root / f'results-{full}-{resolution}'
                with patch.object(evaluator, 'GroundingModel', FixedModel):
                    evaluator.main(['--checkpoint', str(checkpoint), '--data-root', str(self.root),
                                    '--dataset', 'val', '--device', 'cpu', '--batch-size', '2',
                                    '--num-workers', '0', '--output-dir', str(out),
                                    '--metric-resolution', resolution])
                result = json.loads(next(out.glob('results_*.json')).read_text())
                self.assertEqual(result['summary']['mean_iou'], 1.0)
                self.assertEqual(result['summary']['text_mode'], 'qa')
                for name, size in self.sizes.items():
                    with Image.open(out / Path(name).with_suffix('.png')) as mask:
                        self.assertEqual(mask.size, size if resolution == 'original' else (336, 336))
if __name__ == '__main__':
    unittest.main()