import unittest
from unittest.mock import patch
import tempfile
import json
from pathlib import Path
import torch
from transformers import CLIPTokenizer, CLIPTextConfig, CLIPTextModel
from models.text_encoder import TextEncoder

class TextTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        vocabulary = {'<|startoftext|>': 0, '<|endoftext|>': 1}
        for char in 'abcdefghijklmnopqrstuvwxyz?:':
            vocabulary[char] = len(vocabulary)
            vocabulary[char + '</w>'] = len(vocabulary)
        (root / 'vocab.json').write_text(json.dumps(vocabulary))
        (root / 'merges.txt').write_text('#version: 0.2\n')
        tokenizer = CLIPTokenizer(vocab_file=str(root / 'vocab.json'), merges_file=str(root / 'merges.txt'), model_max_length=77)
        text_model = CLIPTextModel(CLIPTextConfig(vocab_size=len(vocabulary), hidden_size=8, intermediate_size=16, num_hidden_layers=1, num_attention_heads=2, bos_token_id=0, eos_token_id=1, pad_token_id=1))
        with patch('models.text_encoder.CLIPTokenizer.from_pretrained', return_value=tokenizer), patch('models.text_encoder.CLIPTextModel.from_pretrained', return_value=text_model):
            self.encoder = TextEncoder()

    def tearDown(self):
        self.tmp.cleanup()

    def test_optional_features_preserve_legacy_tensor_and_mask_padding(self):
        import inspect
        self.assertIn('return_features', inspect.signature(self.encoder.forward).parameters, 'structured text features are missing')
        texts = ['Q: red? A: blue', 'Q: x?']
        self.encoder.eval()
        features = self.encoder(texts, return_features=True)
        torch.testing.assert_close(features.tokens, self.encoder(texts))
        self.assertTrue(features.answer_mask[0].any())
        self.assertFalse(features.answer_mask[1].any())
        self.assertFalse((features.question_mask & features.answer_mask).any())
        self.assertFalse((features.question_mask & ~features.valid_mask).any())
        self.assertFalse((features.answer_mask & ~features.valid_mask).any())
        self.assertFalse(features.valid_mask[:, 0].any())
        self.assertEqual(features.answer_mask[0].sum().item(), 4)
        self.assertEqual(features.question_mask[1].sum().item(), 2)

    def test_single_string_matches_single_item_batch(self):
        self.encoder.eval()
        text = 'Q: red? A: blue'
        single = self.encoder(text, return_features=True)
        batch = self.encoder([text], return_features=True)
        for actual, expected in zip(single, batch):
            torch.testing.assert_close(actual, expected)

    def test_empty_and_truncated_answers_remain_finite(self):
        import inspect
        self.assertIn('return_features', inspect.signature(self.encoder.forward).parameters)
        features = self.encoder(['', 'Q: ' + 'red ' * 100 + ' A: blue', 'Q: x? A: '], return_features=True)
        self.assertLessEqual(features.tokens.shape[1], 77)
        self.assertFalse(features.answer_mask.any())
        self.assertTrue(torch.isfinite(features.tokens).all())

if __name__ == '__main__':
    unittest.main()
