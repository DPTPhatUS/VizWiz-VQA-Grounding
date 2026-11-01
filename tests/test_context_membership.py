"""Offline contracts for the experimental mechanism and its runnable entrypoints."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from models.model import GroundingModel
from models.connections import ConnectionNetwork, ContextMembership, neighborhood
from models.text_encoder import TextFeatures
from losses import SupervisedObjective, membership_loss
from utils import load_model_weights, model_from_checkpoint, read_checkpoint
import train
import eval as evaluation
from predict_save import predict_image


def make_model(mode):
    torch.manual_seed(42)
    return GroundingModel(tiny=True, connection_mode=mode)


def batch():
    return {'image': torch.rand(2, 3, 32, 32),
            'text': ['Q: Number next to Offset?', 'Q: What color is the dog?'],
            'mask': (torch.rand(2, 1, 32, 32) > .7).float()}


class MechanismTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_modes_start_equal_but_parameters_are_independent(self):
        shared, separate = make_model('shared'), make_model('separate')
        for name, weight in shared.state_dict().items():
            torch.testing.assert_close(weight, separate.state_dict()[name], rtol=0, atol=0)
        graph = separate.decoder.connections
        for left, right in zip(graph.context_network.parameters(), graph.membership_network.parameters()):
            self.assertNotEqual(left.data_ptr(), right.data_ptr())
        inputs = batch()
        torch.testing.assert_close(shared(inputs)['logits'], separate(inputs)['logits'], rtol=0, atol=0)
        extra = sum(p.numel() for p in separate.parameters()) - sum(p.numel() for p in shared.parameters())
        self.assertEqual(extra, sum(p.numel() for p in graph.membership_network.parameters()))

    def test_membership_supervision_reaches_only_its_connection_network(self):
        for mode in ('shared', 'separate'):
            model, inputs = make_model(mode), batch()
            membership_loss(model(inputs), inputs['mask']).backward()
            graph = model.decoder.connections
            context_grads = [p.grad for p in graph.context_network.parameters()]
            if mode == 'shared':
                self.assertTrue(any(g is not None and g.abs().sum() > 0 for g in context_grads))
            else:
                self.assertTrue(all(g is None for g in context_grads))
                self.assertGreater(graph.membership_network.attention.in_proj_weight.grad.abs().sum(), 0)
            self.assertTrue(all(p.grad is None for p in graph.context_head.parameters()))

    def test_segmentation_reaches_context_membership_and_frozen_text_stays_frozen(self):
        for mode in ('shared', 'separate'):
            model = make_model(mode)
            loss, _ = SupervisedObjective(edge_loss_weight=0)(model, batch())
            loss.backward()
            graph = model.decoder.connections
            for module in (graph.context_head, graph.membership_head, graph.context_network):
                self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters()))
            self.assertGreater(graph.update_strength.grad.abs().sum(), 0)
            self.assertTrue(all(not p.requires_grad and p.grad is None for p in model.text_encoder.parameters()))
            self.assertTrue(all(p.grad is not None for p in model.parameters() if p.requires_grad))

    def test_token_padding_is_ignored_but_valid_tokens_change_connections(self):
        network = ConnectionNetwork(8, 12, 16, 8).eval()
        feature = torch.randn(1, 8, 3, 4)
        tokens = torch.randn(1, 5, 12)
        attention = torch.tensor([[True, True, True, False, False]])
        def run(t):
            return network.condition(feature, TextFeatures(t, attention, attention))[0]
        initial = run(tokens)
        padded = tokens.clone(); padded[:, 3:] = 10000
        torch.testing.assert_close(initial, run(padded), rtol=0, atol=0)
        changed = tokens.clone(); changed[:, 1] = torch.randn(12) * 3
        self.assertGreater((initial - run(changed)).abs().max(), 1e-5)

    def test_neighbor_edges_do_not_wrap_across_image_borders(self):
        indices, valid = neighborhood(2, 3, 'cpu')
        self.assertEqual(indices[0, valid[0]].tolist(), [1, 3, 4])
        self.assertEqual(indices[2, valid[2]].tolist(), [1, 4, 5])
        self.assertEqual(valid.sum(-1).tolist(), [3, 5, 3, 3, 5, 3])

    def test_include_exclude_and_null_have_the_intended_effect(self):
        graph = ContextMembership(8, 12, 'shared', width=16)
        feature = torch.randn(1, 8, 4, 6)
        attention = torch.ones(1, 4, dtype=torch.bool)
        text = TextFeatures(torch.randn(1, 4, 12), attention, attention)
        with torch.no_grad():
            graph.membership_head.weight.zero_()
            outputs = []
            for role in range(3):
                graph.membership_head.bias.fill_(-30)
                graph.membership_head.bias[role] = 30
                outputs.append(graph(feature, text))
        unary = outputs[0]['unary_logits']
        self.assertTrue(torch.all(outputs[0]['membership_logits'] > unary))
        self.assertTrue(torch.all(outputs[1]['membership_logits'] < unary))
        torch.testing.assert_close(outputs[2]['membership_logits'], unary)

    def test_directional_edge_labels_and_border_masking(self):
        indices, valid = neighborhood(1, 2, 'cpu')
        # Receiver 0 is foreground but its source is background: null.
        # Receiver 1 is background and its source foreground: exclude.
        logits = torch.zeros(1, 2, 8, 3)
        logits[0, 0, :, 2] = 20
        logits[0, 1, :, 1] = 20
        output = {'membership_logits': torch.zeros(1, 1, 1, 2), 'edge_logits': logits,
                  'edge_indices': indices, 'edge_valid': valid}
        truth = torch.tensor([[[[1., 0.]]]])
        self.assertLess(membership_loss(output, truth).item(), 1e-5)
        self.assertGreater(membership_loss(output, 1-truth).item(), 10)
        changed = logits.clone(); changed[:, ~valid] = torch.randn_like(changed[:, ~valid]) * 100
        torch.testing.assert_close(membership_loss(output, truth), membership_loss(dict(output, edge_logits=changed), truth))

    def test_fractional_occupancy_and_empty_full_targets_are_finite(self):
        model = make_model('shared')
        inputs = batch(); output = model(inputs)
        for truth in (torch.zeros_like(inputs['mask']), torch.ones_like(inputs['mask'])):
            self.assertTrue(torch.isfinite(membership_loss(output, truth)))
        tiny_target = torch.zeros_like(inputs['mask']); tiny_target[:, :, 0, 0] = 1
        self.assertGreater(abs((membership_loss(output, tiny_target) - membership_loss(output, tiny_target*0)).item()), 1e-5)

    def test_checkpoint_mode_roundtrip_and_wrong_mode_rejection(self):
        for mode in ('shared', 'separate'):
            model = make_model(mode)
            saved = {'format_version':1, 'model_state_dict': model.state_dict(), 'experiment_config': model.experiment_config,
                     'run_config': {'tiny': True, 'image_size': 32}}
            restored = model_from_checkpoint(saved)
            inputs = batch()
            torch.testing.assert_close(model(inputs)['logits'], restored(inputs)['logits'])
            wrong = make_model('separate' if mode == 'shared' else 'shared')
            with self.assertRaisesRegex(ValueError, 'configuration'):
                load_model_weights(wrong, saved)


class PipelineTests(unittest.TestCase):
    def test_train_resume_evaluate_predict_for_both_modes(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for split in ('train', 'val'):
                (root / split).mkdir()
                (root / 'binary_masks_png' / split).mkdir(parents=True)
                records = {}
                for i in range(2):
                    name = f'example{i}.jpg'
                    records[name] = {'question': 'What is the number?'}
                    Image.new('RGB', (29+i, 21+i), (80, 90, 110)).save(root / split / name)
                    mask = np.zeros((21+i, 29+i), dtype=np.uint8); mask[4:12, 7:19] = 255
                    Image.fromarray(mask).save(root / 'binary_masks_png' / split / f'example{i}.png')
                (root / f'{split}_grounding.json').write_text(json.dumps(records))
            for mode in ('shared', 'separate'):
                out = root / mode
                argv = ['--data-root', str(root), '--output-dir', str(out), '--connection-mode', mode,
                        '--num-epochs', '1', '--save-every', '1', '--batch-size', '2', '--num-workers', '0']
                factory = lambda **kw: GroundingModel(**dict({'tiny': True}, **kw))
                with patch.object(train, 'GroundingModel', side_effect=factory), patch.object(train, 'IMAGE_SIZE', 32), \
                     patch.object(torch.cuda, 'is_available', return_value=False), contextlib.redirect_stdout(io.StringIO()):
                    train.train_main(argv)
                    checkpoint = out / 'checkpoint_epoch1.pt'
                    # Omitted mode is recovered from the checkpoint on resume.
                    resumed = [x for x in argv]
                    pos = resumed.index('--connection-mode'); del resumed[pos:pos+2]
                    resumed[resumed.index('--num-epochs')+1] = '2'
                    train.train_main(resumed + ['--resume-checkpoint', str(checkpoint)])
                    uninterrupted = list(argv)
                    uninterrupted[uninterrupted.index('--num-epochs')+1] = '2'
                    uninterrupted[uninterrupted.index('--output-dir')+1] = str(out / 'uninterrupted')
                    train.train_main(uninterrupted)
                saved = read_checkpoint(out / 'checkpoint_epoch2.pt')
                continuous = read_checkpoint(out / 'uninterrupted/checkpoint_epoch2.pt')
                for key, value in saved['model_state_dict'].items():
                    torch.testing.assert_close(value, continuous['model_state_dict'][key], rtol=0, atol=0)
                self.assertEqual(saved['epoch'], 2)
                self.assertEqual(saved['experiment_config']['connection_mode'], mode)
                self.assertEqual(saved['run_config']['edge_loss_weight'], .1)
                self.assertEqual(len((out / 'history.jsonl').read_text().splitlines()), 2)
                result = out / 'validation'
                with contextlib.redirect_stdout(io.StringIO()):
                    evaluation.eval_main(['--checkpoint', str(out / 'checkpoint_epoch2.pt'),
                        '--data-root', str(root), '--dataset', 'val', '--output-dir', str(result),
                        '--device', 'cpu', '--batch-size', '2', '--num-workers', '0'])
                metrics = json.loads((result / 'metrics.json').read_text())
                self.assertEqual(metrics['summary']['num_samples'], 2)
                self.assertEqual(metrics['summary']['experiment_config']['connection_mode'], mode)
                image, prediction = predict_image(out / 'checkpoint_epoch2.pt', root / 'val/example0.jpg',
                                                   'What is the number?', 'cpu')
                self.assertEqual(image.size, prediction.size)
                self.assertEqual(prediction.size, (29, 21))
                self.assertTrue(set(np.unique(np.asarray(prediction))) <= {0, 255})


if __name__ == '__main__':
    unittest.main()
