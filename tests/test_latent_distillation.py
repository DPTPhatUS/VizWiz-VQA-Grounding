"""Offline latent-distillation regression checks with tiny encoders."""
import tempfile
from pathlib import Path
import unittest
import torch
from torch.nn import functional as F
from losses import (DistillationObjective, SupervisedObjective, attention_target_distribution,
                    attention_guidance_loss)
from models.model import GroundingModel, TeacherModel
from utils import (LATENT_ANSWER_KEYS, COARSENED_LATENT_ANSWER_KEYS, load_model_weights,
                   initialize_weights)
from train import load_teacher, training_parser, check_arguments

class LatentDistillationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
    def setUp(self):
        torch.manual_seed(42)
        self.model = GroundingModel(tiny=True)
        image = torch.rand(2, 3, 32, 32)
        self.batch = {'image': image, 'mask': (image[:, :1] > .5).float(),
                      'text': ['Q: color?', 'Q: object?'],
                      'question_text': ['Q: color?', 'Q: object?'],
                      'answer_text': ['Q: color? A: red', 'Q: object? A: cup'],
                      'answer_only_text': ['A: red', 'A: cup'],
                      'answer_available': torch.ones(2, dtype=torch.bool)}
    def test_attention_has_defined_gain_mass_and_scale_invariance(self):
        fg = torch.ones(1, 1, 336, 336)
        weights = torch.zeros_like(fg)
        weights[:, :, :, :168] = 1 / (336 * 168)
        dist, valid = attention_target_distribution(weights, fg, (24, 24))
        left_mass = dist.view(1, 24, 24)[:, :, :12].sum()
        torch.testing.assert_close(left_mass, torch.tensor(.875))
        scaled, _ = attention_target_distribution(weights * 1e-5, fg, (24, 24))
        torch.testing.assert_close(scaled, dist)
        self.assertEqual(valid.item(), 1)
        mirrored, _ = attention_target_distribution(weights.flip(-1), fg, (24, 24))
        self.assertGreater((dist - mirrored).abs().sum().item(), 1)
    def test_no_gain_fallback_empty_foreground_and_missing_answers(self):
        fg = torch.ones(1, 1, 32, 32)
        dist, _ = attention_target_distribution(torch.zeros_like(fg), fg, (2, 2))
        torch.testing.assert_close(dist, torch.full((1, 4), .25))
        attn = torch.full((1, 1, 2, 2), .25, requires_grad=True)
        for target, available in [(torch.zeros_like(fg), torch.ones(1)), (fg, torch.zeros(1))]:
            loss = attention_guidance_loss(attn, fg, target, available)
            self.assertEqual(loss.item(), 0)
            self.assertTrue(torch.isfinite(loss))
    def test_question_reference_is_supervised_and_independent_of_answer(self):
        captured = {}
        original_forward = self.model.forward
        def capture(batch):
            output = original_forward(batch)
            captured.update(output)
            output['question_logits'].retain_grad()
            output['teacher_logits'].retain_grad()
            return output
        self.model.forward = capture
        self.model.train()
        loss, metrics = DistillationObjective()(self.model, self.batch)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertGreater(captured['question_logits'].grad.abs().sum().item(), 0)
        self.assertGreater(captured['teacher_logits'].grad.abs().sum().item(), 0)
        self.assertFalse(captured['target_answer'].requires_grad)
        self.assertIn('question_segmentation', metrics)
        self.assertNotIn('active_fraction', metrics)
        q_before = captured['question_logits'].detach().clone()
        self.model.forward = original_forward
        changed = dict(self.batch, answer_only_text=['A: blue', 'A: dog'])
        q_after = self.model(changed)['question_logits']
        torch.testing.assert_close(q_before, q_after, rtol=0, atol=0)
    def test_missing_answer_masks_privileged_losses_and_oracle_token(self):
        batch = dict(self.batch, answer_available=torch.zeros(2, dtype=torch.bool))
        self.model.train()
        output = self.model(batch)
        torch.testing.assert_close(output['teacher_logits'], output['question_logits'], rtol=1e-5, atol=1e-6)
        loss, metrics = DistillationObjective()(self.model, batch)
        loss.backward()
        for name in ('distillation', 'latent_alignment', 'attn_guidance', 'kd_pixel_fraction'):
            self.assertEqual(metrics[name].item(), 0)
    def test_raw_full_checkpoint_and_optimizer_compatibility(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-5)
        loss, _ = SupervisedObjective(.5)(self.model, self.batch)
        loss.backward(); optimizer.step()
        self.model.eval()
        with torch.no_grad():
            expected = self.model(self.batch)['logits']
            changed = dict(self.batch, text=['unrelated', 'unrelated'], answer_only_text=['A: dog', 'A: blue'])
            torch.testing.assert_close(self.model(changed)['logits'], expected, rtol=0, atol=0)
        with tempfile.TemporaryDirectory() as tmp:
            for full in (False, True):
                path = Path(tmp) / f'{full}.pt'
                state = self.model.state_dict()
                saved = {'model_state_dict': state, 'experiment_config': self.model.experiment_config,
                         'run_config': {'tiny': True}, 'optimizer_state_dict': optimizer.state_dict(),
                         'epoch': 10} if full else state
                torch.save(saved, path)
                restored = GroundingModel.from_checkpoint(path, device='cpu')
                with torch.no_grad():
                    torch.testing.assert_close(restored(self.batch)['logits'], expected, rtol=0, atol=0)
                if full:
                    resumed = torch.optim.Adam(restored.parameters(), lr=1e-5)
                    resumed.load_state_dict(saved['optimizer_state_dict'])
                    restored.train(); resumed.zero_grad()
                    new_loss, _ = DistillationObjective()(restored, self.batch)
                    new_loss.backward(); resumed.step()
                    self.assertTrue(torch.isfinite(new_loss))
    def test_partial_weights_only_allowed_for_explicit_initialization(self):
        partial = {k: v.clone() for k, v in self.model.state_dict().items() if k not in COARSENED_LATENT_ANSWER_KEYS}
        with self.assertRaises(RuntimeError):
            load_model_weights(GroundingModel(tiny=True), {'model_state_dict': partial})
        initialized = GroundingModel(tiny=True)
        initialize_weights({'model_state_dict': partial}, initialized)
        for key, tensor in partial.items():
            torch.testing.assert_close(initialized.state_dict()[key], tensor, rtol=0, atol=0)
    def test_external_teacher_rejects_partial_and_accepts_complete_weights(self):
        teacher = TeacherModel(tiny=True)
        partial = {k: v for k, v in teacher.state_dict().items() if k not in LATENT_ANSWER_KEYS}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'teacher.pt'
            torch.save(partial, path)
            with self.assertRaisesRegex(ValueError, 'lacks trained latent'):
                load_teacher(path, torch.device('cpu'))
            torch.save({'model_state_dict': teacher.state_dict(), 'experiment_config': teacher.experiment_config,
                        'run_config': {'tiny': True, 'text_mode': 'dropout', 'answer_dropout': .5}}, path)
            loaded = load_teacher(path, torch.device('cpu'))
            objective = DistillationObjective(loaded)
            self.assertTrue(all(not p.requires_grad for p in loaded.parameters()))
            loss, _ = objective(self.model, self.batch)
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
    def test_supervised_control_has_no_privileged_branches(self):
        captured = {}
        original_forward = self.model.forward
        def capture(batch):
            output = original_forward(batch)
            captured.update(output)
            return output
        self.model.forward = capture
        loss, _ = SupervisedObjective(.5)(self.model, self.batch)
        loss.backward()
        self.assertNotIn('teacher_logits', captured)
        self.assertNotIn('target_answer', captured)
        check_arguments(training_parser().parse_args(['--supervised-only']))

if __name__ == '__main__':
    unittest.main()
