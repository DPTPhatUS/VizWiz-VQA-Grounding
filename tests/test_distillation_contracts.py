"""Offline checks for unchanged answer-value semantics and checkpoint loading."""
import copy
import tempfile
from pathlib import Path
import unittest
import torch
from torch import nn
from torch.nn import functional as F
from losses import DistillationObjective, distillation_weights, resize_logits, segmentation_loss
from models.model import GroundingModel
from train import training_parser, check_arguments

class FixedTeacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(3.0))
    def encode_image(self, image):
        return image
    def decode(self, features, texts):
        magnitude = torch.tensor([3. if ' A: ' in text else .2 for text in texts]).view(-1, 1, 1, 1)
        return {'logits': torch.where(features[:, :1] > .5, magnitude, -magnitude) + 0 * self.scale}

class DistillationContracts(unittest.TestCase):
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
                      'answer_available': torch.ones(2, dtype=torch.bool)}
    def legacy_loss(self, model, teacher, batch):
        student = resize_logits(model(batch)['logits'], batch['mask']).float()
        supervised = segmentation_loss(student, batch['mask'])
        with torch.no_grad():
            features = teacher.encode_image(batch['image'])
            answer = resize_logits(teacher.decode(features, batch['answer_text'])['logits'], batch['mask']).float()
            question = teacher.decode(features, batch['question_text'])['logits']
            probability = answer.sigmoid()
            entropy = F.binary_cross_entropy_with_logits(answer, probability, reduction='none')
            weights = distillation_weights(answer, question, batch['mask'], probability=probability, entropy=entropy)
            weights *= batch['answer_available'].to(weights).view(-1, 1, 1, 1)
        divergence = F.binary_cross_entropy_with_logits(student, probability, reduction='none') - entropy
        return supervised + (weights * divergence).flatten(1).sum(1).mean()
    def test_loss_and_gradients_exactly_match_previous_objective(self):
        previous = copy.deepcopy(self.model)
        old_loss = self.legacy_loss(previous, FixedTeacher().eval(), self.batch)
        new_loss, metrics = DistillationObjective(FixedTeacher())(self.model, self.batch)
        torch.testing.assert_close(new_loss, old_loss, rtol=0, atol=0)
        old_loss.backward(); new_loss.backward()
        for old, new in zip(previous.parameters(), self.model.parameters()):
            if old.grad is not None:
                torch.testing.assert_close(old.grad, new.grad, rtol=0, atol=0)
        self.assertNotIn('active_fraction', metrics)
        self.assertGreater(metrics['kd_pixel_fraction'].item(), 0)
        self.assertIn('teacher_iou_gain', metrics)
    def test_missing_answers_disable_kd_and_teacher_is_frozen(self):
        batch = dict(self.batch, answer_available=torch.zeros(2, dtype=torch.bool))
        teacher = FixedTeacher()
        loss, metrics = DistillationObjective(teacher)(self.model, batch)
        loss.backward()
        self.assertEqual(metrics['distillation'].item(), 0)
        self.assertEqual(metrics['mean_kd_weight_mass'].item(), 0)
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in teacher.parameters()))
    def test_equal_views_have_zero_gain(self):
        logits = torch.ones_like(self.batch['mask'])
        self.assertEqual(distillation_weights(logits, logits, self.batch['mask']).sum().item(), 0)
    def test_raw_full_checkpoint_roundtrip_and_answer_independence(self):
        self.model.eval()
        with torch.no_grad():
            expected = self.model(self.batch)['logits']
            changed = dict(self.batch, text=['unrelated', 'unrelated'], answer_text=['A: dog', 'A: blue'])
            torch.testing.assert_close(self.model(changed)['logits'], expected, rtol=0, atol=0)
        with tempfile.TemporaryDirectory() as tmp:
            for full in (False, True):
                path = Path(tmp) / f'{full}.pt'
                state = self.model.state_dict()
                saved = {'model_state_dict': state, 'experiment_config': self.model.experiment_config,
                         'run_config': {'tiny': True}, 'epoch': 10} if full else state
                torch.save(saved, path)
                restored = GroundingModel.from_checkpoint(path, device='cpu')
                with torch.no_grad():
                    torch.testing.assert_close(restored(self.batch)['logits'], expected, rtol=0, atol=0)
    def test_supervised_control_does_not_require_teacher(self):
        args = training_parser().parse_args(['--supervised-only'])
        check_arguments(args)
        with self.assertRaises(ValueError):
            check_arguments(training_parser().parse_args([]))

if __name__ == '__main__':
    unittest.main()
