import unittest
import torch
from research.distillation import AnswerStudent, DistillationObjective, distillation_weights
from research.model import ResearchGrounder


class DistillationTests(unittest.TestCase):
    def test_incremental_weights_ignore_harmful_answers_and_balance_regions(self):
        target = torch.tensor([[[[1., 1., 0., 0.]]]])
        answer = torch.tensor([[[[4., -4., -4., 4.]]]], requires_grad=True)
        question = torch.zeros_like(answer)
        weights = distillation_weights(answer, question, target, "incremental")
        self.assertFalse(weights.requires_grad)
        torch.testing.assert_close(weights, torch.tensor([[[[.5, 0., .5, 0.]]]]))

    def test_no_gain_means_no_distillation(self):
        logits = torch.zeros(2, 1, 4, 4)
        weights = distillation_weights(logits, logits, torch.ones_like(logits), "incremental")
        self.assertEqual(weights.sum().item(), 0)

    def test_teacher_frozen_student_receives_gradients(self):
        torch.manual_seed(4)
        student = AnswerStudent(tiny=True)
        teacher = ResearchGrounder(tiny=True)
        objective = DistillationObjective(teacher, "ordinary", 1., 0.)
        batch = {"image": torch.rand(2,3,28,28), "text": ["Q: where?", "Q: what?"],
                 "question_text": ["Q: where?", "Q: what?"],
                 "answer_text": ["Q: where? A: left", "Q: what? A: cat"],
                 "answer_available": torch.tensor([True, False]), "mask": torch.ones(2,1,28,28)}
        student.train()
        loss, metrics = objective(student, batch)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertFalse(teacher.training)
        self.assertTrue(all(p.grad is None for p in teacher.parameters()))
        self.assertTrue(any(p.grad is not None for p in student.parameters()))
        self.assertGreater(metrics["distillation"].item(), 0.)
        self.assertFalse(any("teacher" in k for k in student.state_dict()))

    def test_missing_answers_produce_zero_kd(self):
        student = AnswerStudent(tiny=True)
        objective = DistillationObjective(ResearchGrounder(tiny=True), "ordinary", 1., 0.)
        batch = {"image": torch.rand(1,3,28,28), "text": ["Q: what?"],
                 "question_text": ["Q: what?"], "answer_text": ["Q: what?"],
                 "answer_available": torch.tensor([False]), "mask": torch.zeros(1,1,28,28)}
        loss, metrics = objective(student, batch)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(metrics["distillation"].item(), 0.)
