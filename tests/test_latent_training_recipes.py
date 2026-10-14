"""Loss mathematics, training recipes, resume and DDP valid-target reduction."""
import json
import tempfile
from datetime import timedelta
from pathlib import Path
import unittest
from unittest.mock import patch

from PIL import Image
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from diagnose import diagnostic_views, load_diagnostic_model
from losses import (AnswerAlignmentObjective, DistillationObjective, attention_guidance_loss,
                    attention_target_distribution, bernoulli_kl, latent_answer_loss,
                    masked_loss_mean, segmentation_loss)
from models.model import GroundingModel
import train as trainer


def recipe_args(flags=()):
    args = trainer.training_parser().parse_args(list(flags))
    args.teacher_training = False
    trainer.resolve_training_arguments(args)
    return args


def training_batch():
    image = torch.rand(2, 3, 32, 32)
    return {"image": image, "mask": (image[:, :1] > .5).float(),
            "text": ["Q: color?", "Q: object?"], "question_text": ["Q: color?", "Q: object?"],
            "answer_only_text": ["A: red", "A: cup"],
            "answer_available": torch.tensor([True, False])}


def stable_ddp_worker(rank, rendezvous, output):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank,
                            world_size=2, timeout=timedelta(seconds=45))
    try:
        torch.manual_seed(777)
        model = GroundingModel(tiny=True)
        batch = training_batch()
        batch = {k: v[rank:rank + 1] for k, v in batch.items()}
        ddp = DDP(model, find_unused_parameters=True)
        optimizer = torch.optim.SGD(ddp.parameters(), lr=.001)
        warmup = AnswerAlignmentObjective(loss_version="stable", warmup_epochs=1, detach_predicted_answer=True)
        for objective, epoch in ((AnswerAlignmentObjective(loss_version="stable"), 0),
                                 (warmup, 0), (warmup, 1), (DistillationObjective(loss_version="stable"), 0)):
            optimizer.zero_grad()
            loss, _ = objective(ddp, batch, epoch)
            loss.backward()
            optimizer.step()
        torch.save(model.state_dict(), Path(output) / f"rank{rank}.pt")
    finally:
        dist.destroy_process_group()


class LatentTrainingRecipeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(42)
        self.model = GroundingModel(tiny=True)
        self.batch = training_batch()

    def test_bernoulli_kl_direction_gradients_and_extreme_logits(self):
        student = torch.linspace(-12, 12, 31, requires_grad=True)
        teacher = torch.linspace(9, -9, 31, requires_grad=True)
        actual = bernoulli_kl(student, teacher)
        expected = torch.distributions.kl_divergence(
            torch.distributions.Bernoulli(logits=teacher.detach().double()),
            torch.distributions.Bernoulli(logits=student.detach().double()))
        torch.testing.assert_close(actual.double(), expected, rtol=1e-5, atol=2e-6)
        actual.sum().backward()
        torch.testing.assert_close(student.grad, student.detach().sigmoid() - teacher.detach().sigmoid(), rtol=1e-5, atol=2e-7)
        self.assertIsNone(teacher.grad)
        self.assertTrue(torch.equal(bernoulli_kl(student.detach(), student.detach()), torch.zeros_like(student)))
        extreme = bernoulli_kl(torch.tensor([-1000., 1000.]), torch.tensor([1000., -1000.]))
        torch.testing.assert_close(extreme, torch.tensor([1000., 1000.]))

    def test_stable_attention_is_normalized_forward_kl(self):
        attn = torch.tensor([[[[.5, 0], [.4, .1]]]], requires_grad=True)
        target = torch.tensor([[[[0., 1.], [0., 0.]]]])
        weights = target / 4
        distribution, valid = attention_target_distribution(weights, target, (2, 2))
        q = attn.flatten(1) + 1e-8
        q = q / q.sum(-1, keepdim=True)
        expected = F.kl_div(q.log(), distribution, reduction="batchmean")
        actual = attention_guidance_loss(attn, weights, target, loss_version="stable")
        torch.testing.assert_close(actual, expected)
        self.assertEqual(valid.item(), 1)
        actual.backward()
        self.assertTrue(torch.isfinite(attn.grad).all())
        self.assertNotEqual(attn.grad[0, 0, 0, 1].item(), 0)
        empty = attention_guidance_loss(attn, weights, torch.zeros_like(target), loss_version="stable")
        self.assertEqual(empty.item(), 0)

    def test_alignment_valid_mean_and_no_target_gradient(self):
        latent = torch.randn(2, 16, requires_grad=True)
        target = torch.randn(2, 16, requires_grad=True)
        available = torch.tensor([True, False])
        stable = latent_answer_loss(latent, target, available, loss_version="stable")
        one = latent_answer_loss(latent[:1], target[:1], loss_version="stable")
        torch.testing.assert_close(stable, one)
        legacy = latent_answer_loss(latent, target, available)
        torch.testing.assert_close(stable, 2 * legacy)
        stable.backward()
        self.assertIsNone(target.grad)
        self.assertEqual(latent.grad[1].abs().sum().item(), 0)
        zero = latent_answer_loss(latent, target, torch.zeros(2, dtype=torch.bool), loss_version="stable")
        self.assertEqual(zero.item(), 0)

    def test_segmentation_per_image_matches_mean_and_finite_empty_mask(self):
        logits = torch.randn(2, 1, 5, 7, requires_grad=True)
        target = torch.zeros(2, 1, 11, 13)
        per_image = segmentation_loss(logits, target, .5, reduction="none")
        torch.testing.assert_close(per_image.mean(), segmentation_loss(logits, target, .5))
        per_image.sum().backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_detach_cuts_only_direct_grounding_predictor_path(self):
        for detached in (False, True):
            output = self.model(dict(self.batch, self_distill=False, detach_predicted_answer=detached))
            loss = segmentation_loss(output["logits"], self.batch["mask"], .5)
            gradient = torch.autograd.grad(loss, output["latent_answer"], allow_unused=True, retain_graph=True)[0]
            if detached:
                self.assertIsNone(gradient)
            else:
                self.assertGreater(gradient.abs().sum().item(), 0)
            alignment = latent_answer_loss(output["latent_answer"], output["target_answer"], self.batch["answer_available"])
            predictor_gradient = torch.autograd.grad(alignment, self.model.coarse.visual_to_text.weight)[0]
            self.assertGreater(predictor_gradient.abs().sum().item(), 0)

    def test_warmup_loss_schedule_missing_answer_and_no_extra_branches(self):
        objective = AnswerAlignmentObjective(warmup_epochs=2, transition_epochs=2, loss_version="stable")
        original = self.model.forward
        captured = {}
        def capture(batch):
            captured.clear()
            output = original(batch)
            captured.update(output)
            return output
        self.model.forward = capture
        for epoch, fraction in enumerate((0., 0., .5, 1., 1.)):
            total, metrics = objective(self.model, self.batch, epoch)
            self.assertEqual(metrics["predicted_fraction"].item(), fraction)
            predicted = segmentation_loss(captured["logits"], self.batch["mask"], .5, reduction="none")
            if fraction < 1:
                oracle = segmentation_loss(captured["teacher_logits"], self.batch["mask"], .5, reduction="none")
                expected = ((fraction * predicted[0] + (1 - fraction) * oracle[0]) + predicted[1]) / 2
            else:
                self.assertNotIn("teacher_logits", captured)
                expected = predicted.mean()
            torch.testing.assert_close(metrics["segmentation"], expected.detach())
            torch.testing.assert_close(total, expected + .5 * metrics["latent_alignment"])
            self.assertNotIn("question_logits", captured)

    def test_default_recipe_and_alias_preserve_previous_losses(self):
        args = recipe_args()
        self.assertEqual(args.training_recipe, "full")
        self.assertEqual(args.loss_version, "legacy")
        self.assertEqual(trainer.objective_version(args), "latent-gain-v2")
        actual, _ = trainer.build_objective(args)(self.model, self.batch)
        expected, _ = DistillationObjective()(self.model, self.batch)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        alias = recipe_args(["--supervised-only"])
        direct = recipe_args(["--training-recipe", "grounding-only"])
        torch.testing.assert_close(trainer.build_objective(alias)(self.model, self.batch)[0],
                                   trainer.build_objective(direct)(self.model, self.batch)[0], rtol=0, atol=0)

    def test_stable_full_masks_oracle_gradients_for_missing_answers(self):
        original = self.model.forward
        captured = {}
        def capture(batch):
            output = original(batch)
            captured.update(output)
            output["teacher_logits"].retain_grad()
            output["question_logits"].retain_grad()
            return output
        self.model.forward = capture
        loss, metrics = DistillationObjective(loss_version="stable")(self.model, self.batch)
        loss.backward()
        self.assertGreater(captured["teacher_logits"].grad[0].abs().sum().item(), 0)
        self.assertEqual(captured["teacher_logits"].grad[1].abs().sum().item(), 0)
        self.assertGreater(captured["question_logits"].grad[1].abs().sum().item(), 0)
        expected = sum(metrics[key] for key in ("segmentation", "weighted_oracle_segmentation",
            "weighted_question_segmentation", "weighted_distillation", "weighted_latent_alignment", "weighted_attn_guidance"))
        torch.testing.assert_close(loss.detach(), expected)

    def test_resume_restores_settings_and_rejects_objective_changes(self):
        args = recipe_args(["--training-recipe", "warmup", "--loss-version", "stable", "--detach-predicted-answer",
                            "--warmup-epochs", "2", "--answer-transition-epochs", "3"])
        saved = {"run_config": {"training_config": {k: getattr(args, k) for k in trainer.TRAINING_DEFAULTS},
                                "objective_version": trainer.objective_version(args)}}
        restored = trainer.training_parser().parse_args([])
        restored.teacher_training = False
        trainer.resolve_training_arguments(restored, saved)
        self.assertEqual(trainer.objective_version(restored), trainer.objective_version(args))
        for flags in (["--training-recipe", "alignment"], ["--warmup-epochs", "4"],
                      ["--loss-version", "legacy"], ["--no-detach-predicted-answer"]):
            changed = trainer.training_parser().parse_args(flags)
            changed.teacher_training = False
            with self.assertRaisesRegex(ValueError, "Resume"):
                trainer.resolve_training_arguments(changed, saved)
        old = {"run_config": {"objective_version": "latent-gain-v2", "loss_config": {"latent_weight": .3}}}
        restored = trainer.training_parser().parse_args([])
        restored.teacher_training = False
        trainer.resolve_training_arguments(restored, old)
        self.assertEqual(restored.latent_weight, .3)

    def test_invalid_recipe_combinations_are_rejected(self):
        for flags in (["--supervised-only", "--training-recipe", "alignment"],
                      ["--training-recipe", "grounding-only", "--detach-predicted-answer"],
                      ["--training-recipe", "alignment", "--latent-weight", "0", "--detach-predicted-answer"],
                      ["--training-recipe", "alignment", "--kd-weight", "1"],
                      ["--warmup-epochs", "2"], ["--latent-weight", "nan"], ["--gain-mix", "2"],
                      ["--training-recipe", "warmup", "--num-epochs", "2"]):
            with self.subTest(flags=flags), self.assertRaises(ValueError):
                recipe_args(flags)

    def test_cli_recipes_checkpoint_shapes_and_warmup_resume_equivalence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "train").mkdir()
            (root / "binary_masks_png" / "train").mkdir(parents=True)
            records = {}
            for i in range(4):
                name = f"image{i}.jpg"
                Image.new("RGB", (32, 32), (40 * i, 100, 200)).save(root / "train" / name)
                Image.new("L", (32, 32), 255 if i % 2 else 0).save(root / "binary_masks_png" / "train" / f"image{i}.png")
                records[name] = {"question": "What is this?", "most_common_answer": "cup" if i % 2 else ""}
            (root / "train_grounding.json").write_text(json.dumps(records))
            common = ["--data-root", str(root), "--batch-size", "2", "--num-workers", "0", "--save-every", "1"]
            with patch.object(trainer, "IMAGE_SIZE", 32), patch.object(trainer, "GroundingModel", side_effect=lambda **kwargs: GroundingModel(tiny=True)):
                for recipe in ("full", "grounding-only", "alignment", "warmup"):
                    flags = ["--training-recipe", recipe, "--loss-version", "stable", "--num-epochs", "3" if recipe == "warmup" else "1"]
                    if recipe == "warmup":
                        flags += ["--warmup-epochs", "1", "--answer-transition-epochs", "2", "--detach-predicted-answer"]
                    out = root / recipe
                    trainer.train_main(common + flags + ["--output-dir", str(out)])
                    checkpoint = out / ("checkpoint_epoch3.pt" if recipe == "warmup" else "checkpoint_epoch1.pt")
                    model, _ = load_diagnostic_model(checkpoint, torch.device("cpu"))
                    self.assertEqual(set(model.state_dict()), set(self.model.state_dict()))
                    with torch.no_grad():
                        torch.testing.assert_close(diagnostic_views(model, self.batch)["logits"]["predicted_answer"],
                                                   model(self.batch)["logits"], rtol=0, atol=0)
                resumed = root / "resumed"
                trainer.train_main(common + ["--resume-checkpoint", str(root / "warmup" / "checkpoint_epoch1.pt"),
                                            "--num-epochs", "3", "--output-dir", str(resumed)])
            final = torch.load(root / "warmup" / "model_final_epoch3.pt", weights_only=False)
            continued = torch.load(resumed / "model_final_epoch3.pt", weights_only=False)
            for key in final:
                torch.testing.assert_close(final[key], continued[key], rtol=0, atol=0)
            history = [json.loads(line) for line in (resumed / "history.jsonl").read_text().splitlines()]
            self.assertEqual([r["predicted_fraction"] for r in history], [.5, 1.])

    def test_ddp_valid_target_mean_matches_global_batch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mp.spawn(stable_ddp_worker, args=(str(root / "rendezvous"), str(root)), nprocs=2, join=True)
            torch.manual_seed(777)
            reference = GroundingModel(tiny=True)
            batch = training_batch()
            optimizer = torch.optim.SGD(reference.parameters(), lr=.001)
            warmup = AnswerAlignmentObjective(loss_version="stable", warmup_epochs=1, detach_predicted_answer=True)
            for objective, epoch in ((AnswerAlignmentObjective(loss_version="stable"), 0),
                                     (warmup, 0), (warmup, 1), (DistillationObjective(loss_version="stable"), 0)):
                optimizer.zero_grad()
                loss, _ = objective(reference, batch, epoch)
                loss.backward(); optimizer.step()
            for rank in range(2):
                actual = torch.load(root / f"rank{rank}.pt", weights_only=False)
                for key, value in reference.state_dict().items():
                    torch.testing.assert_close(actual[key], value, rtol=1e-5, atol=2e-7, msg=key)


if __name__ == "__main__":
    unittest.main()
