"""ocrg/main first commits: Dice term, shared scoring, run records and recipe guards. Offline (no CLIP)."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

import train
from evaluation import score_sample
from losses import segmentation_loss, soft_dice_loss
from metrics import compute_iou_per_sample
from run_record import check_expectations, check_resume_recipe, parse_expectations


class TinyModel(nn.Module):
    """Stands in for GroundingModel: (image, text) -> (B, 1, 16*H_p, 16*W_p) logits, with a text effect."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 1, 3, padding=1)
        with torch.no_grad():  # bright pixels -> foreground, so validation scores are not trivially 0
            self.conv.weight.fill_(0.1)
            self.conv.bias.zero_()
        self.text_bias = nn.Parameter(torch.zeros(1))

    def forward(self, image, text):
        x = F.avg_pool2d(image, 1)  # keep 336 px; real model outputs 384 px, any size is resized
        bias = self.text_bias * torch.tensor([len(t) % 3 for t in text], dtype=x.dtype)[:, None, None, None]
        return self.conv(x) + bias


class LossTests(unittest.TestCase):
    def test_zero_weight_is_upstream_bce(self):
        logits, masks = torch.randn(3, 1, 8, 8), (torch.rand(3, 1, 8, 8) > .7).float()
        self.assertTrue(torch.equal(segmentation_loss(logits, masks, 0.0), nn.BCEWithLogitsLoss()(logits, masks)))

    def test_dice_formula_per_image(self):
        logits, masks = torch.randn(2, 1, 4, 4), (torch.rand(2, 1, 4, 4) > .5).float()
        p = torch.sigmoid(logits)
        want = torch.stack([1 - (2 * (p[i] * masks[i]).sum() + 1) / (p[i].sum() + masks[i].sum() + 1) for i in range(2)]).mean()
        self.assertTrue(torch.allclose(soft_dice_loss(logits, masks), want, atol=1e-6))
        total = segmentation_loss(logits, masks, 0.5)
        self.assertTrue(torch.allclose(total, F.binary_cross_entropy_with_logits(logits, masks) + 0.5 * want, atol=1e-6))

    def test_half_precision_logits_are_summed_in_float32(self):
        logits, masks = torch.randn(2, 1, 64, 64), (torch.rand(2, 1, 64, 64) > .9).float()
        self.assertEqual(soft_dice_loss(logits.half(), masks).dtype, torch.float32)
        self.assertTrue(torch.allclose(soft_dice_loss(logits.half(), masks), soft_dice_loss(logits.half().float(), masks)))


class ScoringTests(unittest.TestCase):
    def test_score_sample_equals_previous_eval_code(self):
        torch.manual_seed(0)
        for legacy, size in ((False, (37, 51)), (True, (336, 336)), (False, (5, 3))):
            logits = torch.randn(1, 1, 24, 24)
            truth = (torch.rand(1, 1, *size) > .6).float()
            old = (F.interpolate(logits.float(), size=truth.shape[-2:], mode="bilinear", align_corners=False) > 0).float()
            prediction, score = score_sample(logits, truth, legacy=legacy)
            self.assertTrue(torch.equal(prediction, old))
            self.assertEqual(score, compute_iou_per_sample(old, truth, legacy=legacy).item())

    def test_empty_union_scores_one(self):
        self.assertEqual(score_sample(-torch.ones(1, 1, 4, 4), torch.zeros(1, 1, 9, 9))[1], 1.0)


class RecipeGuardTests(unittest.TestCase):
    def test_expectations(self):
        expected = parse_expectations(["dice_weight=0.5", "text_mode=question", "world_size=2"])
        self.assertEqual(expected, {"dice_weight": 0.5, "text_mode": "question", "world_size": 2})
        check_expectations({"dice_weight": 0.5, "text_mode": "question", "world_size": 2}, expected)
        with self.assertRaisesRegex(ValueError, "dice_weight"):
            check_expectations({"dice_weight": 0.0, "text_mode": "question", "world_size": 2}, expected)
        with self.assertRaisesRegex(ValueError, "missing"):
            check_expectations({}, {"text_mode": "question"})
        with self.assertRaises(ValueError):
            parse_expectations(["dice_weight"])

    def test_resume_recipe(self):
        recipe = {"text_mode": "question", "dice_weight": 0.5, "lr": 1e-5, "batch_size": 4, "seed": 42, "world_size": 2}
        check_resume_recipe({"run_config": {"recipe": dict(recipe)}}, recipe)
        with self.assertRaisesRegex(ValueError, "no recorded recipe"):
            check_resume_recipe({"model_state_dict": {}}, recipe)
        with self.assertRaisesRegex(ValueError, "text_mode"):
            check_resume_recipe({"run_config": {"recipe": {**recipe, "text_mode": "qa"}}}, recipe)


class TrainSmokeTests(unittest.TestCase):
    """train.main end to end on synthetic data with a tiny model (CPU, single process)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "data"
        sizes = {"train": [(20, 16), (16, 20), (24, 18), (18, 24)], "val": [(13, 9), (8, 6), (10, 10)]}
        for split, shapes in sizes.items():
            (self.root / split).mkdir(parents=True)
            (self.root / "binary_masks_png" / split).mkdir(parents=True)
            records = {}
            for i, size in enumerate(shapes):
                name = f"{split}_{i}.jpg"
                image = Image.new("RGB", size, (0, 0, 0))
                image.paste((255, 255, 255), (0, 0, size[0] // 2, size[1] // 3 + i))  # bright block near the target
                image.save(self.root / split / name)
                mask = Image.new("L", size, 0)
                mask.paste(255, (0, 0, size[0] // 2, size[1] // 2))
                mask.save(self.root / "binary_masks_png" / split / f"{split}_{i}.png")
                records[name] = {"question": f"What is item {i}?", "most_common_answer": "secret answer"}
            (self.root / f"{split}_grounding.json").write_text(json.dumps(records))
        self.out = Path(self.tmp.name) / "out"

    def tearDown(self):
        self.tmp.cleanup()

    def run_train(self, *extra):
        argv = ["train.py", "--data-root", str(self.root), "--output-dir", str(self.out), "--num-epochs", "2",
                "--batch-size", "2", "--num-workers", "0", "--save-every", "1", "--validate-every", "1",
                "--text-mode", "question", "--dice-weight", "0.5", *extra]
        with patch.object(sys, "argv", argv), patch.object(train, "GroundingModel", TinyModel):
            train.main()

    def test_records_validation_curve_and_checkpoint_recipe(self):
        self.run_train("--expect", "dice_weight=0.5", "--expect", "text_mode=question")
        history = [json.loads(line) for line in (self.out / "history.jsonl").read_text().splitlines()]
        self.assertEqual([h["epoch"] for h in history], [1, 2])
        self.assertTrue(all(h["training_config"]["dice_weight"] == 0.5 for h in history))
        curve = [json.loads(line) for line in (self.out / "val_curve.jsonl").read_text().splitlines()]
        self.assertEqual([c["num_images"] for c in curve], [3, 3])
        checkpoint = torch.load(self.out / "checkpoint_epoch2.pt", map_location="cpu", weights_only=False)
        self.assertEqual(checkpoint["run_config"]["recipe"]["text_mode"], "question")
        self.assertIn("commit", checkpoint["run_config"]["git"])

        # The logged validation mIoU equals rescoring the saved weights with the shared rule.
        model = TinyModel()
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        val = train.VizWizGroundingDataset(self.root / "val_grounding.json", self.root / "val",
                                           self.root / "binary_masks_png" / "val", is_test=True,
                                           text_mode="question", return_original_mask=True)
        with torch.no_grad():
            scores = [score_sample(model(s["image"][None], [s["text"]]), s["original_mask"][None])[1] for s in val]
        self.assertGreater(curve[-1]["val_miou"], 0.2)  # non-trivial predictions
        self.assertAlmostEqual(curve[-1]["val_miou"], sum(scores) / len(scores), places=12)
        self.assertTrue(all("secret answer" not in s["text"] for s in val))  # question-only input

    def test_expect_mismatch_stops_before_training(self):
        with self.assertRaisesRegex(ValueError, "nothing was trained"):
            self.run_train("--expect", "dice_weight=0")
        self.assertFalse((self.out / "history.jsonl").exists())

    def test_resume_refuses_a_different_recipe(self):
        self.run_train()
        with self.assertRaisesRegex(ValueError, "dice_weight"):
            argv_extra = ("--resume-checkpoint", str(self.out / "checkpoint_epoch1.pt"))
            with patch.object(sys, "argv", ["train.py", "--data-root", str(self.root), "--output-dir", str(self.out),
                                            "--num-epochs", "3", "--batch-size", "2", "--num-workers", "0",
                                            "--text-mode", "question", "--dice-weight", "1.0", *argv_extra]), \
                    patch.object(train, "GroundingModel", TinyModel):
                train.main()


if __name__ == "__main__":
    unittest.main()
