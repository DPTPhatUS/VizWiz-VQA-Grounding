"""Offline tests for ocrg/main (B0): data, model, loss, evaluation, training and run records.

Tiny randomly initialised CLIP encoders (ModelConfig(tiny=True)); only the cached CLIP tokenizer is read.
Run: python -m unittest discover -s tests -p "test_*.py"
"""
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import json
import socket
import sys
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader

import eval as evaluator
import train
from dataset import CLIP_MEAN, CLIP_STD, VizWizGroundingDataset, collate
from evaluation import ShardSampler, evaluate, gather_scores, mean_iou
from losses import segmentation_loss, soft_dice_loss
from metrics import iou_per_sample
from models import FusionDecoder, GroundingModel, ModelConfig
from run_record import check_expectations, check_resume_recipe, parse_expectations


@dataclass(frozen=True)
class TinyConfig(ModelConfig):
    tiny: bool = True


def make_split(root: Path, split: str, sizes, with_answers=True):
    (root / split).mkdir(parents=True)
    (root / "binary_masks_png" / split).mkdir(parents=True)
    records = {}
    for i, (w, h) in enumerate(sizes):
        name = f"{split}_{i}.jpg"
        image = Image.new("RGB", (w, h), (20, 20, 20))
        image.paste((240, 240, 240), (0, 0, w // 2, h // 2 + i))
        image.save(root / split / name)
        mask = Image.new("L", (w, h), 0)
        mask.paste(255, (0, 0, w // 2, h // 2))
        mask.save(root / "binary_masks_png" / split / f"{split}_{i}.png")
        records[name] = {"question": f"What is item {i}?"}
        if with_answers:
            records[name]["most_common_answer"] = "SECRET-ANSWER"
    (root / f"{split}_grounding.json").write_text(json.dumps(records))


class DataTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        make_split(self.root, "val", [(13, 9), (40, 30)])

    def tearDown(self):
        self.tmp.cleanup()

    def test_question_only_text_and_reference_preprocessing(self):
        data = VizWizGroundingDataset(self.root, "val", native_mask=True)
        sample = data[1]
        self.assertEqual(sample["text"], "Q: What is item 1?")
        self.assertNotIn("SECRET", sample["text"])
        with Image.open(self.root / "val" / "val_1.jpg") as im:
            ref = T.Normalize(CLIP_MEAN, CLIP_STD)(TF.to_tensor(TF.resize(im.convert("RGB"), [336, 336],
                                                                           interpolation=T.InterpolationMode.BICUBIC, antialias=True)))
        self.assertTrue(torch.equal(sample["image"], ref))
        self.assertEqual(tuple(sample["mask"].shape), (1, 336, 336))
        self.assertEqual(set(sample["mask"].unique().tolist()), {0.0, 1.0})
        self.assertEqual((tuple(sample["native_mask"].shape), sample["native_mask"].dtype), ((1, 30, 40), torch.bool))
        batch = collate([data[0], data[1]])
        self.assertEqual(tuple(batch["image"].shape), (2, 3, 336, 336))
        self.assertEqual([tuple(m.shape) for m in batch["native_mask"]], [(1, 9, 13), (1, 30, 40)])

    def test_missing_or_misaligned_mask_fails(self):
        (self.root / "binary_masks_png" / "val" / "val_0.png").unlink()
        with self.assertRaises(FileNotFoundError):
            VizWizGroundingDataset(self.root, "val")[0]
        Image.new("L", (5, 5), 0).save(self.root / "binary_masks_png" / "val" / "val_0.png")
        with self.assertRaisesRegex(ValueError, "sizes differ"):
            VizWizGroundingDataset(self.root, "val")[0]


class LossAndMetricTests(unittest.TestCase):
    def test_zero_dice_weight_is_bce(self):
        logits, masks = torch.randn(3, 1, 8, 8), (torch.rand(3, 1, 8, 8) > .7).float()
        self.assertTrue(torch.equal(segmentation_loss(logits, masks, 0.0), nn.BCEWithLogitsLoss()(logits, masks)))

    def test_dice_formula(self):
        logits, masks = torch.randn(2, 1, 4, 4), (torch.rand(2, 1, 4, 4) > .5).float()
        p = torch.sigmoid(logits)
        want = torch.stack([1 - (2 * (p[i] * masks[i]).sum() + 1) / (p[i].sum() + masks[i].sum() + 1) for i in range(2)]).mean()
        self.assertTrue(torch.allclose(soft_dice_loss(logits, masks), want, atol=1e-6))
        self.assertTrue(torch.allclose(segmentation_loss(logits, masks, 0.5),
                                       F.binary_cross_entropy_with_logits(logits, masks) + 0.5 * want, atol=1e-6))
        self.assertEqual(soft_dice_loss(logits.half(), masks).dtype, torch.float32)

    def test_iou(self):
        pred = torch.zeros(3, 1, 4, 4, dtype=torch.bool)
        true = torch.zeros(3, 1, 4, 4, dtype=torch.bool)
        pred[0, :, :2] = True
        true[0, :, :1] = True  # IoU 4/8
        pred[1, :, 0, 0] = True  # no overlap -> 0
        self.assertEqual(iou_per_sample(pred, true).tolist(), [0.5, 0.0, 1.0])  # empty union -> 1


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        cls.model = GroundingModel(TinyConfig())
        cls.images = torch.randn(2, 3, 336, 336)
        cls.texts = ["Q: what is this?", "Q: what does the label on the bottle say?"]

    def test_fusion_decoder_shape_and_identity_start(self):
        decoder = FusionDecoder(in_channels=32)
        feats = [torch.randn(2, 32, 24, 24) for _ in range(4)]
        self.assertEqual(tuple(decoder(*feats).shape), (2, 1, 384, 384))
        x = decoder.input_proj(feats[0])
        for block in decoder.fusion:  # every skip branch starts at exactly zero
            self.assertEqual(block(x + torch.randn_like(x)).abs().max().item(), 0.0)

    def test_forward_shapes_and_upstream_variant(self):
        self.assertEqual(tuple(self.model(self.images, self.texts).shape), (2, 1, 384, 384))
        upstream = GroundingModel(TinyConfig(decoder="unet", attention_residual=False))
        self.assertIsNone(upstream.attention_scale)
        self.assertEqual(tuple(upstream(self.images, self.texts).shape), (2, 1, 384, 384))

    def test_residual_attention(self):
        vision = torch.randn(2, 32, 24, 24)
        tokens, padding = self.model.text_encoder(self.texts)
        with torch.no_grad():
            self.model.attention_scale.fill_(0.0)
            self.assertTrue(torch.equal(self.model.fuse(vision, tokens, padding), vision))  # alpha = 0 -> F = V
            self.model.attention_scale.fill_(0.01)
            fused = self.model.fuse(vision, tokens, padding)
            noisy = tokens.clone()
            noisy[padding] = torch.randn_like(noisy[padding])  # padded tokens must not matter
            self.assertTrue(torch.allclose(self.model.fuse(vision, noisy, padding), fused, atol=1e-6))
        self.assertTrue(padding.any())

    def test_every_trainable_parameter_gets_a_gradient(self):
        """Needed for DDP with find_unused_parameters=False (B0: text encoder frozen)."""
        model = GroundingModel(TinyConfig())
        model.text_encoder.requires_grad_(False)
        loss = segmentation_loss(model(self.images, self.texts), torch.ones(2, 1, 384, 384), 0.5)
        loss.backward()
        missing = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
        self.assertEqual(missing, [])
        self.assertFalse(any(p.requires_grad for n, p in model.named_parameters() if "post_layernorm" in n))


class EvaluationTests(unittest.TestCase):
    def test_shard_sampler_covers_each_index_once(self):
        shards = [list(ShardSampler(7, r, 3)) for r in range(3)]
        self.assertEqual(sorted(sum(shards, [])), list(range(7)))
        self.assertEqual([len(s) for s in shards], [3, 2, 2])

    def test_evaluate_matches_direct_scoring(self):
        with tempfile.TemporaryDirectory() as tmp:
            make_split(Path(tmp), "val", [(13, 9), (40, 30), (21, 33)])
            data = VizWizGroundingDataset(tmp, "val", native_mask=True)
            torch.manual_seed(1)
            model = GroundingModel(TinyConfig()).eval()
            saved = {}
            scores = evaluate(model, DataLoader(data, batch_size=2, collate_fn=collate), torch.device("cpu"),
                              on_prediction=lambda name, mask: saved.setdefault(name, mask))
            for s in data:
                with torch.no_grad():
                    logits = F.interpolate(model(s["image"][None], [s["text"]]), size=s["native_mask"].shape[-2:],
                                           mode="bilinear", align_corners=False)
                self.assertTrue(torch.equal(saved[s["filename"]], logits[0, 0] > 0))
                self.assertEqual(scores[s["filename"]], iou_per_sample(logits > 0, s["native_mask"][None]).item())
            self.assertEqual(mean_iou(scores, 3), sum(scores.values()) / 3)
            with self.assertRaises(RuntimeError):
                mean_iou(scores, 4)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _distributed_eval_worker(rank, world_size, port, root, out_file):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank),
                      WORLD_SIZE=str(world_size), LOCAL_RANK=str(rank))
    from utils import cleanup_distributed, setup_distributed
    rank, world_size, _, device = setup_distributed()
    try:
        torch.manual_seed(1)
        model = GroundingModel(TinyConfig()).eval()
        data = VizWizGroundingDataset(root, "val", native_mask=True)
        loader = DataLoader(data, batch_size=2, sampler=ShardSampler(len(data), rank, world_size), collate_fn=collate)
        scores = gather_scores(evaluate(model, loader, device), world_size)
        if rank == 0:
            Path(out_file).write_text(json.dumps(scores))
    finally:
        cleanup_distributed()


def _distributed_train_worker(rank, world_size, port, argv):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank),
                      WORLD_SIZE=str(world_size), LOCAL_RANK=str(rank))
    with patch.object(sys, "argv", argv), patch.object(train, "ModelConfig", TinyConfig):
        train.main()


class DistributedTests(unittest.TestCase):
    """Two CPU processes (gloo) exercise the multi-GPU code paths."""

    def test_two_process_evaluation_equals_single_process(self):
        with tempfile.TemporaryDirectory() as tmp:
            make_split(Path(tmp), "val", [(13, 9), (40, 30), (21, 33), (8, 8), (30, 12)])
            mp.spawn(_distributed_eval_worker, args=(2, free_port(), tmp, f"{tmp}/scores.json"), nprocs=2)
            torch.manual_seed(1)
            model = GroundingModel(TinyConfig()).eval()
            data = VizWizGroundingDataset(tmp, "val", native_mask=True)
            single = evaluate(model, DataLoader(data, batch_size=2, collate_fn=collate), torch.device("cpu"))
            self.assertEqual(json.loads(Path(f"{tmp}/scores.json").read_text()), single)

    def test_two_process_training_runs_without_unused_parameter_search(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, out = Path(tmp) / "data", Path(tmp) / "out"
            make_split(root, "train", [(20, 16), (16, 20), (24, 18), (18, 24)])
            make_split(root, "val", [(13, 9), (8, 6), (10, 10)])
            argv = ["train.py", "--data-root", str(root), "--output-dir", str(out), "--num-epochs", "1",
                    "--batch-size", "2", "--num-workers", "0", "--validate-every", "1", "--dice-weight", "0.5",
                    "--freeze-text-encoder", "--no-freeze-vision-encoder", "--expect", "world_size=2"]
            mp.spawn(_distributed_train_worker, args=(2, free_port(), argv), nprocs=2)
            curve = [json.loads(line) for line in (out / "val_curve.jsonl").read_text().splitlines()]
            self.assertEqual(curve[0]["num_images"], 3)
            self.assertTrue((out / "checkpoint_epoch1.pt").exists())


class RecipeGuardTests(unittest.TestCase):
    def test_expectations(self):
        expected = parse_expectations(["dice_weight=0.5", "decoder=fusion", "freeze_text_encoder=true"])
        self.assertEqual(expected, {"dice_weight": 0.5, "decoder": "fusion", "freeze_text_encoder": True})
        check_expectations({"dice_weight": 0.5, "decoder": "fusion", "freeze_text_encoder": True}, expected)
        with self.assertRaisesRegex(ValueError, "dice_weight"):
            check_expectations({"dice_weight": 0.0, "decoder": "fusion", "freeze_text_encoder": True}, expected)
        with self.assertRaisesRegex(ValueError, "missing"):
            check_expectations({}, {"decoder": "fusion"})
        with self.assertRaises(ValueError):
            parse_expectations(["dice_weight"])

    def test_resume_recipe(self):
        recipe = {"dice_weight": 0.5, "lr": 1e-5}
        with self.assertRaisesRegex(ValueError, "no recorded recipe"):
            check_resume_recipe({"model_state_dict": {}}, recipe)


class TrainingTests(unittest.TestCase):
    """train.main and eval.main end to end on synthetic data (CPU, one process, tiny encoders)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root, self.out = Path(self.tmp.name) / "data", Path(self.tmp.name) / "out"
        make_split(self.root, "train", [(20, 16), (16, 20), (24, 18), (18, 24)])
        make_split(self.root, "val", [(13, 9), (8, 6), (10, 10)])

    def tearDown(self):
        self.tmp.cleanup()

    def run_train(self, *extra, epochs="2"):
        argv = ["train.py", "--data-root", str(self.root), "--output-dir", str(self.out), "--num-epochs", epochs,
                "--batch-size", "2", "--num-workers", "0", "--save-every", "1", "--validate-every", "1",
                "--dice-weight", "0.5", "--freeze-text-encoder", "--no-freeze-vision-encoder", *extra]
        with patch.object(sys, "argv", argv), patch.object(train, "ModelConfig", TinyConfig):
            train.main()

    def test_records_and_eval_agreement(self):
        self.run_train("--expect", "dice_weight=0.5", "--expect", "decoder=fusion", "--expect", "attention_residual=true")
        history = [json.loads(line) for line in (self.out / "history.jsonl").read_text().splitlines()]
        self.assertEqual([h["epoch"] for h in history], [1, 2])
        self.assertTrue(all(h["training_config"]["freeze_text_encoder"] is True for h in history))
        self.assertIn("data_wait_seconds", history[0])
        curve = [json.loads(line) for line in (self.out / "val_curve.jsonl").read_text().splitlines()]
        checkpoint = torch.load(self.out / "checkpoint_epoch2.pt", map_location="cpu", weights_only=False)
        self.assertTrue(checkpoint["model_config"]["tiny"])
        self.assertIn("commit", checkpoint["run_config"]["git"])
        evaluator.main(["--checkpoint", str(self.out / "checkpoint_epoch2.pt"), "--split", "val", "--data-root",
                        str(self.root), "--output-dir", str(self.out / "eval"), "--num-workers", "0", "--save-masks"])
        result = json.loads((self.out / "eval" / "metrics.json").read_text())
        self.assertEqual(result["summary"]["mean_iou"], curve[-1]["val_miou"])  # logged validation == eval.py
        self.assertEqual(result["summary"]["recipe"]["dice_weight"], 0.5)
        with Image.open(self.out / "eval" / "masks" / "val_0.png") as mask:
            self.assertEqual(mask.size, (13, 9))
        with self.assertRaises(FileExistsError):
            evaluator.main(["--checkpoint", str(self.out / "checkpoint_epoch2.pt"), "--data-root", str(self.root),
                            "--output-dir", str(self.out / "eval"), "--num-workers", "0"])

    def test_expect_mismatch_stops_before_training(self):
        with self.assertRaisesRegex(ValueError, "nothing was trained"):
            self.run_train("--expect", "dice_weight=0")
        self.assertFalse((self.out / "history.jsonl").exists())

    def test_resume(self):
        self.run_train(epochs="1")
        self.run_train("--resume-checkpoint", str(self.out / "checkpoint_epoch1.pt"), epochs="2")
        self.assertEqual([json.loads(l)["epoch"] for l in (self.out / "history.jsonl").read_text().splitlines()], [1, 2])
        with self.assertRaisesRegex(ValueError, "dice_weight"):
            argv_extra = ("--resume-checkpoint", str(self.out / "checkpoint_epoch1.pt"))
            with patch.object(sys, "argv", ["train.py", "--data-root", str(self.root), "--output-dir", str(self.out),
                                            "--num-epochs", "3", "--batch-size", "2", "--num-workers", "0",
                                            "--dice-weight", "1.0", "--freeze-text-encoder",
                                            "--no-freeze-vision-encoder", *argv_extra]), \
                    patch.object(train, "ModelConfig", TinyConfig):
                train.main()


if __name__ == "__main__":
    unittest.main()
