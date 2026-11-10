"""CPU-only SigLIP2 Base/16-384 paired-backbone contracts; no pretrained weight downloads or GPU work."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import torch
import torch.multiprocessing as mp
from PIL import Image
from test_ocrg import TinyConfig, _distributed_train_worker, free_port, make_split
from transformers import GemmaTokenizer, SiglipTextModel, SiglipVisionModel

import eval as evaluator
import train
from dataset import (
    CLIP_MEAN,
    CLIP_STD,
    IMAGE_SIZE,
    SIGLIP_IMAGE_SIZE,
    SIGLIP_MEAN,
    SIGLIP_STD,
    VizWizGroundingDataset,
)
from losses import segmentation_loss
from models.model import (
    SIGLIP2_NAME,
    SIGLIP2_REVISION,
    TEXT_REVISION,
    GroundingModel,
    ModelConfig,
    _tiny_configs,
)

torch.set_num_threads(1)


class SigLIP2Tests(unittest.TestCase):
    def test_pretrained_identity_and_legacy_defaults(self):
        config = ModelConfig(vision_backbone="siglip2-b16-384")
        self.assertEqual((config.vision_name, config.clip_name), (SIGLIP2_NAME, SIGLIP2_NAME))
        self.assertEqual((config.vision_revision, config.text_revision), (SIGLIP2_REVISION, SIGLIP2_REVISION))
        self.assertEqual((config.image_size, config.image_mean, config.image_std),
                         (SIGLIP_IMAGE_SIZE, SIGLIP_MEAN, SIGLIP_STD))
        self.assertEqual(ModelConfig(**json.loads(json.dumps(config.to_dict()))), config)
        legacy = ModelConfig()
        self.assertEqual((legacy.image_size, legacy.image_mean, legacy.image_std),
                         (IMAGE_SIZE, CLIP_MEAN, CLIP_STD))
        old = {k: v for k, v in legacy.to_dict().items()
               if k not in ("vision_backbone", "vision_revision", "text_revision")}
        self.assertEqual(ModelConfig(**old), legacy)
        with self.assertRaisesRegex(ValueError, "pair must share"):
            ModelConfig(vision_backbone="siglip2-b16-384", text_revision=TEXT_REVISION)
        with self.assertRaisesRegex(ValueError, "immutable"):
            ModelConfig(vision_backbone="siglip2-b16-384", vision_revision="main")
        with self.assertRaisesRegex(ValueError, "paired SigLIP2 text encoder"):
            ModelConfig(vision_backbone="siglip2-b16-384", clip_name="openai/clip-vit-base-patch16")
        # Exercise actual from_pretrained argument routing with tiny substitutes.
        vision, text = _tiny_configs(True)
        with (
            patch("models.image_encoder.SiglipVisionModel.from_pretrained",
                  return_value=SiglipVisionModel(vision)) as vload,
            patch("models.text_encoder.SiglipTextModel.from_pretrained",
                  return_value=SiglipTextModel(text)) as tload,
            patch("models.text_encoder.AutoTokenizer.from_pretrained",
                  return_value=GemmaTokenizer(padding_side="right", add_bos_token=False, add_eos_token=True)) as tok,
        ):
            built = GroundingModel(config)
        vload.assert_called_once_with(SIGLIP2_NAME, revision=SIGLIP2_REVISION)
        tload.assert_called_once_with(SIGLIP2_NAME, revision=SIGLIP2_REVISION)
        tok.assert_called_once_with(SIGLIP2_NAME, revision=SIGLIP2_REVISION)
        self.assertFalse(built.image_encoder.vision_encoder.use_head)
        self.assertIsInstance(built.image_encoder.vision_encoder.head, torch.nn.Identity)
        self.assertFalse(any(p.requires_grad for p in built.text_encoder.model.head.parameters()))

    def test_native_384_features_projection_and_gradients(self):
        torch.manual_seed(0)
        model = GroundingModel(TinyConfig(vision_backbone="siglip2-b16-384"))
        model.text_encoder.requires_grad_(False)
        images = torch.randn(2, 3, 384, 384)
        features = model.image_encoder(images)
        self.assertEqual([tuple(f.shape) for f in features], [(2, 32, 24, 24)] * 4)
        # Native 384x384 with patch_size=16 yields 24x24 = 576 positions without a CLS token.
        position = next(p for n, p in model.image_encoder.named_parameters() if "position_embedding.weight" in n)
        self.assertEqual(position.shape[0], 576)
        raw = model.image_encoder.vision_encoder(images, output_hidden_states=True)
        for actual, expected in zip(
            features,
            (raw.hidden_states[4], raw.hidden_states[7], raw.hidden_states[9], raw.last_hidden_state),
            strict=True,
        ):
            torch.testing.assert_close(actual, model.image_encoder._to_map(expected), rtol=0, atol=0)
        self.assertIsInstance(model.text_proj, torch.nn.Linear)
        self.assertEqual((model.text_proj.in_features, model.text_proj.out_features), (16, 32))
        self.assertEqual([block.text_proj.in_features for block in model.decoder.stage_attention], [16] * 3)
        stages = []
        hooks = [stage.register_forward_hook(lambda module, inputs, output: stages.append(tuple(output.shape[-2:])))
                 for stage in model.decoder.stages]
        logits = model(images, ["Q: what is this?", "Q: what text is on the box?"])
        for hook in hooks:
            hook.remove()
        self.assertEqual(logits.shape, (2, 1, 384, 384))
        self.assertEqual(stages, [(48, 48), (96, 96), (192, 192), (384, 384)])
        segmentation_loss(logits, torch.ones_like(logits), 0.5).backward()
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        self.assertGreater(position.grad.abs().sum().item(), 0)
        self.assertTrue(all(p.grad is None for p in model.text_encoder.parameters()))
        self.assertGreater(model.text_proj.weight.grad.abs().sum().item(), 0)
        # Native 384px SigLIP2 rejects 336px inputs instead of silently interpolating.
        with self.assertRaises(RuntimeError):
            model.image_encoder.vision_encoder(torch.randn(1, 3, 336, 336))

    def test_train_reload_native_evaluation_predict_and_resume_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, out = Path(tmp) / "data", Path(tmp) / "out"
            make_split(root, "train", [(16, 20), (20, 16)])
            make_split(root, "val", [(13, 9), (8, 6), (10, 10)])
            ds = VizWizGroundingDataset(root, "train", image_size=SIGLIP_IMAGE_SIZE,
                                        mean=SIGLIP_MEAN, std=SIGLIP_STD)
            sample = ds[0]
            self.assertEqual(tuple(sample["image"].shape), (3, 384, 384))
            self.assertEqual(tuple(sample["mask"].shape), (1, 384, 384))
            argv = ["train.py", "--vision-backbone", "siglip2-b16-384", "--data-root", str(root),
                    "--output-dir", str(out), "--num-epochs", "1", "--batch-size", "2", "--num-workers", "0",
                    "--validate-every", "1", "--dice-weight", "0.5", "--freeze-text-encoder",
                    "--no-freeze-vision-encoder", "--optimizer", "adamw", "--weight-decay", "0.05",
                    "--lr", "1e-4", "--backbone-lr", "1e-5", "--lr-scheduler", "cosine",
                    "--warmup-ratio", "0.05", "--grad-clip-norm", "1.0",
                    "--expect", "vision_backbone=siglip2-b16-384"]
            with patch.object(sys, "argv", argv), patch.object(train, "ModelConfig", TinyConfig):
                train.main()
            path = out / "checkpoint_epoch1.pt"
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            self.assertEqual(checkpoint["model_config"]["vision_backbone"], "siglip2-b16-384")
            model, _ = GroundingModel.from_checkpoint(path)
            self.assertEqual(model.config.image_size, 384)
            with patch.object(model, "forward", wraps=model.forward) as forward:
                pil = Image.new("RGB", (13, 9), (255, 0, 128))
                self.assertEqual(model.predict(pil, "What is this?").shape, (9, 13))
                fed = forward.call_args.args[0]
                self.assertEqual(fed.shape, (1, 3, 384, 384))
                self.assertAlmostEqual(fed[0, 0, 0, 0].item(), 1.0, places=5)
                self.assertAlmostEqual(fed[0, 1, 0, 0].item(), -1.0, places=5)
            evaluator.main(["--checkpoint", str(path), "--data-root", str(root), "--split", "val",
                            "--output-dir", str(out / "eval"), "--num-workers", "0", "--batch-size", "4", "--save-masks"])
            scores = json.loads((out / "eval/metrics.json").read_text())
            curve = json.loads((out / "val_curve.jsonl").read_text().splitlines()[0])
            self.assertEqual(scores["summary"]["mean_iou"], curve["val_miou"])
            self.assertEqual(len(scores["per_sample"]), 3)
            for name, size in (("val_0.png", (13, 9)), ("val_1.png", (8, 6)), ("val_2.png", (10, 10))):
                with Image.open(out / "eval/masks" / name) as mask:
                    self.assertEqual(mask.size, size)
            # Exact snapshot reconstruction and no checkpoint mutation.
            self.assertEqual(set(model.state_dict()), set(checkpoint["model_state_dict"]))
            for name, value in model.state_dict().items():
                torch.testing.assert_close(value, checkpoint["model_state_dict"][name], rtol=0, atol=0)
            mismatch = [a if a != "siglip2-b16-384" else "clip-vit-l14-336" for a in argv]
            mismatch = mismatch[:mismatch.index("--expect")] + ["--resume-checkpoint", str(path)]
            with (
                patch.object(sys, "argv", mismatch),
                patch.object(train, "ModelConfig", TinyConfig),
                self.assertRaisesRegex(ValueError, "model_config differs"),
            ):
                train.main()
            # A legacy config without extension fields still reconstructs and resumes.
            legacy = GroundingModel(TinyConfig())
            saved = {"model_config": {k: v for k, v in legacy.config.to_dict().items()
                                      if k not in ("vision_backbone", "vision_revision", "text_revision")},
                     "model_state_dict": legacy.state_dict()}
            old_path = Path(tmp) / "old.pt"
            torch.save(saved, old_path)
            restored, _ = GroundingModel.from_checkpoint(old_path)
            with torch.no_grad():
                image = torch.randn(1, 3, 336, 336)
                torch.testing.assert_close(restored(image, ["Q: item?"]), legacy.eval()(image, ["Q: item?"]), rtol=0, atol=0)

    def test_two_rank_distributed_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, out = Path(tmp) / "data", Path(tmp) / "out"
            make_split(root, "train", [(16, 16), (16, 16), (16, 16), (16, 16)])
            make_split(root, "val", [(13, 9), (8, 6), (10, 10)])
            argv = ["train.py", "--vision-backbone", "siglip2-b16-384", "--data-root", str(root),
                    "--output-dir", str(out), "--num-epochs", "1", "--batch-size", "2", "--num-workers", "0",
                    "--validate-every", "1", "--dice-weight", "0.5", "--freeze-text-encoder",
                    "--no-freeze-vision-encoder"]
            mp.spawn(_distributed_train_worker, args=(2, free_port(), argv), nprocs=2)
            curve = json.loads((out / "val_curve.jsonl").read_text().splitlines()[0])
            self.assertEqual(curve["num_images"], 3)
            self.assertTrue((out / "checkpoint_epoch1.pt").exists())


if __name__ == "__main__":
    unittest.main()
