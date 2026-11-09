"""CPU-only image-backbone contracts; no pretrained weight downloads or GPU work."""

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
from transformers import CLIPTextModel, CLIPVisionModel

import eval as evaluator
import train
from losses import segmentation_loss
from models.model import (
    CLIP_NAME,
    TEXT_REVISION,
    VITB16_NAME,
    VITB16_REVISION,
    GroundingModel,
    ModelConfig,
    _tiny_configs,
)

torch.set_num_threads(1)


class ViTB16Tests(unittest.TestCase):
    def test_pretrained_identity_and_legacy_defaults(self):
        config = ModelConfig(vision_backbone="clip-vit-b16")
        self.assertEqual((config.vision_name, config.clip_name), (VITB16_NAME, CLIP_NAME))
        self.assertEqual((config.vision_revision, config.text_revision), (VITB16_REVISION, TEXT_REVISION))
        self.assertEqual(ModelConfig(**json.loads(json.dumps(config.to_dict()))), config)
        legacy = ModelConfig()
        old = {k: v for k, v in legacy.to_dict().items()
               if k not in ("vision_backbone", "vision_revision", "text_revision")}
        self.assertEqual(ModelConfig(**old), legacy)
        with self.assertRaisesRegex(ValueError, "retains the original"):
            ModelConfig(vision_backbone="clip-vit-b16", clip_name=VITB16_NAME)
        with self.assertRaisesRegex(ValueError, "immutable"):
            ModelConfig(vision_backbone="clip-vit-b16", vision_revision="main")
        # Exercise actual from_pretrained argument routing with tiny substitutes.
        vision, text = _tiny_configs(True)
        with (
            patch("models.image_encoder.CLIPVisionModel.from_pretrained", return_value=CLIPVisionModel(vision)) as vload,
            patch("models.text_encoder.CLIPTextModel.from_pretrained", return_value=CLIPTextModel(text)) as tload,
            patch("models.text_encoder.CLIPTokenizer.from_pretrained") as tokenizer,
        ):
            GroundingModel(config)
        vload.assert_called_once_with(VITB16_NAME, revision=VITB16_REVISION)
        tload.assert_called_once_with(CLIP_NAME, revision=TEXT_REVISION)
        tokenizer.assert_called_once_with(CLIP_NAME, revision=TEXT_REVISION)

    def test_interpolated_features_projection_and_gradients(self):
        torch.manual_seed(0)
        model = GroundingModel(TinyConfig(vision_backbone="clip-vit-b16"))
        model.text_encoder.requires_grad_(False)
        images = torch.randn(2, 3, 336, 336)
        features = model.image_encoder(images)
        self.assertEqual([tuple(f.shape) for f in features], [(2, 32, 21, 21)] * 4)
        # The unchanged checkpoint position table is 14x14 + CLS, not reinitialized at21x21.
        position = next(p for n, p in model.image_encoder.named_parameters() if "position_embedding.weight" in n)
        self.assertEqual(position.shape[0], 197)
        raw = model.image_encoder.vision_encoder(images, output_hidden_states=True, interpolate_pos_encoding=True)
        for actual, index in zip(features, (4, 7, 9, 10), strict=True):
            torch.testing.assert_close(actual, model.image_encoder._to_map(raw.hidden_states[index]), rtol=0, atol=0)
        self.assertIsInstance(model.text_proj, torch.nn.Linear)
        self.assertEqual(model.text_proj.in_features, model.text_proj.out_features)
        stages = []
        hooks = [stage.register_forward_hook(lambda module, inputs, output: stages.append(tuple(output.shape[-2:])))
                 for stage in model.decoder.stages]
        logits = model(images, ["Q: what is this?", "Q: what text is on the box?"])
        for hook in hooks:
            hook.remove()
        self.assertEqual(logits.shape, (2, 1, 336, 336))
        self.assertEqual(stages, [(42, 42), (84, 84), (168, 168), (336, 336)])
        segmentation_loss(logits, torch.ones_like(logits), 0.5).backward()
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        self.assertGreater(position.grad.abs().sum().item(), 0)
        self.assertTrue(all(p.grad is None for p in model.text_encoder.parameters()))
        self.assertGreater(model.text_proj.weight.grad.abs().sum().item(), 0)
        # Without explicit interpolation, the pretrained224 position contract rejects336.
        with self.assertRaises(ValueError):
            model.image_encoder.vision_encoder(images)

    def test_train_reload_native_evaluation_predict_and_resume_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, out = Path(tmp) / "data", Path(tmp) / "out"
            make_split(root, "train", [(16, 20), (20, 16)])
            make_split(root, "val", [(13, 9), (8, 6), (10, 10)])
            argv = ["train.py", "--vision-backbone", "clip-vit-b16", "--data-root", str(root),
                    "--output-dir", str(out), "--num-epochs", "1", "--batch-size", "2", "--num-workers", "0",
                    "--validate-every", "1", "--dice-weight", "0.5", "--freeze-text-encoder",
                    "--no-freeze-vision-encoder", "--optimizer", "adamw", "--weight-decay", "0.05",
                    "--lr", "1e-4", "--backbone-lr", "1e-5", "--lr-scheduler", "cosine",
                    "--warmup-ratio", "0.05", "--grad-clip-norm", "1.0", "--expect", "vision_backbone=clip-vit-b16"]
            with patch.object(sys, "argv", argv), patch.object(train, "ModelConfig", TinyConfig):
                train.main()
            path = out / "checkpoint_epoch1.pt"
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            self.assertEqual(checkpoint["model_config"]["vision_backbone"], "clip-vit-b16")
            model, _ = GroundingModel.from_checkpoint(path)
            self.assertTrue(model.image_encoder.interpolate_pos_encoding)
            with patch.object(model, "forward", wraps=model.forward) as forward:
                self.assertEqual(model.predict(Image.new("RGB", (13, 9)), "What is this?").shape, (9, 13))
                self.assertEqual(forward.call_args.args[0].shape, (1, 3, 336, 336))
            evaluator.main(["--checkpoint", str(path), "--data-root", str(root), "--split", "val",
                            "--output-dir", str(out / "eval"), "--num-workers", "0", "--batch-size", "4", "--save-masks"])
            scores = json.loads((out / "eval/metrics.json").read_text())
            curve = json.loads((out / "val_curve.jsonl").read_text().splitlines()[0])
            self.assertEqual(scores["summary"]["mean_iou"], curve["val_miou"])
            self.assertEqual(len(scores["per_sample"]), 3)
            with Image.open(out / "eval/masks/val_0.png") as mask:
                self.assertEqual(mask.size, (13, 9))
            # Exact snapshot reconstruction and no checkpoint mutation.
            self.assertEqual(set(model.state_dict()), set(checkpoint["model_state_dict"]))
            for name, value in model.state_dict().items():
                torch.testing.assert_close(value, checkpoint["model_state_dict"][name], rtol=0, atol=0)
            mismatch = [a if a != "clip-vit-b16" else "clip-vit-l14-336" for a in argv]
            mismatch = mismatch[:mismatch.index("--expect")] + ["--resume-checkpoint", str(path)]
            with (patch.object(sys, "argv", mismatch), patch.object(train, "ModelConfig", TinyConfig),
                  self.assertRaisesRegex(ValueError, "model_config differs")):
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

    def test_two_rank_training_with_interpolated_positions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, out = Path(tmp) / "data", Path(tmp) / "out"
            make_split(root, "train", [(16, 20), (20, 16), (10, 10), (12, 8)])
            make_split(root, "val", [(13, 9), (8, 6), (10, 10)])
            argv = ["train.py", "--vision-backbone", "clip-vit-b16", "--data-root", str(root),
                    "--output-dir", str(out), "--num-epochs", "1", "--batch-size", "2", "--num-workers", "0",
                    "--validate-every", "1", "--dice-weight", "0.5", "--freeze-text-encoder",
                    "--no-freeze-vision-encoder"]
            mp.spawn(_distributed_train_worker, args=(2, free_port(), argv), nprocs=2)
            curve = json.loads((out / "val_curve.jsonl").read_text().splitlines()[0])
            self.assertEqual(curve["num_images"], 3)
            self.assertTrue((out / "checkpoint_epoch1.pt").exists())


if __name__ == "__main__":
    unittest.main()
