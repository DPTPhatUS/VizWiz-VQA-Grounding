"""CPU-only ConvNeXt-Base image-backbone contracts; no pretrained weight downloads or GPU work."""

import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from PIL import Image
from test_ocrg import TinyConfig, _distributed_train_worker, free_port, make_split
from torch import nn

import eval as evaluator
import train
from dataset import CLIP_MEAN, CLIP_STD, CONVNEXT_IMAGE_SIZE, IMAGE_SIZE
from losses import segmentation_loss
from models.image_encoder import TinyConvNeXtVisionConfig, _TinyConvNeXtVision
from models.model import (
    CONVNEXT_NAME,
    CONVNEXT_REVISION,
    CONVNEXT_TOKENIZER_NAME,
    CONVNEXT_TOKENIZER_REVISION,
    TEXT_REVISION,
    GroundingModel,
    ModelConfig,
)
from models.text_encoder import _open_clip_to_hf_text_model

torch.set_num_threads(1)


class _DummyOpenCLIPBlock(nn.Module):
    def __init__(self, width: int, heads: int, mlp_width: int):
        super().__init__()
        self.ln_1 = nn.LayerNorm(width)
        self.attn = nn.MultiheadAttention(width, heads)
        self.ln_2 = nn.LayerNorm(width)
        self.mlp = nn.Module()
        self.mlp.c_fc = nn.Linear(width, mlp_width)
        self.mlp.c_proj = nn.Linear(mlp_width, width)


class _DummyOpenCLIPModel(nn.Module):
    def __init__(self, vocab_size: int = 128, width: int = 16, context_length: int = 77, layers: int = 2, heads: int = 2):
        super().__init__()
        self.visual = _TinyConvNeXtVision(TinyConvNeXtVisionConfig(stage_channels=(128, 256, 512, 1024)))
        self.token_embedding = nn.Embedding(vocab_size, width)
        self.positional_embedding = nn.Parameter(torch.randn(context_length, width) * 0.01)
        self.transformer = nn.Module()
        self.transformer.resblocks = nn.ModuleList(_DummyOpenCLIPBlock(width, heads, width * 4) for _ in range(layers))
        self.ln_final = nn.LayerNorm(width)


class ConvNeXtTests(unittest.TestCase):
    def test_pretrained_identity_and_legacy_defaults(self):
        config = ModelConfig(vision_backbone="convnext-base")
        self.assertEqual((config.vision_name, config.clip_name), (CONVNEXT_NAME, CONVNEXT_NAME))
        self.assertEqual((config.vision_revision, config.text_revision), (CONVNEXT_REVISION, CONVNEXT_REVISION))
        self.assertEqual(config.image_size, CONVNEXT_IMAGE_SIZE)
        self.assertEqual(CONVNEXT_IMAGE_SIZE, 384)
        self.assertEqual(ModelConfig(**json.loads(json.dumps(config.to_dict()))), config)
        legacy = ModelConfig()
        self.assertEqual(legacy.image_size, IMAGE_SIZE)
        old = {k: v for k, v in legacy.to_dict().items()
               if k not in ("vision_backbone", "vision_revision", "text_revision")}
        self.assertEqual(ModelConfig(**old), legacy)
        with self.assertRaisesRegex(ValueError, "paired CLIP text encoder"):
            ModelConfig(vision_backbone="convnext-base", clip_name="openai/clip-vit-base-patch16")
        with self.assertRaisesRegex(ValueError, "pair must share"):
            ModelConfig(vision_backbone="convnext-base", text_revision=TEXT_REVISION)
        with self.assertRaisesRegex(ValueError, "immutable"):
            ModelConfig(vision_backbone="convnext-base", vision_revision="main")

        # Exercise open_clip + hf_hub_download loader routing and text weight conversion.
        dummy_clip = _DummyOpenCLIPModel().eval()
        fake_open_clip = types.ModuleType("open_clip")
        fake_open_clip.create_model = MagicMock(return_value=dummy_clip)
        with (
            patch.dict(sys.modules, {"open_clip": fake_open_clip}),
            patch("huggingface_hub.hf_hub_download", return_value="/tmp/open_clip_pytorch_model.bin") as dl,
            patch("models.text_encoder.CLIPTokenizer.from_pretrained") as tokenizer,
        ):
            model = GroundingModel(config)
        self.assertEqual(dl.call_count, 2)
        dl.assert_called_with(CONVNEXT_NAME, "open_clip_pytorch_model.bin", revision=CONVNEXT_REVISION)
        self.assertEqual(fake_open_clip.create_model.call_count, 2)
        fake_open_clip.create_model.assert_called_with("convnext_base", pretrained="/tmp/open_clip_pytorch_model.bin")
        tokenizer.assert_called_once_with(CONVNEXT_TOKENIZER_NAME, revision=CONVNEXT_TOKENIZER_REVISION)
        self.assertIsInstance(model.image_encoder.vision_encoder.trunk.norm_pre, nn.Identity)
        self.assertIsInstance(model.image_encoder.vision_encoder.trunk.head, nn.Identity)
        self.assertIsInstance(model.image_encoder.vision_encoder.head, nn.Identity)

        # Verify numerical equivalence of _open_clip_to_hf_text_model against manual GELU forward.
        hf_text = _open_clip_to_hf_text_model(dummy_clip).eval()
        input_ids = torch.tensor([[1, 5, 9, 2, 0, 0], [1, 3, 4, 7, 2, 0]])
        L = input_ids.shape[1]
        x = dummy_clip.token_embedding(input_ids) + dummy_clip.positional_embedding[:L]
        x = x.permute(1, 0, 2)
        causal = torch.triu(torch.full((L, L), float("-inf")), diagonal=1)
        for block in dummy_clip.transformer.resblocks:
            h = block.ln_1(x)
            a, _ = block.attn(h, h, h, need_weights=False, attn_mask=causal)
            x = x + a
            m = F.gelu(block.mlp.c_fc(block.ln_2(x)))
            x = x + block.mlp.c_proj(m)
        expected = dummy_clip.ln_final(x.permute(1, 0, 2))
        with torch.no_grad():
            actual = hf_text(input_ids=input_ids).last_hidden_state
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)

    def test_pyramid_features_hierarchical_decoder_and_gradients(self):
        self.assertEqual(CLIP_MEAN, (0.48145466, 0.4578275, 0.40821073))
        self.assertEqual(CLIP_STD, (0.26862954, 0.26130258, 0.27577711))
        torch.manual_seed(0)
        model = GroundingModel(TinyConfig(vision_backbone="convnext-base"))
        model.text_encoder.requires_grad_(False)
        self.assertIsInstance(model.image_encoder.vision_encoder.trunk.norm_pre, nn.Identity)
        self.assertIsInstance(model.image_encoder.vision_encoder.trunk.head, nn.Identity)
        self.assertIsInstance(model.image_encoder.vision_encoder.head, nn.Identity)
        images = torch.randn(2, 3, 384, 384)
        features = model.image_encoder(images)
        self.assertEqual(
            [tuple(f.shape) for f in features],
            [(2, 8, 96, 96), (2, 16, 48, 48), (2, 24, 24, 24), (2, 32, 12, 12)],
        )
        self.assertIsInstance(model.text_proj, torch.nn.Linear)
        self.assertEqual((model.text_proj.in_features, model.text_proj.out_features), (16, 32))
        self.assertEqual([block.text_proj.in_features for block in model.decoder.stage_attention], [16] * 3)
        self.assertTrue(model.decoder.hierarchical_skips)
        self.assertEqual(
            [(proj.in_channels, proj.out_channels) for proj in model.decoder.skip_proj],
            [(24, 256), (16, 128), (8, 64)],
        )
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
        self.assertTrue(all(p.grad is None for p in model.text_encoder.parameters()))
        self.assertGreater(model.text_proj.weight.grad.abs().sum().item(), 0)

    def test_train_reload_native_evaluation_predict_and_resume_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, out = Path(tmp) / "data", Path(tmp) / "out"
            make_split(root, "train", [(16, 20), (20, 16)])
            make_split(root, "val", [(13, 9), (8, 6), (10, 10)])
            argv = ["train.py", "--vision-backbone", "convnext-base", "--data-root", str(root),
                    "--output-dir", str(out), "--num-epochs", "1", "--batch-size", "2", "--num-workers", "0",
                    "--validate-every", "1", "--dice-weight", "0.5", "--freeze-text-encoder",
                    "--no-freeze-vision-encoder", "--optimizer", "adamw", "--weight-decay", "0.05",
                    "--lr", "1e-4", "--backbone-lr", "1e-5", "--lr-scheduler", "cosine",
                    "--warmup-ratio", "0.05", "--grad-clip-norm", "1.0", "--expect", "vision_backbone=convnext-base"]
            with patch.object(sys, "argv", argv), patch.object(train, "ModelConfig", TinyConfig):
                train.main()
            path = out / "checkpoint_epoch1.pt"
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            self.assertEqual(checkpoint["model_config"]["vision_backbone"], "convnext-base")
            model, _ = GroundingModel.from_checkpoint(path)
            self.assertTrue(model.decoder.hierarchical_skips)
            with patch.object(model, "forward", wraps=model.forward) as forward:
                self.assertEqual(model.predict(Image.new("RGB", (13, 9)), "What is this?").shape, (9, 13))
                self.assertEqual(forward.call_args.args[0].shape, (1, 3, 384, 384))
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
            mismatch = [a if a != "convnext-base" else "clip-vit-l14-336" for a in argv]
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

    def test_two_rank_training_with_convnext_pyramid(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, out = Path(tmp) / "data", Path(tmp) / "out"
            make_split(root, "train", [(16, 20), (20, 16), (10, 10), (12, 8)])
            make_split(root, "val", [(13, 9), (8, 6), (10, 10)])
            argv = ["train.py", "--vision-backbone", "convnext-base", "--data-root", str(root),
                    "--output-dir", str(out), "--num-epochs", "1", "--batch-size", "2", "--num-workers", "0",
                    "--validate-every", "1", "--dice-weight", "0.5", "--freeze-text-encoder",
                    "--no-freeze-vision-encoder"]
            mp.spawn(_distributed_train_worker, args=(2, free_port(), argv), nprocs=2)
            curve = json.loads((out / "val_curve.jsonl").read_text().splitlines()[0])
            self.assertEqual(curve["num_images"], 3)
            self.assertTrue((out / "checkpoint_epoch1.pt").exists())


if __name__ == "__main__":
    unittest.main()
