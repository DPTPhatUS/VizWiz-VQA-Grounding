"""Offline integration checks using the real NaFlex processor and a tiny ViT.

These validate geometry/checkpoints/gradients, not pretrained accuracy or CUDA.
"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from torch import nn
from PIL import Image, ImageDraw
from transformers import (CLIPTextConfig, CLIPTextModel, Siglip2Config, Siglip2Model,
                          Siglip2TextConfig, Siglip2ImageProcessor, Siglip2VisionConfig)

from dataset import VizWizGroundingDataset, collate_eval_samples
from models.image_encoder import BASE_NAFLEX, ImageEncoder, make_naflex_processor
from models import GroundingModel
from utils import grounding_loss, to_device
import eval as evaluator
import train as trainer

torch.set_num_threads(1)


class TinyText(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.model = CLIPTextModel(CLIPTextConfig(vocab_size=32, hidden_size=16,
            intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
            max_position_embeddings=12, bos_token_id=1, eos_token_id=2, pad_token_id=0))
        self.output_dim = 16

    def forward(self, texts, return_mask=False):
        device = next(self.parameters()).device
        ids = torch.zeros(len(texts), 12, dtype=torch.long, device=device)
        for i, text in enumerate(texts):
            values = [1] + [3 + ord(c) % 29 for c in text[:8]] + [2]
            ids[i, :len(values)] = torch.tensor(values, device=device)
        mask = ids != 0
        output = self.model(input_ids=ids, attention_mask=mask.long()).last_hidden_state
        return (output, ~mask) if return_mask else output


def tiny_model(max_num_patches=24, **kwargs):
    config = Siglip2VisionConfig(hidden_size=16, intermediate_size=32, num_hidden_layers=3,
                               num_attention_heads=2, num_patches=16, patch_size=16)
    return GroundingModel(vision_model=BASE_NAFLEX, max_num_patches=max_num_patches,
        pretrained=False, vision_config=config.to_dict(), text_encoder=TinyText(),
        image_processor=Siglip2ImageProcessor().to_dict(), feature_layers=(1, 2, 3),
        n_heads=2, decoder_channels=(16, 12, 8, 4))


class NaFlexBaselineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.sizes = {"landscape.jpg": (120, 40), "portrait.jpg": (40, 120), "square.jpg": (80, 80)}
        for split in ("train", "val"):
            (self.root / split).mkdir()
            (self.root / "binary_masks_png" / split).mkdir(parents=True)
            records = {}
            for name, size in self.sizes.items():
                Image.new("RGB", size, "white").save(self.root / split / name)
                mask = Image.new("L", size, 0)
                ImageDraw.Draw(mask).rectangle((size[0] // 2, 0, size[0] - 1, size[1] - 1), fill=255)
                mask.save(self.root / "binary_masks_png" / split / Path(name).with_suffix(".png"))
                records[name] = {"question": "Which side?", "most_common_answer": "right"}
            (self.root / f"{split}_grounding.json").write_text(json.dumps(records))

    def tearDown(self):
        self.tmp.cleanup()

    def dataset(self, **kwargs):
        return VizWizGroundingDataset(self.root / "val_grounding.json", self.root / "val",
            self.root / "binary_masks_png" / "val", is_test=True,
            naflex_processor=Siglip2ImageProcessor(max_num_patches=24), **kwargs)

    def test_processor_mask_alignment_and_variable_collation(self):
        dataset = self.dataset(return_original_mask=True)
        samples = [dataset[i] for i in range(3)]
        batch = collate_eval_samples(samples)
        self.assertEqual(tuple(batch["image"]["pixel_values"].shape), (3, 24, 768))
        grids = batch["image"]["spatial_shapes"].tolist()
        self.assertGreater(grids[0][1], grids[0][0])
        self.assertGreater(grids[1][0], grids[1][1])
        for sample, (h, w) in zip(samples, grids):
            self.assertEqual(tuple(sample["mask"].shape), (1, 16 * h, 16 * w))
            self.assertLessEqual(h * w, 24)
            self.assertTrue(torch.all(sample["image"]["pixel_values"][:h * w] == 1))
            self.assertEqual(set(sample["mask"].unique().tolist()), {0.0, 1.0})
            self.assertLess(sample["mask"][:, :, :w * 8].mean().item(), 0.1)
            self.assertGreater(sample["mask"][:, :, w * 8:].mean().item(), 0.9)
        self.assertEqual([tuple(m.shape) for m in batch["original_mask"]], [(1, 40, 120), (1, 120, 40), (1, 80, 80)])

    def test_siglip_normalization_once_and_clip_stats_rejected(self):
        model = tiny_model()
        processor = make_naflex_processor(model.model_config)
        for value in (0, 64, 255):
            inputs = processor(images=Image.new("RGB", (120, 40), (value,) * 3), return_tensors="pt")
            valid = inputs["pixel_values"][inputs["pixel_attention_mask"].bool()]
            self.assertTrue(torch.allclose(valid, torch.full_like(valid, 2 * value / 255 - 1), atol=1e-6))
        bad = dict(model.model_config["image_processor"], image_mean=[0.48145466, 0.4578275, 0.40821073])
        with self.assertRaisesRegex(ValueError, "mean/std"):
            make_naflex_processor({"image_processor": bad})
        dataset = self.dataset()
        self.assertIsNone(dataset.normalize)
        self.assertEqual(dataset[0]["text"], "Q: Which side? A: right")

    def test_padding_is_ignored_and_grids_are_not_transposed(self):
        model = tiny_model().eval()
        batch = collate_eval_samples([self.dataset()[i] for i in (0, 1, 0)])
        with torch.no_grad():
            expected = model(batch["image"], batch["text"])
            changed = {k: v.clone() for k, v in batch["image"].items()}
            changed["pixel_values"][~changed["pixel_attention_mask"].bool()] = 1000
            actual = model(changed, batch["text"])
        for i, (h, w) in enumerate(batch["image"]["spatial_shapes"].tolist()):
            self.assertEqual(tuple(expected[i].shape), (1, 1, h * 16, w * 16))
            self.assertTrue(torch.allclose(expected[i], actual[i], atol=1e-6))
        self.assertTrue(torch.equal(expected[0], expected[2]))
        changed["pixel_attention_mask"][0, 0] = 0
        with self.assertRaisesRegex(ValueError, "contiguous prefix"):
            model(changed, batch["text"])

    def test_real_vision_gradients_and_equal_image_weighting(self):
        model = tiny_model().train()
        batch = to_device(collate_eval_samples([self.dataset()[i] for i in range(3)]), "cpu")
        output = model(batch["image"], batch["text"])
        loss = grounding_loss(output, batch["mask"], nn.BCEWithLogitsLoss())
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        grads = [p.grad for p in model.image_encoder.parameters() if p.requires_grad]
        self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in grads))
        self.assertTrue(any(g.abs().sum() > 0 for g in grads))
        values = [torch.full((1, 1, 2, 3), -1.0), torch.full((1, 1, 5, 8), 2.0)]
        masks = [torch.zeros(1, 2, 3), torch.ones(1, 5, 8)]
        expected = (nn.functional.softplus(torch.tensor(-1.0)) + nn.functional.softplus(torch.tensor(-2.0))) / 2
        self.assertTrue(torch.allclose(grounding_loss(values, masks, nn.BCEWithLogitsLoss()), expected))

    def test_checkpoint_roundtrip_and_native_predict(self):
        model = tiny_model().eval()
        checkpoint = self.root / "naflex.pt"
        torch.save({"model_state_dict": model.state_dict(), "model_config": model.model_config,
                    "text_mode": "question"}, checkpoint)
        batch = collate_eval_samples([self.dataset()[0], self.dataset()[1]])
        with patch("models.model.TextEncoder", TinyText):
            restored = GroundingModel.from_checkpoint(checkpoint, device="cpu")
        self.assertIsInstance(restored, GroundingModel)
        with torch.no_grad():
            for a, b in zip(model(batch["image"], batch["text"]), restored(batch["image"], batch["text"])):
                self.assertTrue(torch.equal(a, b))
        result = restored.predict(self.root / "val" / "portrait.jpg", question="Which side?",
                                  gt_mask=self.root / "binary_masks_png" / "val" / "portrait.png")
        self.assertEqual(result["pred_pil"].size, (40, 120))
        self.assertIsNotNone(result["iou"])

    def test_pretrained_vision_loader_extracts_full_siglip_bundle(self):
        vc = Siglip2VisionConfig(hidden_size=16, intermediate_size=32, num_hidden_layers=3,
                                num_attention_heads=2, num_patches=16, patch_size=16)
        tc = Siglip2TextConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
                              num_hidden_layers=1, num_attention_heads=2, max_position_embeddings=12,
                              bos_token_id=1, eos_token_id=2, pad_token_id=0)
        full = Siglip2Model(Siglip2Config(vision_config=vc.to_dict(), text_config=tc.to_dict())).eval()
        path = self.root / "pretrained_bundle"
        full.save_pretrained(path)
        extracted = ImageEncoder(str(path), config=vc.to_dict(), pretrained=True,
                                 image_processor=Siglip2ImageProcessor().to_dict(), feature_layers=(1, 2, 3)).eval()
        for key, value in extracted.vision_encoder.state_dict().items():
            self.assertTrue(torch.equal(value, full.vision_model.state_dict()[key]), key)
        batch = collate_eval_samples([self.dataset()[0]])
        with torch.no_grad():
            expected = full.vision_model(**batch["image"]).last_hidden_state
            actual = extracted(batch["image"])[-1]
        self.assertTrue(torch.equal(expected, actual))

    def test_train_eval_resume_and_legacy_export(self):
        out = self.root / "run"
        common = ["--data-root", str(self.root), "--batch-size", "3", "--num-workers", "0",
                  "--validate-every", "1", "--save-every", "1", "--output-dir", str(out)]
        with patch.object(trainer, "GroundingModel", tiny_model):
            trainer.main(common + ["--num-epochs", "1", "--max-num-patches", "24", "--text-mode", "question"])
        ckpt = out / "checkpoint_epoch1.pt"
        self.assertEqual(torch.load(ckpt, weights_only=False)["model_config"]["max_num_patches"], 24)
        with patch("models.model.TextEncoder", TinyText):
            trainer.main(common + ["--num-epochs", "2", "--resume-checkpoint", str(ckpt)])
            for resolution in ("original", "resized"):
                eval_out = self.root / resolution
                evaluator.main(["--checkpoint", str(out / "model_final_epoch2.pt"), "--data-root", str(self.root),
                    "--device", "cpu", "--batch-size", "3", "--num-workers", "0", "--output-dir", str(eval_out),
                    "--metric-resolution", resolution])
                results = json.loads(next(eval_out.glob("results_*.json")).read_text())
                self.assertEqual(results["summary"]["text_mode"], "question")
                self.assertEqual(results["summary"]["max_num_patches"], 24)
                for name, size in self.sizes.items():
                    with Image.open(eval_out / Path(name).with_suffix(".png")) as mask:
                        self.assertEqual(mask.size, size if resolution == "original" else (336, 336))
            with self.assertRaises(SystemExit):
                trainer.main(common + ["--num-epochs", "2", "--resume-checkpoint", str(ckpt), "--max-num-patches", "32"])


if __name__ == "__main__":
    unittest.main()
