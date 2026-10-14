"""Diagnostic dataflow and native-metric checks; no pretrained accuracy claims."""
import csv
import json
import re
import tempfile
from pathlib import Path
import unittest

import torch
from PIL import Image

from diagnose import (MODES, diagnostic_views, load_diagnostic_model, main,
                      paired_rows, score_summary, select_cases)
from models.model import GroundingModel


class LatentDiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(42)
        self.model = GroundingModel(tiny=True).eval()
        self.batch = {"image": torch.rand(2, 3, 32, 32),
                      "text": ["Q: color?", "Q: object?"],
                      "question_text": ["Q: color?", "Q: object?"],
                      "answer_only_text": ["A: red", "A: cup"],
                      "answer_available": torch.ones(2, dtype=torch.bool)}

    def test_views_match_inference_and_do_not_leak_real_answers(self):
        before = {k: v.clone() for k, v in self.model.state_dict().items()}
        views = diagnostic_views(self.model, self.batch)
        torch.testing.assert_close(views["logits"]["predicted_answer"], self.model(self.batch)["logits"], rtol=0, atol=0)
        changed = diagnostic_views(self.model, dict(self.batch, answer_only_text=["A: blue", "A: dog"]))
        for mode in ("question", "predicted_answer"):
            torch.testing.assert_close(views["logits"][mode], changed["logits"][mode], rtol=0, atol=0)
        self.assertGreater((views["logits"]["oracle_answer"] - changed["logits"]["oracle_answer"]).abs().max().item(), 0)
        self.assertGreater((views["cosine"] - changed["cosine"]).abs().max().item(), 0)
        for key, value in self.model.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)
        self.assertFalse(any(p.grad is not None for p in self.model.parameters()))
        self.model.train()
        with self.assertRaisesRegex(ValueError, "eval"):
            diagnostic_views(self.model, self.batch)

    def test_native_metrics_missing_answers_and_export_dimensions(self):
        batch = dict(self.batch, filename=["a.jpg", "b.jpg"],
                     answer_available=torch.tensor([True, False]),
                     original_mask=[torch.ones(1, 3, 5), torch.zeros(1, 5, 2)])
        views = {"logits": {"question": torch.full((2, 1, 2, 2), -2.),
                            "predicted_answer": torch.full((2, 1, 2, 2), 2.),
                            "oracle_answer": torch.full((2, 1, 2, 2), 2.)},
                 "cosine": torch.tensor([.8, .99]), "smooth_l1": torch.tensor([.1, .2]),
                 "latent_norm": torch.ones(2), "target_norm": torch.ones(2)}
        records = {"a.jpg": {"question": "color", "most_common_answer": "red"}, "b.jpg": {}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = paired_rows(views, batch, records, root)
            self.assertEqual(rows[0]["iou_predicted_answer"], 1)
            self.assertEqual(rows[1]["iou_question"], 1)  # empty union convention
            self.assertIsNone(rows[1]["iou_oracle_answer"])
            self.assertIsNone(rows[1]["answer_cosine"])
            summary = score_summary(rows)
            self.assertEqual(summary["all_samples"]["mean_iou_question"], .5)
            self.assertEqual(summary["all_samples"]["mean_iou_predicted_answer"], .5)
            self.assertEqual(summary["answer_available_subset"]["iou_oracle_answer"], 1)
            self.assertEqual(summary["num_with_answer"], 1)
            with Image.open(root / "predicted_answer" / "b.png") as image:
                self.assertEqual(image.size, (2, 5))
            self.assertFalse((root / "oracle_answer" / "b.png").exists())
        selections = select_cases(rows, 4, 42)
        self.assertEqual(selections["predicted_helps"], ["a.jpg"])
        self.assertEqual(selections["predicted_hurts"], ["b.jpg"])
        self.assertEqual(selections, select_cases(rows, 4, 42))
        mismatch = dict(rows[0], answer_cosine=.9, iou_predicted_answer=.2)
        self.assertEqual(select_cases([mismatch], 1, 42)["high_alignment_low_iou"], ["a.jpg"])

    def test_cli_whole_split_gallery_and_existing_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "val").mkdir()
            (root / "binary_masks_png" / "val").mkdir(parents=True)
            records = {}
            for i, (w, h, answer) in enumerate([(48, 32, "red"), (32, 48, ""), (40, 40, "cup")]):
                name = f"image{i}.jpg"
                Image.new("RGB", (w, h), (30 * i, 100, 200)).save(root / "val" / name)
                Image.new("L", (w, h), 255 if i != 1 else 0).save(root / "binary_masks_png" / "val" / f"image{i}.png")
                records[name] = {"question": "What <is> this?", "most_common_answer": answer}
            (root / "val_grounding.json").write_text(json.dumps(records))
            checkpoint = root / "old_checkpoint.pt"
            torch.save({"model_state_dict": self.model.state_dict(),
                        "experiment_config": self.model.experiment_config,
                        "run_config": {"tiny": True, "image_size": 32}, "epoch": 20}, checkpoint)
            review = root / "review.txt"
            review.write_text("image1.jpg\n")
            out = root / "diagnostic"
            argv = ["--checkpoint", str(checkpoint), "--data-root", str(root), "--dataset", "val",
                    "--batch-size", "2", "--num-workers", "0", "--device", "cpu",
                    "--output-dir", str(out), "--top-k", "2", "--save-masks", "--review-list", str(review)]
            main(argv)
            summary = json.loads((out / "summary.json").read_text())
            self.assertEqual(summary["num_samples"], 3)
            self.assertEqual(summary["num_with_answer"], 2)
            self.assertEqual(summary["objective_version"], "unknown (not stored)")
            rows = json.loads((out / "per_image.json").read_text())
            self.assertEqual([r["filename"] for r in rows], list(records))
            with (out / "per_image.csv").open() as f:
                self.assertEqual(len(list(csv.DictReader(f))), 3)
            self.assertIn("What &lt;is&gt; this?", (out / "gallery.html").read_text())
            self.assertTrue((out / "cases" / "00001_image.png").exists())
            self.assertFalse((out / "cases" / "00001_oracle_answer.png").exists())
            self.assertEqual(len(list((out / "masks" / "question").glob("*.png"))), 3)
            for relative in re.findall("src='([^']+)'", (out / "gallery.html").read_text()):
                self.assertTrue((out / relative).is_file(), relative)
            # Every displayed prediction is the native scored mask, overlaid for review.
            for row in rows:
                if not row["review_id"]:
                    continue
                prefix = out / "cases" / row["review_id"]
                with Image.open(f"{prefix}_image.png") as image, Image.open(
                        out / "masks" / "predicted_answer" / Path(row["filename"]).with_suffix(".png")) as mask:
                    red = Image.new("RGB", image.size, (255, 0, 0))
                    expected = Image.composite(Image.blend(image, red, .4), image,
                                               mask.resize(image.size, Image.Resampling.NEAREST))
                    with Image.open(f"{prefix}_predicted_answer.png") as actual:
                        self.assertEqual(actual.tobytes(), expected.tobytes())
            with self.assertRaises(SystemExit):
                main(argv)  # existing outputs must not be mixed
            # Raw legacy weights also load, without inventing their training-objective version.
            raw = root / "raw.pt"
            torch.save(self.model.state_dict(), raw)
            restored, metadata = load_diagnostic_model(raw, torch.device("cpu"))
            torch.testing.assert_close(restored(self.batch)["logits"], self.model(self.batch)["logits"], rtol=0, atol=0)
            self.assertEqual(metadata["objective_version"], "unknown (not stored)")


if __name__ == "__main__":
    unittest.main()
