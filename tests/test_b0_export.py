"""Offline CPU tests for deployment guards and native-mask verification."""

import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "export_b0", ROOT / "scripts/export_b0_validation_masks.py"
)
export = importlib.util.module_from_spec(spec)
spec.loader.exec_module(export)


class ExportTests(unittest.TestCase):
    def test_original_source_guard_rejects_backbone_branch(self):
        # This experiment deliberately extends model/train sources. The historical
        # exporter must reject it; do not update the original source manifest.
        manifest = export.load(ROOT / "scripts/b0_source_manifest.json")
        with self.assertRaisesRegex(ValueError, "Original B0 source changed"):
            export.verify_source()
        self.assertEqual(
            manifest["baseline_commit"], "0677214ffbc3e05659719f66d30db519509eba4c"
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "scripts").mkdir()
            (root / "dataset.py").write_text("changed")
            (root / "scripts/b0_source_manifest.json").write_text(
                json.dumps({"files": {"dataset.py": manifest["files"]["dataset.py"]}})
            )
            with self.assertRaisesRegex(ValueError, "source changed"):
                export.verify_source(root)

    def test_gpu_selection(self):
        self.assertEqual(export.gpu_count("auto", 1), 1)
        self.assertEqual(export.gpu_count("auto", 2), 2)
        with self.assertRaises(ValueError):
            export.gpu_count("auto", 0)
        with self.assertRaises(ValueError):
            export.gpu_count("2", 1)

    def test_native_verification_and_sealed_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data, output = root / "data", root / "output"
            for folder in (
                data / "val",
                data / "binary_masks_png/val",
                output / "masks",
            ):
                folder.mkdir(parents=True)
            records = {
                f"image_{i}.jpg": {"question": "What is this?"} for i in range(1131)
            }
            scores = {name: 1.0 for name in records}
            reference = {
                "summary": {
                    "split": "val",
                    "epoch": 20,
                    "num_samples": 1131,
                    "text_mode": "question",
                    "answer_in_input": False,
                    "recipe": {"seed": 41},
                    "model_config": {},
                    "mean_iou": 1.0,
                },
                "per_sample": scores,
            }
            (data / "val_grounding.json").write_text(json.dumps(records))
            (output / "metrics.json").write_text(json.dumps(reference))
            for name in records:
                Image.new("RGB", (5, 3)).save(data / "val" / name)
                filename = Path(name).stem + ".png"
                Image.new("L", (5, 3)).save(data / "binary_masks_png/val" / filename)
                Image.new("L", (5, 3)).save(output / "masks" / filename)
            result = export.verify_masks(data, output, reference, 41)
            self.assertEqual(result["count"], 1131)
            self.assertEqual(result["max_original_iou_error"], 0.0)
            archive = root / "reference.zip"
            with ZipFile(archive, "w") as z:
                z.writestr(
                    "ocrg_b0_s41/eval_val_ep20/metrics.json", json.dumps(reference)
                )
                z.writestr(
                    "ocrg_b0_s41/eval_test_ep20/metrics.json", "SEALED INVALID JSON"
                )
            self.assertEqual(export.reference(archive, 41), reference)
            Image.new("L", (5, 3), 255).save(output / "masks/image_0.png")
            with self.assertRaisesRegex(ValueError, "IoU mismatch"):
                export.verify_masks(data, output, reference, 41)
            wrong = copy.deepcopy(reference)
            wrong["summary"]["split"] = "test"
            with self.assertRaises(ValueError):
                export.validate_metrics(wrong, 41)
            Image.new("L", (3, 5)).save(output / "masks/image_0.png")
            with self.assertRaisesRegex(ValueError, "dimensions"):
                export.verify_masks(data, output, reference, 41)
            (output / "masks/image_0.png").unlink()
            with self.assertRaisesRegex(ValueError, "1,131"):
                export.verify_masks(data, output, reference, 41)

    def test_checkpoint_identity(self):
        ref = {
            "summary": {
                "recipe": {"seed": 41, "text_mode": "question"},
                "model_config": {},
            }
        }
        c = {
            "epoch": 20,
            "model_config": {},
            "model_state_dict": {"dummy": 1},
            "run_config": {
                "recipe": ref["summary"]["recipe"],
                "git": {"commit": "baseline", "dirty": False},
            },
        }
        export.validate_checkpoint(c, 41, ref, "baseline")
        c["epoch"] = 10
        with self.assertRaises(ValueError):
            export.validate_checkpoint(c, 41, ref, "baseline")


if __name__ == "__main__":
    unittest.main()
