"""CPU synthetic correctness checks. No real-parent training, probes or downloads."""

import copy
import hashlib
import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from zipfile import ZipFile

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from PIL import Image

from b1_data import OCRDataset, build_features, candidates, ocr_collate
from b1_run import (
    evaluate_snapshot,
    fresh_controls,
    new_model,
    read_snapshot,
    train_arm,
    write_snapshot,
)
from models.model import ModelConfig
from models.simple_ocr import ARMS, OCRAttention, SimpleOCRModel
from scripts.run_b1_session import (
    materialize_cache,
    package,
    plan_commands,
    training_barrier,
)

torch.set_num_threads(1)


def distributed_worker(rank, init):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=init, rank=rank, world_size=2)
    try:
        torch.manual_seed(123)
        model = OCRAttention(64, 32)
        reference = copy.deepcopy(model)
        wrapped = torch.nn.parallel.DistributedDataParallel(model)
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        ropt = torch.optim.SGD(reference.parameters(), lr=0.1)
        inputs = []
        for r in (0, 1):
            gen = torch.Generator().manual_seed(100 + r)
            inputs.append(
                (
                    torch.randn(1, 64, 4, 4, generator=gen),
                    torch.randn(1, 3, 32, generator=gen),
                    torch.randn(1, 3, 5, generator=gen),
                    torch.full((1, 3), bool(r)),
                )
            )
        for _ in range(3):
            opt.zero_grad(set_to_none=True)
            ropt.zero_grad(set_to_none=True)
            wrapped(*inputs[rank]).square().mean().backward()
            reference(
                *(torch.cat([x[i] for x in inputs]) for i in range(4))
            ).square().mean().backward()
            for a, b in zip(model.parameters(), reference.parameters(), strict=True):
                if a.grad is None:
                    raise AssertionError(
                        "Empty rank disconnected a trainable parameter"
                    )
                torch.testing.assert_close(a.grad, b.grad, atol=2e-6, rtol=2e-4)
            opt.step()
            ropt.step()
    finally:
        dist.destroy_process_group()


class BranchTests(unittest.TestCase):
    def test_identity_empty_gradients_and_learning(self):
        torch.manual_seed(7)
        branch = OCRAttention(64, 32)
        x, e, g = torch.randn(2, 64, 4, 4), torch.randn(2, 3, 32), torch.randn(2, 3, 5)
        valid = torch.tensor([[True, True, False], [False, False, False]])
        torch.testing.assert_close(branch(x, e, g, valid), x, rtol=0, atol=0)
        loss = branch(x, e, g, valid).square().mean()
        loss.backward()
        self.assertGreater(abs(branch.beta.grad.item()), 0)
        for name, p in branch.named_parameters():
            self.assertIsNotNone(p.grad, name)
            if name != "beta":
                self.assertEqual(p.grad.abs().sum().item(), 0, name)
        branch.zero_grad(set_to_none=True)
        branch.beta.data.fill_(0.2)
        output = branch(x, e, g, valid)
        torch.testing.assert_close(output[1], x[1], rtol=0, atol=0)
        output.square().mean().backward()
        self.assertGreater(branch.attention.in_proj_weight.grad.abs().sum().item(), 0)
        # Padding values cannot change predictions.
        e2 = e.clone()
        e2[0, 2] = 100
        torch.testing.assert_close(branch(x, e2, g, valid), output)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            self.assertTrue(
                torch.isfinite(branch(x, e, g, torch.zeros_like(valid))).all()
            )
        # A synthetic branch can learn a correction; this is not a research score.
        branch = OCRAttention(64, 32)
        opt = torch.optim.AdamW(branch.parameters(), lr=0.03, weight_decay=0)
        losses = []
        for _ in range(40):
            opt.zero_grad(set_to_none=True)
            # A constant residual is representable; arbitrary pixel noise is
            # not a suitable overfit target for a two-token attention branch.
            loss = (
                (branch(x[:1], e[:1], g[:1], valid[:1]) - (x[:1] + 0.5)).square().mean()
            )
            loss.backward()
            opt.step()
            losses.append(loss.item())
        self.assertLess(losses[-1], losses[0] * 0.05)

    def test_two_process_empty_rank_gradient_equivalence(self):
        with tempfile.TemporaryDirectory() as tmp:
            mp.spawn(
                distributed_worker,
                args=("file://" + str(Path(tmp) / "init"),),
                nprocs=2,
                join=True,
            )

    def test_warm_start_shapes_and_parameter_counts(self):
        torch.manual_seed(5)
        parent_model = SimpleOCRModel("b0_continued", ModelConfig(tiny=True))
        parent = {
            "model_config": parent_model.base.config.to_dict(),
            "model_state_dict": parent_model.base.state_dict(),
        }
        image = torch.randn(1, 3, 336, 336)
        inputs = {
            "questions": ["What is this?"],
            "embedding": torch.randn(1, 2, 32),
            "geometry": torch.rand(1, 2, 5),
            "valid": torch.ones(1, 2, dtype=torch.bool),
        }
        parent_model.eval()
        with torch.no_grad():
            expected = parent_model(image, inputs)
        for arm in ARMS[1:]:
            model = new_model(arm, parent).eval()
            with torch.no_grad():
                actual = model(image, inputs)
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            controls = fresh_controls(model, 10, 1)
            self.assertFalse(model.base.text_encoder.model.training)
            groups = controls.optimizer.param_groups
            self.assertFalse(controls.optimizer.state)
            self.assertTrue(
                all(
                    abs(
                        base
                        - (
                            1e-4
                            if group["group_name"].startswith("ocr_")
                            else 1e-6
                            if "backbone" in group["group_name"]
                            else 1e-5
                        )
                    )
                    < 1e-12
                    for group, base in zip(groups, controls.base_lrs, strict=True)
                )
            )
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "checkpoint_epoch5.pt"
                write_snapshot(model, path, 5, {"synthetic": True})
                rebuilt, metadata = read_snapshot(path)
                self.assertEqual(metadata["epoch"], 5)
                self.assertNotIn("optimizer_state_dict", metadata)
                with torch.no_grad():
                    torch.testing.assert_close(rebuilt.eval()(image, inputs), expected)
                path.write_bytes(b"corrupt")
                with self.assertRaisesRegex(ValueError, "integrity"):
                    read_snapshot(path)
        self.assertEqual(
            sum(p.numel() for p in OCRAttention(64, 768).parameters()), 66369
        )
        self.assertEqual(
            sum(p.numel() for p in OCRAttention(128, 768).parameters()), 165505
        )


class DataAndWorkflowTests(unittest.TestCase):
    def test_frozen_features_and_interventions(self):
        from dataset import load_image

        class FakeEncoder(torch.nn.Module):
            output_dim = 4
            model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=77))

            def tokenizer(self, texts, truncation=False):
                return {"input_ids": [[1, 2, 3] for _ in texts]}

            def forward(self, texts):
                return torch.arange(len(texts) * 3 * 4, dtype=torch.float32).reshape(
                    len(texts), 3, 4
                ), torch.zeros(len(texts), 3, dtype=torch.bool)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / "cache.zip"
            with ZipFile(archive, "w") as z:
                z.writestr("ocr_cache/manifest.json", "{}")
                for split in ("train", "val"):
                    (root / split).mkdir()
                    (root / "binary_masks_png" / split).mkdir(parents=True)
                    records = {}
                    for i in range(2):
                        name = f"{split}{i}.jpg"
                        records[name] = {
                            "question": "Q?",
                            "most_common_answer": "SECRET",
                        }
                        exif = Image.Exif()
                        exif[274] = 6
                        Image.new("RGB", (9, 5), "red").save(
                            root / split / name, exif=exif
                        )
                        image = load_image(root / split / name)
                        Image.new("L", image.size, 255).save(
                            root / "binary_masks_png" / split / f"{split}{i}.png"
                        )
                        r = {
                            "version": "upright-ocr-v1",
                            "filename": name,
                            "size": list(image.size),
                            "pixel_sha256": hashlib.sha256(
                                np.asarray(image).tobytes()
                            ).hexdigest(),
                            "regions": [
                                {
                                    "text": " café ",
                                    "confidence": 0.8,
                                    "polygon": [[0, 0], [2, 0], [2, 2], [0, 2]],
                                }
                            ]
                            if i == 0
                            else [],
                        }
                        z.writestr(f"ocr_cache/{split}/{split}{i}.json", json.dumps(r))
                    (root / f"{split}_grounding.json").write_text(json.dumps(records))
            bank = build_features(
                root,
                archive,
                FakeEncoder(),
                torch.device("cpu"),
                {"train": 2, "val": 2},
            )
            unpacked = root / "unpacked"
            with ZipFile(archive) as z:
                z.extractall(unpacked)
            repacked = materialize_cache(unpacked / "ocr_cache", root / "repacked.zip")
            with ZipFile(archive) as a, ZipFile(repacked) as b:
                self.assertEqual(set(a.namelist()), set(b.namelist()))
                self.assertTrue(
                    all(a.read(name) == b.read(name) for name in a.namelist())
                )
            self.assertEqual(bank["val"]["val0.jpg"]["texts"], ["café"])
            self.assertEqual(bank["val"]["val0.jpg"]["native_wh"], [5, 9])
            self.assertEqual(bank["val"]["val1.jpg"]["embedding"].shape, (0, 4))
            cross = OCRDataset(root, "val", bank["val"], view="cross_image")
            self.assertTrue(all(k != v for k, v in cross.donors.items()))
            self.assertEqual(len(cross[0]["ocr"][0]), 0)
            empty = OCRDataset(root, "val", bank["val"], view="empty")
            self.assertEqual(len(empty[0]["ocr"][0]), 0)
            geom = OCRDataset(root, "val", bank["val"], view="geometry")[0]
            self.assertEqual(geom["ocr"][0].abs().sum().item(), 0.0)
            self.assertEqual(geom["ocr"][1][0, 4].item(), 1.0)
            record = bank["val"]["val0.jpg"]
            record["embedding"] = torch.arange(12, dtype=torch.float32).reshape(3, 4)
            record["geometry"] = torch.tensor(
                [
                    [0.0, 0.0, 0.2, 0.3, 0.0],
                    [0.2, 0.2, 0.4, 0.4, 1.0],
                    [0.5, 0.5, 0.8, 0.8, 0.0],
                ]
            )
            from b1_data import text_permutation

            order = text_permutation("val0.jpg", 3)
            sample = OCRDataset(root, "val", bank["val"], view="text_shuffle")[0]
            torch.testing.assert_close(sample["ocr"][0], record["embedding"][order])
            torch.testing.assert_close(
                sample["ocr"][1][:, :4], record["geometry"][:, :4]
            )
            torch.testing.assert_close(
                sample["ocr"][1][:, 4], record["geometry"][order, 4]
            )
            self.assertNotIn("SECRET", sample["text"])

    def test_complete_analysis_on_synthetic_native_masks(self):
        from b1_analysis import analyze_all
        from dataset import load_image

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "session"
            data = Path(tmp) / "data"
            cache = Path(tmp) / "cache.zip"
            records = {f"im{i}.jpg": {"question": "What?"} for i in range(2)}
            (data / "val").mkdir(parents=True)
            (data / "binary_masks_png/val").mkdir(parents=True)
            (data / "val_grounding.json").write_text(json.dumps(records))
            with ZipFile(cache, "w") as z:
                for name in records:
                    Image.new("RGB", (7, 3), "white").save(data / "val" / name)
                    Image.new("L", (7, 3), 255).save(
                        data / "binary_masks_png/val" / (Path(name).stem + ".png")
                    )
                    image = load_image(data / "val" / name)
                    z.writestr(
                        f"ocr_cache/val/{Path(name).stem}.json",
                        json.dumps(
                            {
                                "filename": name,
                                "size": [7, 3],
                                "pixel_sha256": hashlib.sha256(
                                    np.asarray(image).tobytes()
                                ).hexdigest(),
                                "regions": [],
                            }
                        ),
                    )
            refs = {"summary": {}, "per_sample": {name: 1.0 for name in records}}
            for arm in ARMS:
                folder = root / "training" / arm
                folder.mkdir(parents=True)
                history = [
                    {"epoch": e, "val_miou": 1.0, "train_loss": 1 / e, "ocr_beta": 0.1}
                    for e in range(1, 11)
                ]
                for filename in ("history.jsonl", "val_curve.jsonl"):
                    (folder / filename).write_text("\n".join(map(json.dumps, history)))
            _, commands = plan_commands(root, data, Path("parent"), Path("features"))
            for command in commands:
                out = Path(command[command.index("--output") + 1])
                (out / "masks").mkdir(parents=True)
                epoch = (
                    20
                    if "--parent-eval" in command
                    else int(
                        Path(
                            command[command.index("--checkpoint") + 1]
                        ).stem.removeprefix("checkpoint_epoch")
                    )
                )
                arm = (
                    "original_b0"
                    if epoch == 20
                    else Path(command[command.index("--checkpoint") + 1]).parent.name
                )
                summary = {
                    "epoch": epoch,
                    "arm": arm,
                    "split": "val",
                    "num_samples": 2,
                    "mean_iou": 1.0,
                    "text_mode": "question",
                    "answer_in_input": False,
                    "recipe": {"seed": 42},
                    "model_config": {},
                }
                (out / "metrics.json").write_text(
                    json.dumps({"summary": summary, "per_sample": refs["per_sample"]})
                )
                for name in records:
                    Image.new("L", (7, 3), 255).save(
                        out / "masks" / (Path(name).stem + ".png")
                    )
            result = analyze_all(root, data, cache, refs, repeats=10, expected_count=2)
            self.assertEqual(result["evaluations"], 23)
            self.assertEqual(set(result["selected_epochs"].values()), {5})
            self.assertTrue((root / "analysis/ocr_validation.json").exists())
            self.assertTrue(
                (
                    root
                    / "analysis/comparisons/b1_ocr_48_ep5_clean_minus_b0_same_epoch.json"
                ).exists()
            )

    def test_geometry_validation_and_collation(self):
        record = {
            "version": "upright-ocr-v1",
            "size": [7, 3],
            "regions": [
                {
                    "polygon": [[0, 0], [2, 0], [2, 1], [0, 1]],
                    "confidence": 0.5,
                    "text": "café",
                }
            ],
        }
        self.assertEqual(candidates(record)[0][0], 0)
        record["regions"][0]["polygon"] = [[1, 1], [1, 1], [1, 1]]
        with self.assertRaises(ValueError):
            candidates(record)
        samples = [
            {
                "image": torch.zeros(3, 336, 336),
                "mask": torch.zeros(1, 336, 336),
                "text": "Q?",
                "filename": "x.jpg",
                "ocr": (torch.empty(0, 32), torch.empty(0, 5)),
            }
        ]
        batch = ocr_collate(samples)
        self.assertEqual(tuple(batch["text"]["embedding"].shape), (1, 1, 32))
        self.assertFalse(batch["text"]["valid"].any())
        self.assertNotIn("mask", batch["text"])

    def test_order_barrier_selection_and_zip(self):
        from b1_analysis import selected_epochs

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "session"
            root.mkdir()
            train, evaluation = plan_commands(
                root, Path("data"), Path("parent"), Path("features")
            )
            self.assertEqual(len(train), 3)
            self.assertEqual(len(evaluation), 23)
            self.assertTrue(all("train" in cmd for cmd in train))
            self.assertTrue(all("evaluate" in cmd for cmd in evaluation))
            self.assertFalse(any("test" in cmd for cmd in train + evaluation))
            with self.assertRaises(FileNotFoundError):
                training_barrier(root)
            from scripts.export_b0_validation_masks import digest

            for arm in ARMS:
                folder = root / "training" / arm
                folder.mkdir(parents=True)
                (folder / "TRAINING_COMPLETE.json").write_text(
                    json.dumps({"epochs": 10, "snapshots": [5, 10]})
                )
                (folder / "history.jsonl").write_text(
                    "\n".join(json.dumps({"epoch": e}) for e in range(1, 11))
                )
                (folder / "val_curve.jsonl").write_text(
                    "\n".join(
                        json.dumps(
                            {
                                "epoch": e,
                                "val_miou": 0.9 if e == 7 else 0.5 if e == 5 else 0.4,
                            }
                        )
                        for e in range(1, 11)
                    )
                )
                for e in (5, 10):
                    path = folder / f"checkpoint_epoch{e}.pt"
                    path.write_bytes(b"snapshot")
                    path.with_suffix(".sha256").write_text(digest(path))
            training_barrier(root)
            self.assertTrue(all(e == 5 for e in selected_epochs(root).values()))
            (root / "analysis").mkdir()
            (root / "logs").mkdir()
            for f in ("provenance.json", "original_val_metrics.json", "COMPLETE.json"):
                (root / f).write_text("{}")
            (root / "analysis/per_image.json").write_text("{}")
            archive = package(root)
            with ZipFile(archive) as z:
                self.assertFalse(any(n.endswith(".pt") for n in z.namelist()))
                self.assertIn("archive_manifest.json", z.namelist())
            with self.assertRaises(FileExistsError):
                package(root)

    def test_ten_epoch_cpu_continuation_contract(self):
        # Complete one tiny synthetic arm; verifies actual training/validation,
        # fresh optimizer, weight-only 5/10 persistence and strict reconstruction.
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root = Path(tmp)
            bank = {}
            for split in ("train", "val"):
                (root / split).mkdir()
                (root / "binary_masks_png" / split).mkdir(parents=True)
                records = {}
                bank[split] = {}
                for i in range(2):
                    name = f"{split}_{i}.jpg"
                    records[name] = {"question": "What is this?"}
                    Image.new("RGB", (16, 12), (20, 20, 20)).save(root / split / name)
                    im = Image.new("L", (16, 12), 0)
                    im.paste(255, (0, 0, 8, 12))
                    im.save(root / "binary_masks_png" / split / f"{split}_{i}.png")
                    bank[split][name] = {
                        "embedding": torch.randn(1, 32),
                        "geometry": torch.tensor([[0.0, 0.0, 0.5, 1.0, 0.0]]),
                    }
                (root / f"{split}_grounding.json").write_text(json.dumps(records))
            features = root / "features.pt"
            torch.save(bank, features)
            base = SimpleOCRModel("b0_continued", ModelConfig(tiny=True))
            parent = {
                "model_config": base.base.config.to_dict(),
                "model_state_dict": base.base.state_dict(),
            }
            args = SimpleNamespace(
                output=root / "train_output",
                parent=root / "unused",
                features=features,
                data=root,
                arm="b1_ocr_96",
                batch_size=2,
                eval_batch_size=2,
                workers=0,
            )
            with patch("b1_run.load_parent", return_value=parent):
                train_arm(args)
            self.assertEqual(
                [p.name for p in sorted(args.output.glob("*.pt"))],
                ["checkpoint_epoch10.pt", "checkpoint_epoch5.pt"],
            )
            h = [
                json.loads(l)
                for l in (args.output / "history.jsonl").read_text().splitlines()
            ]
            self.assertEqual(len(h), 10)
            self.assertTrue(all(r["attempted_updates"] == 1 for r in h))
            self.assertEqual(sum(r["successful_updates"] for r in h), 10)
            rebuilt, _ = read_snapshot(args.output / "checkpoint_epoch10.pt")
            self.assertEqual(rebuilt.arm, "b1_ocr_96")
            evaluation = SimpleNamespace(
                parent_eval=False,
                checkpoint=args.output / "checkpoint_epoch10.pt",
                parent=args.parent,
                output=root / "evaluation",
                features=features,
                data=root,
                view="clean",
                eval_batch_size=2,
                workers=0,
            )
            evaluate_snapshot(evaluation)
            metrics = json.loads((evaluation.output / "metrics.json").read_text())
            self.assertEqual(len(metrics["per_sample"]), 2)
            self.assertAlmostEqual(
                metrics["summary"]["mean_iou"], h[-1]["val_miou"], places=7
            )
            with Image.open(evaluation.output / "masks/val_0.png") as im:
                self.assertEqual(im.size, (16, 12))


if __name__ == "__main__":
    unittest.main()
