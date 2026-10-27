"""Frozen-parent wrapper for native crop and coordinate refinement."""

from models.native_refiner import NativeRefiner
from models.ocr_model import OCRGroundingModel
from native_refinement_ops import global_native

EXPERIMENT = "ocr-native-refinement-v1"


class NativeRefinementModel(OCRGroundingModel):
    def __init__(self, parent, *, architecture="native_crop", **options):
        self.architecture = architecture
        super().__init__(parent, **options)
        self.experiment_config.update(
            experiment=EXPERIMENT,
            architecture=architecture,
            geometry="float64-centres-and-native-logit-interpolation-v1",
            output_protocol="native-logits-v1",
            crop_size=[192, 192],
            context_scale=2.0,
            native_min_context=32,
            native_overlap="mean-local-overwrite-v1",
            adapter_width=32 if self.experiment_config["tiny"] else 64,
            query_chunk=65536,
            support_context=2.0,
            objective="native-refinement-v1",
            point_pixel_samples=256 if self.experiment_config["tiny"] else 8192,
            point_class_samples=64 if self.experiment_config["tiny"] else 1024,
        )

    def _make_adapter(self, text_dim, detail_dim, width):
        return NativeRefiner(
            text_dim,
            detail_dim,
            self.parent.decoder.output_channels,
            32 if self.parent.experiment_config["tiny"] else 64,
            self.architecture,
            self.parent.experiment_config["tiny"],
        )

    def _adapter_features(self, base, batch):
        return {
            "detail": base["detail_features"],
            "semantic": base["semantic_features"],
            "refine_boxes": batch["refine_boxes"],
            "native_rgb": batch["native_rgb"],
            "native_crops": batch.get("native_crops"),
            "truths": batch.get("original_mask") if self.training else None,
            "global_only": not self.training and self.inference_view == "global",
        }

    def forward(self, batch):
        output = super().forward(batch)
        if "global_delta" not in output:
            base = output["logits"]
            output["base_logits"] = base
            output["native_logits"] = [
                global_native(
                    base[i : i + 1], base[i : i + 1] * 0, tuple(rgb.shape[-2:])
                )
                for i, (rgb,) in enumerate(batch["native_rgb"])
            ]
        return output

    @classmethod
    def from_parent(cls, checkpoint, **options):
        model = super().from_parent(checkpoint, **options)
        model.parent_training_provenance = checkpoint.get("run_config", {})
        return model
