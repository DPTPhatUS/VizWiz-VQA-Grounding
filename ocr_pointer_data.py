"""Native OCR scope labels and a deterministic, train-only scorer holdout."""

import hashlib
import json

from ocr_data import OCRGroundingDataset
from ocr_selection_data import OCRSelectionDataset, selection_targets

SPLIT_VERSION = "ocr-pointer-image-split-v1"
TARGET_VERSION = "ocr-native-sufficiency-v1"


def pointer_partition(names, seed):
    """Stable image split, independent of labels, cache contents and parent outputs."""
    ordered = sorted(
        names,
        key=lambda n: hashlib.sha256(
            f"{SPLIT_VERSION}:{seed}:{n}".encode()
        ).hexdigest(),
    )
    if len(ordered) < 2:
        raise ValueError("Pointer training needs at least two images for a holdout")
    count = max(1, round(0.15 * len(ordered)))
    return ordered[count:], ordered[:count]


class OCRPointerDataset(OCRSelectionDataset):
    def __init__(self, *args, partition="all", split_seed=42, **kwargs):
        super().__init__(*args, **kwargs)
        self.source_supervision_digest = self.supervision_digest
        self.split_protocol = None
        if partition not in {"all", "fit", "holdout"}:
            raise ValueError("Invalid scorer partition")
        if partition != "all":
            if self.split != "train":
                raise ValueError("Scorer partitions are train-only")
            fit, held = pointer_partition(self.names, split_seed)
            self.names = sorted(fit if partition == "fit" else held)
            self.split_protocol = {
                "version": SPLIT_VERSION,
                "seed": split_seed,
                "fit": sorted(fit),
                "holdout": sorted(held),
                "partition": partition,
            }
            self.supervision_digest = hashlib.sha256(
                json.dumps(
                    [self.supervision_digest, self.split_protocol], sort_keys=True
                ).encode()
            ).hexdigest()

    def _training_targets(self, mask, inputs):
        return selection_targets(mask, inputs, include_word_purity=True)

    def __getitem__(self, index):
        # No answer labels or answer encodings are generated for this experiment.
        return OCRGroundingDataset.__getitem__(self, index)
