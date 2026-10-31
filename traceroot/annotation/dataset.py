"""Dataset loader, validator, and train/val/test splitting utilities."""

import json
from pathlib import Path
import random
from typing import Dict, List, Optional, Tuple, Union
from traceroot.annotation.schema import HumanAnnotation, HumanDataset, MultiAnnotatorRecord


class DatasetManager:
    """Manages storage, validation, and stratified splitting of human annotation datasets."""

    @staticmethod
    def load_dataset(file_path: Union[str, Path]) -> HumanDataset:
        """Load human annotation dataset from a JSON file."""
        p = Path(file_path)
        if not p.exists():
            raise FileNotFoundError(f"Annotation file not found: {file_path}")
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
            return HumanDataset(**data)

    @staticmethod
    def save_dataset(dataset: HumanDataset, file_path: Union[str, Path]) -> None:
        """Save human annotation dataset to a JSON file."""
        p = Path(file_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            f.write(json.dumps(dataset.model_dump(), indent=2, default=str))

    @staticmethod
    def split(
        dataset: HumanDataset,
        val_ratio: float = 0.3,
        test_ratio: float = 0.2,
        seed: int = 42,
    ) -> Tuple[HumanDataset, HumanDataset, HumanDataset]:
        """Split dataset into train, validation, and test subsets."""
        records = list(dataset.records)
        random.seed(seed)
        random.shuffle(records)

        n = len(records)
        n_test = int(n * test_ratio)
        n_val = int(n * val_ratio)
        n_train = n - n_val - n_test

        train_records = records[:n_train]
        val_records = records[n_train : n_train + n_val]
        test_records = records[n_train + n_val :]

        return (
            HumanDataset(name=f"{dataset.name}_train", records=train_records),
            HumanDataset(name=f"{dataset.name}_val", records=val_records),
            HumanDataset(name=f"{dataset.name}_test", records=test_records),
        )
