"""Utility helpers for preparing classifier dataloaders."""

from __future__ import annotations

from typing import Tuple

import numpy as np
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder

from config import ClassifierConfig, get_classifier_config
from utils.helper import fix_randomness
from utils.image_transform import get_transform


def split_indices(
    labels: np.ndarray,
    train_ratio: float,
    valid_ratio: float,
    test_ratio: float,
    seed: int,
    stratify: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not np.isclose(train_ratio + valid_ratio + test_ratio, 1.0):
        raise ValueError("train/valid/test ratios must sum to 1.0")

    label_counts = np.bincount(labels)
    min_count = label_counts[label_counts > 0].min() if label_counts.any() else 0
    stratify_labels = labels if stratify and np.unique(labels).size > 1 and min_count >= 2 else None

    train_idx, temp_idx, _, temp_labels = train_test_split(
        np.arange(labels.shape[0]),
        labels,
        test_size=valid_ratio + test_ratio,
        stratify=stratify_labels,
        random_state=seed,
    )

    if valid_ratio == 0:
        return train_idx, np.array([], dtype=int), temp_idx
    if test_ratio == 0:
        return train_idx, temp_idx, np.array([], dtype=int)

    valid_relative = valid_ratio / (valid_ratio + test_ratio)

    if stratify_labels is not None:
        temp_label_counts = np.bincount(temp_labels)
        min_temp = temp_label_counts[temp_label_counts > 0].min()
        stratify_temp = temp_labels if min_temp >= 2 else None
    else:
        stratify_temp = None

    valid_idx, test_idx = train_test_split(
        temp_idx,
        test_size=1 - valid_relative,
        stratify=stratify_temp,
        random_state=seed,
    )

    return train_idx, valid_idx, test_idx


def prepare_dataloaders(
    data_root: str,
    model_name: str,
    cfg: ClassifierConfig | None = None,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    cfg = cfg or get_classifier_config()

    fix_randomness(cfg.seed, cfg.deterministic)

    base_dataset = ImageFolder(data_root)
    labels = np.array(base_dataset.targets)

    train_idx, valid_idx, test_idx = split_indices(
        labels,
        cfg.train_ratio,
        cfg.valid_ratio,
        cfg.test_ratio,
        cfg.seed,
        cfg.is_stratified,
    )

    train_dataset = ImageFolder(
        data_root,
        transform=get_transform(
            model_name,
            augment=cfg.augment,
            downsample_size=cfg.pretrain_downsample_size,
        ),
    )
    valid_dataset = ImageFolder(
        data_root,
        transform=get_transform(
            model_name,
            augment=False,
            downsample_size=cfg.pretrain_downsample_size,
        ),
    )
    test_dataset = ImageFolder(
        data_root,
        transform=get_transform(
            model_name,
            augment=False,
            downsample_size=cfg.pretrain_downsample_size,
        ),
    )

    train_loader = DataLoader(
        Subset(train_dataset, train_idx.tolist()),
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    valid_loader = DataLoader(
        Subset(valid_dataset, valid_idx.tolist()),
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        Subset(test_dataset, test_idx.tolist()),
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, valid_loader, test_loader


def build_hf_datasets(
    data_root: str,
    model_name: str,
    cfg: ClassifierConfig,
    sample_limit: int | None = None,
    hf_dataset: str | None = None,
    hf_splits: dict[str, str | None] | None = None,
    hf_image_column: str | None = None,
    hf_label_column: str | None = None,
):
    label_column = hf_label_column or "label"

    if hf_dataset:
        dataset_dict = load_dataset(hf_dataset)
        hf_splits = hf_splits or {}

        def _resolve_split(key: str, default: str | None = None):
            name = hf_splits.get(key, default) if hf_splits else default
            if not name:
                return None
            if name not in dataset_dict:
                print(f"Warning: split '{name}' not found in dataset '{hf_dataset}'; skipping")
                return None
            return dataset_dict[name]

        train_dataset = _resolve_split("train", "train")
        if train_dataset is None:
            raise ValueError(f"Dataset '{hf_dataset}' must provide a train split")

        if sample_limit is not None and len(train_dataset) > sample_limit:
            train_dataset = train_dataset.select(range(sample_limit))

        valid_dataset = _resolve_split("validation", None)
        test_dataset = _resolve_split("test", "test")
        if valid_dataset is None and test_dataset is not None:
            valid_dataset = test_dataset

        # === 안전한 라벨 탐색 ===
        reference_split = train_dataset or valid_dataset or test_dataset
        candidate_labels = [
            label_column,
            "label",
            "identity",
            "id",
            "class",
            "target",
        ]
        label_found = None
        for c in candidate_labels:
            if c in reference_split.column_names:
                label_found = c
                break

        if label_found is None:
            raise ValueError(
                f"Dataset '{hf_dataset}' must contain a label column; "
                f"none of {candidate_labels} found. "
                f"Available: {reference_split.column_names}"
            )

        if label_found != "label":
            print(f"Info: using '{label_found}' as label column for dataset '{hf_dataset}'")
            for ds_name in ("train_dataset", "valid_dataset", "test_dataset"):
                ds = locals().get(ds_name)
                if ds is not None and label_found in ds.column_names:
                    locals()[ds_name] = ds.rename_column(label_found, "label")

        # === 이미지 컬럼 식별 ===
        if hf_image_column:
            image_column = hf_image_column
        else:
            image_column = None
            for name, feature in reference_split.features.items():
                if feature.__class__.__name__ == "Image":
                    image_column = name
                    break
            if image_column is None:
                if "image" in reference_split.column_names:
                    image_column = "image"
                else:
                    raise ValueError("Unable to identify image column in dataset")

        # === 클래스 개수 추론 ===
        features = train_dataset.features
        if "label" in features and hasattr(features["label"], "names"):
            num_classes = len(features["label"].names)
        else:
            num_classes = len(set(train_dataset["label"]))

        # === 변환 적용 ===
        train_dataset = train_dataset.with_transform(
            _build_transform(
                model_name,
                image_column,
                augment=cfg.augment,
                downsample_size=cfg.pretrain_downsample_size,
            )
        )
        eval_transform = _build_transform(
            model_name,
            image_column,
            augment=False,
            downsample_size=cfg.pretrain_downsample_size,
        )

        valid_dataset = (
            valid_dataset.with_transform(eval_transform) if valid_dataset is not None else None
        )
        test_dataset = (
            test_dataset.with_transform(eval_transform) if test_dataset is not None else None
        )

        return train_dataset, valid_dataset, test_dataset, num_classes

    # === 로컬 이미지 폴더 ===
    split_str = f"train[:{sample_limit}]" if sample_limit is not None else "train"
    dataset = load_dataset("imagefolder", data_dir=data_root, split=split_str)

    image_column = None
    for name, feature in dataset.features.items():
        if feature.__class__.__name__ == "Image":
            image_column = name
            break
    if image_column is None:
        image_column = "image"

    labels = np.array(dataset["label"])
    train_idx, valid_idx, test_idx = split_indices(
        labels,
        cfg.train_ratio,
        cfg.valid_ratio,
        cfg.test_ratio,
        cfg.seed,
        cfg.is_stratified,
    )

    train_dataset = dataset.select(train_idx.tolist()).with_transform(
        _build_transform(
            model_name,
            image_column,
            augment=cfg.augment,
            downsample_size=cfg.pretrain_downsample_size,
        )
    )
    eval_transform = _build_transform(
        model_name,
        image_column,
        augment=False,
        downsample_size=cfg.pretrain_downsample_size,
    )

    valid_dataset = (
        dataset.select(valid_idx.tolist()).with_transform(eval_transform)
        if valid_idx.size
        else None
    )
    test_dataset = (
        dataset.select(test_idx.tolist()).with_transform(eval_transform)
        if test_idx.size
        else None
    )

    num_classes = len(dataset.features["label"].names)
    return train_dataset, valid_dataset, test_dataset, num_classes


def _build_transform(
    model_name: str,
    image_column: str,
    augment: bool,
    downsample_size: int | None = None,
) -> callable:
    transform = get_transform(model_name, augment=augment, downsample_size=downsample_size)

    def _apply(batch):
        images_raw = batch.get(image_column)
        if images_raw is None:
            raise KeyError(f"Column '{image_column}' not found in batch: available keys {list(batch.keys())}")

        processed = []
        for image in images_raw:
            if hasattr(image, "convert"):
                processed.append(transform(image.convert("RGB")))
            else:
                from PIL import Image
                processed.append(transform(Image.open(image).convert("RGB")))

        batch["pixel_values"] = processed
        batch["labels"] = batch["label"]
        batch.pop(image_column, None)
        batch.pop("label", None)
        return batch

    return _apply
