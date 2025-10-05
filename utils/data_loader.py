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


# ============================================================
# Split helper
# ============================================================
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


# ============================================================
# Torch Dataloader builder (local ImageFolder)
# ============================================================
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

    downsample_size = None

    train_dataset = ImageFolder(
        data_root,
        transform=get_transform(
            model_name,
            augment=cfg.augment,
            downsample_size=downsample_size,
        ),
    )
    valid_dataset = ImageFolder(
        data_root,
        transform=get_transform(
            model_name,
            augment=False,
            downsample_size=downsample_size,
        ),
    )
    test_dataset = ImageFolder(
        data_root,
        transform=get_transform(
            model_name,
            augment=False,
            downsample_size=downsample_size,
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


# ============================================================
# HuggingFace dataset builder
# ============================================================
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

        def _resolve_split(
            key: str,
            default: str | None = None,
            alternatives: tuple[str, ...] = (),
        ):
            # priority: explicit mapping → default → alternatives
            if hf_splits and key in hf_splits and hf_splits[key]:
                candidates = tuple(filter(None, (hf_splits[key],)))
            else:
                base = (default,) if default else ()
                candidates = base + alternatives

            attempted: list[str] = []
            for name in candidates:
                if not name:
                    continue
                if name in dataset_dict:
                    return dataset_dict[name]
                attempted.append(name)

            if attempted:
                tried = ", ".join(attempted)
                print(f"Warning: none of splits [{tried}] found in dataset '{hf_dataset}'")
            return None

        train_dataset = _resolve_split("train", "train")
        valid_dataset = _resolve_split("validation", "validation", ("valid",))
        test_dataset = _resolve_split("test", "test", ())

        reference_split = train_dataset or valid_dataset or test_dataset
        if reference_split is None:
            raise ValueError(f"Dataset '{hf_dataset}' produced no available split")

        # ====================================================
        # CelebA 전용 처리 (celeb_id → label)
        # ====================================================
        if "celeb_id" in reference_split.column_names:
            print(f"Detected CelebA dataset: using 'celeb_id' as label column")
            def _attach_celeb_id(ds):
                if ds is None:
                    return None
                return ds.add_column("label", ds["celeb_id"])
            train_dataset = _attach_celeb_id(train_dataset)
            valid_dataset = _attach_celeb_id(valid_dataset)
            test_dataset = _attach_celeb_id(test_dataset)
        else:
            # 일반 fallback
            candidate_labels = [label_column, "label", "identity", "id", "class", "target"]
            label_found = None
            for c in candidate_labels:
                if c in (reference_split.column_names if reference_split else []):
                    label_found = c
                    break
            if label_found is None:
                raise ValueError(
                    f"Dataset '{hf_dataset}' must contain a label column; "
                    f"none of {candidate_labels} found. Available columns: {reference_split.column_names}"
                )
            if label_found != "label":
                print(f"Info: using '{label_found}' as label column for dataset '{hf_dataset}'")
                for name in ["train_dataset", "valid_dataset", "test_dataset"]:
                    ds = locals().get(name)
                    if ds is not None and label_found in ds.column_names:
                        locals()[name] = ds.rename_column(label_found, "label")

        # ====================================================
        # Image column 탐색
        # ====================================================
        if hf_image_column:
            image_column = hf_image_column
        else:
            image_column = None
            for name, feature in reference_split.features.items():
                if feature.__class__.__name__ == "Image":
                    image_column = name
                    break
            if image_column is None and "image" in reference_split.column_names:
                image_column = "image"
            elif image_column is None:
                raise ValueError("Unable to identify image column in dataset")

        # ====================================================
        # 클래스 개수 계산
        # ====================================================
        features = train_dataset.features
        if "label" in features and hasattr(features["label"], "names"):
            num_classes = len(features["label"].names)
        else:
            num_classes = len(set(train_dataset["label"]))

        # ====================================================
        # Transform 적용
        # ====================================================
        train_transform = _build_transform_pipeline(
            model_name,
            augment=cfg.augment,
            downsample_size=None,
        )
        eval_transform = _build_transform_pipeline(
            model_name,
            augment=False,
            downsample_size=None,
        )

        train_dataset = _apply_transform_map(
            train_dataset,
            transform=train_transform,
            image_column=image_column,
            label_column="label",
            desc=f"Transforming train split for {model_name}",
        )
        valid_dataset = (
            _apply_transform_map(
                valid_dataset,
                transform=eval_transform,
                image_column=image_column,
                label_column="label",
                desc=f"Transforming validation split for {model_name}",
            )
            if valid_dataset is not None
            else None
        )
        test_dataset = (
            _apply_transform_map(
                test_dataset,
                transform=eval_transform,
                image_column=image_column,
                label_column="label",
                desc=f"Transforming test split for {model_name}",
            )
            if test_dataset is not None
            else None
        )

        return train_dataset, valid_dataset, test_dataset, num_classes

    # ====================================================
    # Local ImageFolder fallback
    # ====================================================
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

    train_transform = _build_transform_pipeline(
        model_name,
        augment=cfg.augment,
        downsample_size=None,
    )
    eval_transform = _build_transform_pipeline(
        model_name,
        augment=False,
        downsample_size=None,
    )

    train_dataset = dataset.select(train_idx.tolist())
    valid_dataset = dataset.select(valid_idx.tolist()) if valid_idx.size else None
    test_dataset = dataset.select(test_idx.tolist()) if test_idx.size else None

    train_dataset = _apply_transform_map(
        train_dataset,
        transform=train_transform,
        image_column=image_column,
        label_column="label",
        desc=f"Transforming train split for {model_name}",
    )
    valid_dataset = (
        _apply_transform_map(
            valid_dataset,
            transform=eval_transform,
            image_column=image_column,
            label_column="label",
            desc=f"Transforming validation split for {model_name}",
        )
        if valid_dataset is not None
        else None
    )
    test_dataset = (
        _apply_transform_map(
            test_dataset,
            transform=eval_transform,
            image_column=image_column,
            label_column="label",
            desc=f"Transforming test split for {model_name}",
        )
        if test_dataset is not None
        else None
    )

    num_classes = len(dataset.features["label"].names)
    return train_dataset, valid_dataset, test_dataset, num_classes


# ============================================================
# Transform builder
# ============================================================
def _build_transform_pipeline(
    model_name: str,
    augment: bool,
    downsample_size: int | None = None,
):
    return get_transform(model_name, augment=augment, downsample_size=downsample_size)


def _apply_transform_map(
    dataset,
    *,
    transform,
    image_column: str,
    label_column: str,
    desc: str,
):
    if dataset is None:
        return None

    from PIL import Image

    def _process(batch):
        images_raw = batch.get(image_column)
        if images_raw is None:
            raise KeyError(
                f"Column '{image_column}' not found in batch: available keys {list(batch.keys())}"
            )

        pixel_values = []
        for image in images_raw:
            if hasattr(image, "convert"):
                tensor = transform(image.convert("RGB"))
            else:
                tensor = transform(Image.open(image).convert("RGB"))
            pixel_values.append(tensor.numpy())

        labels = [int(label) for label in batch[label_column]]
        batch["pixel_values"] = pixel_values
        batch["labels"] = labels
        return batch

    remove_columns = [col for col in dataset.column_names if col == image_column]

    mapped = dataset.map(
        _process,
        batched=True,
        batch_size=64,
        desc=desc,
        remove_columns=remove_columns,
        num_proc=8,
    )
    mapped.set_format(type="torch", columns=["pixel_values", "labels"])
    return mapped
