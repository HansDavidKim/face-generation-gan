"""Reusable utilities for Hugging Face-based classifier training."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForImageClassification,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from config import ClassifierConfig, get_classifier_config
from utils.helper import ensure_wandb_login, fix_randomness, get_hf_credentials
from utils.image_transform import get_transform

try:  # optional experiment tracker
    import trackio
except ModuleNotFoundError:  # pragma: no cover
    trackio = None


def _sanitize_model_name(model_name: str) -> str:
    return model_name.replace("/", "__").replace(":", "_")


def _infer_dataset_name(data_root: str) -> str:
    stripped = str(data_root).strip()
    if not stripped:
        return "dataset"

    trimmed = stripped.rstrip("/ ")
    path = Path(trimmed)
    name = path.name or path.parent.name
    return name or "dataset"


def _split_indices(
    labels: np.ndarray,
    train_ratio: float,
    valid_ratio: float,
    test_ratio: float,
    seed: int,
    stratify: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not np.isclose(train_ratio + valid_ratio + test_ratio, 1.0):
        raise ValueError("train/valid/test ratios must sum to 1.0")

    stratify_labels = labels if stratify and np.unique(labels).size > 1 else None

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
    stratify_temp = temp_labels if stratify_labels is not None else None

    valid_idx, test_idx = train_test_split(
        temp_idx,
        test_size=1 - valid_relative,
        stratify=stratify_temp,
        random_state=seed,
    )

    return train_idx, valid_idx, test_idx


def _build_transform(model_name: str, augment: bool) -> callable:
    transform = get_transform(model_name, augment=augment)

    def _apply_transforms(batch):
        images = [transform(image.convert("RGB")) for image in batch["image"]]
        batch["pixel_values"] = images
        batch["labels"] = batch["label"]
        return batch

    return _apply_transforms


def _prepare_datasets(
    data_root: str,
    model_name: str,
    cfg: ClassifierConfig,
):
    dataset_dict = load_dataset("imagefolder", data_dir=data_root)
    dataset = dataset_dict["train"]

    labels = np.array(dataset["label"])
    train_idx, valid_idx, test_idx = _split_indices(
        labels,
        cfg.train_ratio,
        cfg.valid_ratio,
        cfg.test_ratio,
        cfg.seed,
        cfg.is_stratified,
    )

    train_dataset = dataset.select(train_idx.tolist())
    valid_dataset = dataset.select(valid_idx.tolist()) if valid_idx.size else None
    test_dataset = dataset.select(test_idx.tolist()) if test_idx.size else None

    train_dataset = train_dataset.with_transform(_build_transform(model_name, augment=cfg.augment))
    eval_transform = _build_transform(model_name, augment=False)

    if valid_dataset is not None:
        valid_dataset = valid_dataset.with_transform(eval_transform)
    if test_dataset is not None:
        test_dataset = test_dataset.with_transform(eval_transform)

    label_names = dataset.features["label"].names
    num_classes = len(label_names)

    return train_dataset, valid_dataset, test_dataset, num_classes


def _create_model(model_name: str, num_classes: int) -> torch.nn.Module:
    model = AutoModelForImageClassification.from_pretrained(
        model_name,
        num_labels=num_classes,
        ignore_mismatched_sizes=True,
    )

    if hasattr(model, "config"):
        model.config.label2id = {str(i): i for i in range(num_classes)}
        model.config.id2label = {i: str(i) for i in range(num_classes)}

    return model


def _create_training_arguments(
    model_name: str,
    output_dir: Path,
    cfg: ClassifierConfig,
    evaluation: bool,
) -> TrainingArguments:
    run_dir = output_dir / _sanitize_model_name(model_name)
    run_dir.mkdir(parents=True, exist_ok=True)

    eval_strategy = "epoch" if evaluation else "no"
    save_strategy = eval_strategy if evaluation else "no"

    report_to: list[str] = []
    run_name: Optional[str] = None
    if cfg.use_wandb:
        if ensure_wandb_login():
            report_to = ["wandb"]
            run_name = _sanitize_model_name(model_name)
        else:
            print("wandb setup failed; disabling wandb logging.")

    args_kwargs = {
        "output_dir": str(run_dir),
        "per_device_train_batch_size": cfg.batch_size,
        "per_device_eval_batch_size": cfg.batch_size,
        "learning_rate": cfg.learning_rate,
        "num_train_epochs": cfg.epochs,
        "evaluation_strategy": eval_strategy,
        "save_strategy": save_strategy,
        "load_best_model_at_end": evaluation,
        "metric_for_best_model": "loss",
        "greater_is_better": False,
        "logging_strategy": "epoch",
        "seed": cfg.seed,
        "report_to": report_to,
        "dataloader_pin_memory": True,
    }
    if run_name is not None:
        args_kwargs["run_name"] = run_name

    return TrainingArguments(**args_kwargs)


def _maybe_push_to_hf(
    trainer: Trainer,
    model_name: str,
    data_root: str,
    cfg: ClassifierConfig,
) -> None:
    if not getattr(cfg, "upload_hf", False):
        return

    username, token = get_hf_credentials()
    if not username or not token:
        print("Hugging Face 자격 증명이 없어 업로드를 건너뜁니다.")
        return

    dataset_slug = _sanitize_model_name(_infer_dataset_name(data_root))
    repo_name = f"{_sanitize_model_name(model_name)}_{dataset_slug}"
    repo_id = f"{username}/{repo_name}"

    try:
        import huggingface_hub  # type: ignore  # noqa: F401
    except ModuleNotFoundError:
        print("huggingface_hub 패키지가 없어 모델 업로드를 건너뜁니다.")
        return

    trainer.args.hub_model_id = repo_id
    trainer.args.hub_token = token
    trainer.args.push_to_hub = True

    try:
        trainer.push_to_hub(
            token=token,
            commit_message=f"Add {model_name} trained on {dataset_slug}",
            model_name=repo_name,
        )
        print(f"모델을 Hugging Face Hub에 업로드했습니다: {repo_id}")
    except Exception as exc:  # pragma: no cover - network/git dependent
        print(f"Hugging Face 업로드 실패: {exc}")


def _compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    accuracy = (preds == labels).mean() if labels.size > 0 else math.nan
    return {"accuracy": float(accuracy)}


def _init_tracker(cfg: ClassifierConfig, model_name: str):  # pragma: no cover
    if not cfg.use_trackio or trackio is None:
        return None
    return trackio.init(space=cfg.space_name, run_name=model_name, config={"model": model_name})


def train_single_classifier(
    data_root: str,
    model_name: str,
    output_dir: str | Path = "checkpoints/classifiers",
    device: Optional[str] = None,
    cfg: Optional[ClassifierConfig] = None,
) -> Dict[str, float]:
    cfg = cfg or get_classifier_config()
    fix_randomness(cfg.seed, cfg.deterministic)

    train_dataset, valid_dataset, test_dataset, num_classes = _prepare_datasets(
        data_root,
        model_name,
        cfg,
    )

    model = _create_model(model_name, num_classes)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device)

    has_validation = valid_dataset is not None and len(valid_dataset) > 0
    training_args = _create_training_arguments(
        model_name,
        Path(output_dir),
        cfg,
        evaluation=has_validation,
    )

    callbacks = []
    if has_validation and cfg.early_stopping:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=cfg.patience))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset if has_validation else None,
        compute_metrics=_compute_metrics if has_validation else None,
        callbacks=callbacks,
    )

    tracker = _init_tracker(cfg, model_name)

    start_time = time.time()
    trainer.train()
    duration = time.time() - start_time

    log_history = trainer.state.log_history
    test_metrics: Dict[str, float] = {}
    if test_dataset is not None and len(test_dataset) > 0:
        test_metrics = trainer.evaluate(test_dataset, metric_key_prefix="test")

    summary = {
        "model": model_name,
        "train_runtime": trainer.state.train_runtime,
        "train_samples": len(train_dataset),
        "epochs_ran": trainer.state.epoch,
        "test_loss": test_metrics.get("test_loss", math.nan),
        "test_accuracy": test_metrics.get("test_accuracy", math.nan),
        "training_time_sec": duration,
    }

    metrics_dir = Path(output_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / f"{_sanitize_model_name(model_name)}_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as fh:
        json.dump({"history": log_history, "summary": summary}, fh, indent=2)

    if tracker is not None:
        tracker.log(**summary)
        if hasattr(tracker, "close"):
            tracker.close()
        elif hasattr(tracker, "finish"):
            tracker.finish()

    _maybe_push_to_hf(trainer, model_name, data_root, cfg)

    return summary


def train_all_classifiers(
    data_root: str,
    output_dir: str | Path = "checkpoints/classifiers",
    device: Optional[str] = None,
    cfg: Optional[ClassifierConfig] = None,
) -> Dict[str, Dict[str, float]]:
    cfg = cfg or get_classifier_config()
    results: Dict[str, Dict[str, float]] = {}

    for model_name in cfg.model_names:
        print("=" * 80)
        print(f"Training classifier: {model_name}")
        print("=" * 80)
        results[model_name] = train_single_classifier(
            data_root=data_root,
            model_name=model_name,
            output_dir=output_dir,
            device=device,
            cfg=cfg,
        )
    return results


__all__ = ["train_single_classifier", "train_all_classifiers"]
