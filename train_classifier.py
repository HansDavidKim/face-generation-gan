"""Classifier training using Hugging Face Trainer."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import inspect

from transformers import (
    AutoModelForImageClassification,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)

from tqdm.auto import tqdm

from config import ClassifierConfig, get_classifier_config
from utils.data_loader import build_hf_datasets
from utils.helper import ensure_wandb_login, fix_randomness, get_hf_credentials


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
    dataset_name: str | None = None,
) -> TrainingArguments:
    run_dir = output_dir / _sanitize_model_name(model_name)
    run_dir.mkdir(parents=True, exist_ok=True)

    eval_strategy = "epoch" if evaluation else "no"
    save_strategy = "epoch"

    report_to: list[str] = []
    run_name: Optional[str] = None
    if cfg.use_wandb:
        if ensure_wandb_login():
            report_to = ["wandb"]
            sanitized_model = _sanitize_model_name(model_name)
            if dataset_name:
                sanitized_dataset = _sanitize_model_name(dataset_name)
                run_name = f"{sanitized_dataset}__{sanitized_model}"
            else:
                run_name = sanitized_model
        else:
            print("wandb setup failed; disabling wandb logging.")

    kwargs = {
        "output_dir": str(run_dir),
        "per_device_train_batch_size": cfg.batch_size,
        "per_device_eval_batch_size": cfg.batch_size,
        "learning_rate": cfg.learning_rate,
        "num_train_epochs": cfg.epochs,
        "seed": cfg.seed,
    }

    signature = inspect.signature(TrainingArguments.__init__)

    if "evaluation_strategy" in signature.parameters:
        kwargs["evaluation_strategy"] = eval_strategy
    if "eval_strategy" in signature.parameters:
        kwargs["eval_strategy"] = eval_strategy
    if "do_eval" in signature.parameters:
        kwargs["do_eval"] = evaluation

    if "save_strategy" in signature.parameters:
        kwargs["save_strategy"] = save_strategy
    if "load_best_model_at_end" in signature.parameters:
        kwargs["load_best_model_at_end"] = evaluation
    if "metric_for_best_model" in signature.parameters:
        kwargs["metric_for_best_model"] = "loss"
    if "greater_is_better" in signature.parameters:
        kwargs["greater_is_better"] = False

    # === pretrain 여부에 따라 로깅 전략 분기 ===
    is_pretrain = "pretrain" in str(output_dir).lower()
    if "logging_strategy" in signature.parameters:
        kwargs["logging_strategy"] = "steps" if is_pretrain else "epoch"
    if "logging_steps" in signature.parameters:
        kwargs["logging_steps"] = 10 if is_pretrain else 500
    if "logging_first_step" in signature.parameters:
        kwargs["logging_first_step"] = True if is_pretrain else False

    if "report_to" in signature.parameters:
        kwargs["report_to"] = report_to
    if run_name and "run_name" in signature.parameters:
        kwargs["run_name"] = run_name
    if "dataloader_pin_memory" in signature.parameters:
        kwargs["dataloader_pin_memory"] = True
    if "remove_unused_columns" in signature.parameters:
        kwargs["remove_unused_columns"] = False
    if "save_steps" in signature.parameters and "save_strategy" not in kwargs:
        kwargs["save_steps"] = 1_000_000  # effectively disable frequent saves

    if "disable_tqdm" in signature.parameters:
        kwargs["disable_tqdm"] = False

    return TrainingArguments(**kwargs)


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


def train_single_classifier(
    data_root: str,
    model_name: str,
    output_dir: str | Path = "checkpoints/classifiers",
    device: Optional[str] = None,
    cfg: Optional[ClassifierConfig] = None,
    sample_limit: int | None = None,
    hf_dataset: Optional[str] = None,
    hf_splits: Optional[dict[str, str | None]] = None,
    hf_label_column: Optional[str] = None,
    hf_image_column: Optional[str] = None,
    model_init_path: Optional[str] = None,
    progress_description: Optional[str] = None,
) -> Dict[str, float]:
    cfg = cfg or get_classifier_config()
    fix_randomness(cfg.seed, cfg.deterministic)

    train_dataset, valid_dataset, test_dataset, num_classes = build_hf_datasets(
        data_root,
        model_name,
        cfg,
        sample_limit=sample_limit,
        hf_dataset=hf_dataset,
        hf_splits=hf_splits,
        hf_image_column=hf_image_column,
        hf_label_column=hf_label_column,
    )

    model_source = model_init_path or model_name
    model = _create_model(model_source, num_classes)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device)

    has_validation = valid_dataset is not None and len(valid_dataset) > 0
    dataset_name = _infer_dataset_name(data_root)
    training_args = _create_training_arguments(
        model_name,
        Path(output_dir),
        cfg,
        evaluation=has_validation,
        dataset_name=dataset_name,
    )

    callbacks: list = []
    if has_validation and cfg.early_stopping:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=cfg.patience))

    progress_label = progress_description or f"Training {model_name}"
    callbacks.append(_ProgressBarCallback(progress_label))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset if has_validation else None,
        compute_metrics=_compute_metrics if has_validation else None,
        callbacks=callbacks,
    )

    start_time = time.time()
    trainer.train()
    duration = time.time() - start_time

    log_history = trainer.state.log_history
    test_metrics: Dict[str, float] = {}
    if test_dataset is not None and len(test_dataset) > 0:
        test_metrics = trainer.evaluate(test_dataset, metric_key_prefix="test")

    best_checkpoint = getattr(trainer.state, "best_model_checkpoint", None)
    if not best_checkpoint:
        best_checkpoint = getattr(trainer.state, "last_model_checkpoint", None)
    if not best_checkpoint:
        best_checkpoint = trainer.args.output_dir

    summary = {
        "model": model_name,
        "train_samples": len(train_dataset),
        "epochs_ran": getattr(trainer.state, "epoch", cfg.epochs),
        "test_loss": test_metrics.get("test_loss", math.nan),
        "test_accuracy": test_metrics.get("test_accuracy", math.nan),
        "training_time_sec": duration,
        "best_model_checkpoint": best_checkpoint,
    }

    metrics_dir = Path(output_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / f"{_sanitize_model_name(model_name)}_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as fh:
        json.dump({"history": log_history, "summary": summary}, fh, indent=2)

    _maybe_push_to_hf(trainer, model_name, data_root, cfg)

    if cfg.use_wandb:
        try:
            import wandb  # type: ignore
        except ModuleNotFoundError:
            pass
        else:
            if wandb.run is not None:
                wandb.finish()

    return summary


def train_all_classifiers(
    data_root: str,
    output_dir: str | Path = "checkpoints/classifiers",
    device: Optional[str] = None,
    cfg: Optional[ClassifierConfig] = None,
    skip_models: Optional[list[str]] = None,
    pretrained_checkpoints: Optional[dict[str, str]] = None,
) -> Dict[str, Dict[str, float]]:
    cfg = cfg or get_classifier_config()
    results: Dict[str, Dict[str, float]] = {}
    skip_set = set(skip_models or [])
    skip_set.add("google/mobilenet_v2_1.0_224")

    models_to_run = [name for name in cfg.model_names if name not in skip_set]

    if not models_to_run:
        print("No classifiers scheduled for fine-tuning.")
        return results

    total_models = len(models_to_run)

    for idx, model_name in enumerate(models_to_run, start=1):
        print("=" * 80)
        print(f"[Fine-tune {idx}/{total_models}] {model_name}")
        print("=" * 80)
        init_path = pretrained_checkpoints.get(model_name) if pretrained_checkpoints else None
        results[model_name] = train_single_classifier(
            data_root=data_root,
            model_name=model_name,
            output_dir=output_dir,
            device=device,
            cfg=cfg,
            model_init_path=init_path,
            progress_description=f"Fine-tuning {model_name} ({idx}/{total_models})",
        )
    return results


def pretrain_classifiers(
    dataset_name: str,
    output_dir: str | Path = "checkpoints/pretrain",
    device: Optional[str] = None,
    cfg: Optional[ClassifierConfig] = None,
    skip_models: Optional[list[str]] = None,
) -> Dict[str, Dict[str, float]]:
    cfg = cfg or get_classifier_config()

    if not dataset_name:
        raise ValueError("Pretraining dataset name must be provided")

    results: Dict[str, Dict[str, float]] = {}
    skip_set = set(skip_models or [])

    models_to_run = [name for name in cfg.model_names if name not in skip_set]

    if not models_to_run:
        print("No classifiers scheduled for pretraining.")
        return results

    total_models = len(models_to_run)

    for idx, model_name in enumerate(models_to_run, start=1):
        print("=" * 80)
        print(f"[Pretrain {idx}/{total_models}] {model_name} on {dataset_name}")
        print("=" * 80)
        results[model_name] = train_single_classifier(
            data_root=dataset_name,
            model_name=model_name,
            output_dir=output_dir,
            device=device,
            cfg=cfg,
            sample_limit=cfg.pretrain_sample_limit,
            hf_dataset=dataset_name,
            hf_splits={
                "train": cfg.pretrain_train_split,
                "validation": cfg.pretrain_valid_split,
                "test": cfg.pretrain_test_split,
            },
            hf_label_column=cfg.pretrain_label_column,
            hf_image_column=cfg.pretrain_image_column,
            progress_description=f"Pretraining {model_name} ({idx}/{total_models})",
        )

    return results


class _ProgressBarCallback(TrainerCallback):
    def __init__(self, description: str):
        self.description = description
        self._pbar: Optional[tqdm] = None
        self._last_step: int = 0

    def on_train_begin(self, args, state, control, **kwargs):
        total = state.max_steps if state.max_steps and state.max_steps > 0 else None
        self._pbar = tqdm(total=total, desc=self.description, leave=True, dynamic_ncols=True)
        self._last_step = 0

    def on_step_end(self, args, state, control, **kwargs):
        if not self._pbar:
            return
        step = state.global_step or 0
        delta = step - self._last_step
        if delta > 0:
            self._pbar.update(delta)
            self._last_step = step

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not self._pbar or not logs:
            return
        if "loss" in logs:
            self._pbar.set_postfix(loss=f"{logs['loss']:.4f}")
        elif "eval_loss" in logs:
            self._pbar.set_postfix(eval_loss=f"{logs['eval_loss']:.4f}")

    def on_train_end(self, args, state, control, **kwargs):
        if self._pbar:
            self._pbar.close()
            self._pbar = None


if __name__ == "__main__":
    cfg = get_classifier_config()

    from utils.helper import get_dataset_list
    dataset_list = get_dataset_list()
    dataset_list = list(map(lambda x: x.split('/')[1], dataset_list))

    pretrained_map: Dict[str, str] = {}

    if cfg.pretrain_enabled and cfg.pretrain_dataset:
        pretrain_dir = Path(cfg.pretrain_output_dir or "checkpoints/pretrain")
        pretrain_dir.mkdir(parents=True, exist_ok=True)
        pretrain_results = pretrain_classifiers(
            dataset_name=cfg.pretrain_dataset,
            output_dir=pretrain_dir,
            device=None,
            cfg=cfg,
            skip_models=None,
        )
        for name, summary in pretrain_results.items():
            checkpoint = summary.get("best_model_checkpoint")
            if checkpoint:
                pretrained_map[name] = checkpoint

    for data in dataset_list[:-1]:
        train_all_classifiers(
            data_root=f'private/{data}',
            output_dir=f'checkpoints/{data}',
            device=None,
            cfg=cfg,
            skip_models=None,
            pretrained_checkpoints=pretrained_map,
        )
