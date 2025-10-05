"""Train a linear classifier on AuraFace embeddings."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import ImageFolder
from tqdm.auto import tqdm

from config import ClassifierConfig, get_classifier_config
from utils.aura_feature import extract_embedding
from utils.data_loader import split_indices
from utils.helper import fix_randomness


def _compute_embeddings(
    dataset: ImageFolder,
    indices: np.ndarray,
    cache_dir: Path,
    split_name: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{split_name}_embeddings.npz"

    if cache_path.exists():
        cached = np.load(cache_path)
        return (
            torch.from_numpy(cached["embeddings"]),
            torch.from_numpy(cached["labels"]),
        )

    embeddings = []
    labels = []

    with tqdm(total=len(indices), desc=f"Extracting {split_name} embeddings") as pbar:
        for idx in indices:
            path, label = dataset.samples[idx]
            try:
                embedding = extract_embedding(path)
            except RuntimeError as exc:
                print(f"Warning: {exc}; skipping")
                continue
            embeddings.append(embedding)
            labels.append(label)
            pbar.update(1)

    if not embeddings:
        raise RuntimeError(f"No embeddings generated for split '{split_name}'")

    emb_array = np.stack(embeddings).astype(np.float32)
    label_array = np.asarray(labels, dtype=np.int64)
    np.savez(cache_path, embeddings=emb_array, labels=label_array)

    return torch.from_numpy(emb_array), torch.from_numpy(label_array)


def _evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    criterion = nn.CrossEntropyLoss()
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
            total_loss += loss.item() * targets.size(0)

    accuracy = correct / total if total else math.nan
    loss_value = total_loss / total if total else math.nan
    return {"loss": loss_value, "accuracy": accuracy}


def train_feature_classifier(
    data_root: str,
    output_dir: str | Path = "checkpoints/aura_classifier",
    device: str | None = None,
    cfg: ClassifierConfig | None = None,
    cache_dir: str | Path = "cache/aura_embeddings",
) -> Dict[str, Dict[str, float]]:
    cfg = cfg or get_classifier_config()
    fix_randomness(cfg.seed, cfg.deterministic)

    dataset = ImageFolder(data_root)
    labels = np.array(dataset.targets)
    train_idx, valid_idx, test_idx = split_indices(
        labels,
        cfg.train_ratio,
        cfg.valid_ratio,
        cfg.test_ratio,
        cfg.seed,
        cfg.is_stratified,
    )

    cache_dir = Path(cache_dir) / Path(data_root).name
    train_embeddings, train_labels = _compute_embeddings(dataset, train_idx, cache_dir, "train")
    valid_embeddings, valid_labels = _compute_embeddings(dataset, valid_idx, cache_dir, "valid")
    test_embeddings, test_labels = _compute_embeddings(dataset, test_idx, cache_dir, "test")

    embedding_dim = train_embeddings.size(1)
    num_classes = len(dataset.classes)

    train_dataset = TensorDataset(train_embeddings, train_labels)
    valid_dataset = TensorDataset(valid_embeddings, valid_labels)
    test_dataset = TensorDataset(test_embeddings, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=cfg.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)

    torch_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = nn.Linear(embedding_dim, num_classes).to(torch_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    criterion = nn.CrossEntropyLoss()

    metrics: Dict[str, Dict[str, float]] = {}
    best_valid_accuracy = -math.inf
    best_state = None

    start = time.time()
    for epoch in range(cfg.epochs):
        model.train()
        epoch_loss = 0.0
        for inputs, targets in train_loader:
            inputs = inputs.to(torch_device)
            targets = targets.to(torch_device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * targets.size(0)

        train_metrics = _evaluate(model, train_loader, torch_device)
        valid_metrics = _evaluate(model, valid_loader, torch_device)
        metrics[f"epoch_{epoch + 1}"] = {
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "valid_loss": valid_metrics["loss"],
            "valid_accuracy": valid_metrics["accuracy"],
        }

        if valid_metrics["accuracy"] > best_valid_accuracy:
            best_valid_accuracy = valid_metrics["accuracy"]
            best_state = model.state_dict()

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = _evaluate(model, test_loader, torch_device)
    total_time = time.time() - start

    summary = {
        "train_samples": len(train_dataset),
        "valid_samples": len(valid_dataset),
        "test_samples": len(test_dataset),
        "test_loss": test_metrics["loss"],
        "test_accuracy": test_metrics["accuracy"],
        "training_time_sec": total_time,
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "aura_classifier_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as fh:
        json.dump({"history": metrics, "summary": summary}, fh, indent=2)

    model_path = output_dir / "aura_classifier.pt"
    torch.save({"state_dict": model.state_dict(), "embedding_dim": embedding_dim, "num_classes": num_classes}, model_path)

    return {"summary": summary, "history": metrics}


if __name__ == "__main__":
    cfg = get_classifier_config()
    from utils.helper import get_dataset_list

    dataset_list = get_dataset_list()
    dataset_list = list(map(lambda x: x.split('/')[1], dataset_list))

    for data in dataset_list[:-1]:
        train_feature_classifier(
            data_root=f"private/{data}",
            output_dir=f"checkpoints/aura_{data}",
            device=None,
            cfg=cfg,
            cache_dir=f"cache/aura_embeddings/{data}",
        )

