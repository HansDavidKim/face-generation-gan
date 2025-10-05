"""Train a linear classifier on AuraFace embeddings (clean version)."""

from __future__ import annotations
import json, math, time
from pathlib import Path
from typing import Dict, Tuple
import numpy as np, torch
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

    embeddings, labels = [], []
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
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            pred = out.argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            total_loss += loss.item() * y.size(0)

    return {
        "loss": total_loss / total if total else math.nan,
        "accuracy": correct / total if total else math.nan,
    }


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
    train_emb, train_lab = _compute_embeddings(dataset, train_idx, cache_dir, "train")
    valid_emb, valid_lab = _compute_embeddings(dataset, valid_idx, cache_dir, "valid")
    test_emb, test_lab = _compute_embeddings(dataset, test_idx, cache_dir, "test")

    embedding_dim = train_emb.size(1)
    num_classes = len(dataset.classes)

    train_ds = TensorDataset(train_emb, train_lab)
    valid_ds = TensorDataset(valid_emb, valid_lab)
    test_ds = TensorDataset(test_emb, test_lab)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=cfg.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)

    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = nn.Linear(embedding_dim, num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    criterion = nn.CrossEntropyLoss()

    metrics: Dict[str, Dict[str, float]] = {}
    best_acc, best_state = -math.inf, None
    start = time.time()

    for epoch in range(cfg.epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

        tr = _evaluate(model, train_loader, device)
        va = _evaluate(model, valid_loader, device)
        metrics[f"epoch_{epoch+1}"] = {
            "train_loss": tr["loss"], "train_accuracy": tr["accuracy"],
            "valid_loss": va["loss"], "valid_accuracy": va["accuracy"]
        }

        if va["accuracy"] > best_acc:
            best_acc, best_state = va["accuracy"], model.state_dict()

    if best_state:
        model.load_state_dict(best_state)

    te = _evaluate(model, test_loader, device)
    total_time = time.time() - start

    summary = {
        "train_samples": len(train_ds),
        "valid_samples": len(valid_ds),
        "test_samples": len(test_ds),
        "test_loss": te["loss"],
        "test_accuracy": te["accuracy"],
        "training_time_sec": total_time,
    }

    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    with (out / "aura_classifier_metrics.json").open("w", encoding="utf-8") as f:
        json.dump({"history": metrics, "summary": summary}, f, indent=2)

    torch.save({
        "state_dict": model.state_dict(),
        "embedding_dim": embedding_dim,
        "num_classes": num_classes,
    }, out / "aura_classifier.pt")

    return {"summary": summary, "history": metrics}


if __name__ == "__main__":
    # config에서 설정된 값 그대로 사용
    cfg = get_classifier_config()
    # 단일 데이터셋 실행
    train_feature_classifier(
        data_root="private/sample_dataset",
        output_dir="checkpoints/aura_sample",
        device=None,
        cfg=cfg,
        cache_dir="cache/aura_embeddings/sample_dataset",
    )
