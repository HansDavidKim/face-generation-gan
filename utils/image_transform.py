"""Image transform utilities shared across classifier training scripts."""

from __future__ import annotations

from typing import Any, Sequence

from torchvision import transforms

from config import train_config


def _select_transform_values(values: Any, index: int, default: Sequence[float]) -> Sequence[float]:
    """Return per-model mean/std values with sensible fallbacks."""

    if isinstance(values, list) and values:
        if all(isinstance(v, (int, float)) for v in values):
            return values
        if index < len(values) and isinstance(values[index], list):
            return values[index]
    return default


def get_transform(
    model_name: str,
    augment: bool | None = None,
    downsample_size: int | None = None,
) -> transforms.Compose:
    """Construct a torchvision transform pipeline for ``model_name``."""

    cfg = train_config.get("classifier", {}) or {}
    models = list(cfg.get("model_list", []))
    cfg_augment = bool(cfg.get("augment", False))
    augment_flag = cfg_augment if augment is None else augment

    if model_name not in models:
        raise ValueError(f"Unsupported classifier model: {model_name}")

    index = models.index(model_name)

    size_list = list(cfg.get("input_size", []))
    try:
        size = int(size_list[index])
    except (IndexError, ValueError, TypeError):
        size = 224

    mean = _select_transform_values(cfg.get("transform_mean"), index, default=[0.5, 0.5, 0.5])
    std = _select_transform_values(cfg.get("transform_std"), index, default=[0.5, 0.5, 0.5])

    transform_steps = []

    if downsample_size:
        transform_steps.append(
            transforms.Resize(downsample_size, interpolation=transforms.InterpolationMode.BICUBIC)
        )

    if augment_flag:
        transform_steps.extend(
            [
                transforms.RandomResizedCrop(size, scale=(0.65, 1.0), ratio=(0.85, 1.2)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.08),
                    ],
                    p=0.6,
                ),
                transforms.RandomApply(
                    [transforms.RandomGrayscale(p=1.0)],
                    p=0.15,
                ),
                transforms.RandomApply(
                    [transforms.RandomRotation(degrees=15)],
                    p=0.7,
                ),
                transforms.RandomApply(
                    [transforms.RandomPerspective(distortion_scale=0.3, p=1.0)],
                    p=0.35,
                ),
                transforms.RandomApply(
                    [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))],
                    p=0.25,
                ),
                transforms.RandomApply(
                    [transforms.RandomAutocontrast()],
                    p=0.25,
                ),
                transforms.RandomApply(
                    [transforms.RandomEqualize()],
                    p=0.1,
                ),
                transforms.RandomApply(
                    [transforms.RandomSolarize(threshold=128)],
                    p=0.1,
                ),
                transforms.ToTensor(),
                transforms.RandomErasing(p=0.3, scale=(0.02, 0.15), ratio=(0.3, 3.3), value="random"),
            ]
        )
    else:
        transform_steps.extend(
            [
                transforms.Resize(size),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
            ]
        )

    #transform_steps.append(transforms.Normalize(mean=mean, std=std))

    return transforms.Compose(transform_steps)
