from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import tomllib


class EnvConfig:
    """Load key/value pairs from a .env file once and expose them as attributes."""

    def __init__(self, env_path: Optional[Path] = None) -> None:
        base_dir = Path(__file__).resolve().parent
        self.env_path = env_path or base_dir / ".env"
        self._values: Dict[str, str] = self._load()

    def _load(self) -> Dict[str, str]:
        if not self.env_path.exists():
            raise FileNotFoundError(f".env file not found at {self.env_path}")

        values: Dict[str, str] = {}
        for raw_line in self.env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue

            key, raw_value = line.split("=", 1)
            key = key.strip()
            value = self._strip_quotes(raw_value.strip())
            values[key] = value
        return values

    @staticmethod
    def _strip_quotes(value: str) -> str:
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            return value[1:-1]
        return value

    def get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        return self._values.get(key, default)

    def reload(self) -> None:
        self._values = self._load()

    def __getattr__(self, item: str) -> str:
        try:
            return self._values[item]
        except KeyError as exc:
            raise AttributeError(f"No environment variable named '{item}'") from exc


class BaseTomlConfig:
    """Common loader for TOML-backed configuration files."""

    filename: str

    def __init__(self, config_path: Optional[Path] = None) -> None:
        base_dir = Path(__file__).resolve().parent
        default_path = base_dir / "configs" / self.filename
        self.config_path = config_path or default_path
        self._values: Dict[str, Any] = self._load()

    def _load(self) -> Dict[str, Any]:
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration not found at {self.config_path}")

        with self.config_path.open("rb") as fh:
            return tomllib.load(fh)

    def get(self, key: str, default: Optional[Any] = None) -> Optional[Any]:
        return self._values.get(key, default)

    def reload(self) -> None:
        self._values = self._load()

    def __getattr__(self, item: str) -> Any:
        try:
            return self._values[item]
        except KeyError as exc:
            raise AttributeError(f"No configuration value named '{item}'") from exc


class DataConfig(BaseTomlConfig):
    """Load dataset configuration from configs/datasets.toml using tomllib."""
    filename = "datasets.toml"


class TrainConfig(BaseTomlConfig):
    """Load training configuration from configs/train.toml."""

    filename = "train.toml"


env_config = EnvConfig()
"""Singleton-like configuration object ready for import."""

data_config = DataConfig()
"""Singleton-like dataset configuration loaded from datasets.toml."""

train_config = TrainConfig()
"""Singleton-like training configuration loaded from train.toml."""


@dataclass
class ClassifierConfig:
    model_names: List[str]
    input_sizes: List[int]
    optimizer_name: str
    learning_rate: float
    momentum: float
    weight_decay: float
    batch_size: int
    epochs: int
    train_ratio: float
    valid_ratio: float
    test_ratio: float
    is_stratified: bool
    early_stopping: bool
    patience: int
    seed: int
    deterministic: bool
    augment: bool
    use_wandb: bool = False
    use_trackio: bool = False
    space_name: Optional[str] = None
    upload_hf: bool = False
    pretrain_enabled: bool = False
    pretrain_dataset: Optional[str] = None
    pretrain_train_split: str = "train"
    pretrain_valid_split: Optional[str] = None
    pretrain_test_split: Optional[str] = None
    pretrain_image_column: Optional[str] = None
    pretrain_label_column: str = "label"
    pretrain_sample_limit: Optional[int] = None
    pretrain_output_dir: Optional[str] = None


def get_classifier_config() -> ClassifierConfig:
    cfg = train_config.get("classifier", {})
    if not isinstance(cfg, dict):
        raise ValueError("[classifier] section must be a table in configs/train.toml")

    model_names: List[str] = list(cfg.get("model_list", []))
    input_sizes: List[int] = list(cfg.get("input_size", []))

    if len(model_names) != len(input_sizes):
        raise ValueError("classifier.model_list and classifier.input_size must have the same length")

    seed_value = cfg.get("seed", cfg.get("random_state", 42))

    pretrain_cfg = cfg.get("pretrain", {}) or {}
    if not isinstance(pretrain_cfg, dict):
        raise ValueError("classifier.pretrain must be a table when provided in configs/train.toml")

    pretrain_dataset = pretrain_cfg.get("dataset")
    pretrain_enabled = bool(pretrain_cfg.get("enabled", bool(pretrain_dataset)))

    def _optional_split(key: str) -> Optional[str]:
        value = pretrain_cfg.get(key)
        if value is None:
            return None
        value = str(value).strip()
        return value or None

    pretrain_train_split = _optional_split("train_split") or "train"
    pretrain_valid_split = _optional_split("valid_split")
    pretrain_test_split = _optional_split("test_split")
    pretrain_image_column = _optional_split("image_column")
    pretrain_label_column = _optional_split("label_column") or "label"

    pretrain_sample_limit = pretrain_cfg.get("sample_limit")
    if pretrain_sample_limit is not None:
        pretrain_sample_limit = int(pretrain_sample_limit)

    pretrain_output_dir = _optional_split("output_dir")

    return ClassifierConfig(
        model_names=model_names,
        input_sizes=input_sizes,
        optimizer_name=str(cfg.get("optimizer", "SGD")),
        learning_rate=float(cfg.get("learning_rate", 1e-3)),
        momentum=float(cfg.get("momentum", 0.0)),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
        batch_size=int(cfg.get("batch_size", 32)),
        epochs=int(cfg.get("epoch", 1)),
        train_ratio=float(cfg.get("train_ratio", 0.8)),
        valid_ratio=float(cfg.get("valid_ratio", 0.1)),
        test_ratio=float(cfg.get("test_ratio", 0.1)),
        is_stratified=bool(cfg.get("is_stratified", True)),
        early_stopping=bool(cfg.get("early_stopping", False)),
        patience=int(cfg.get("patience", 10)),
        seed=int(seed_value),
        deterministic=bool(cfg.get("deterministic", False)),
        augment=bool(cfg.get("augment", True)),
        use_wandb=bool(cfg.get("use_wandb", False)),
        use_trackio=bool(cfg.get("use_trackio", False)),
        space_name=cfg.get("space_name"),
        upload_hf=bool(cfg.get("upload_hf", False)),
        pretrain_enabled=pretrain_enabled,
        pretrain_dataset=pretrain_dataset,
        pretrain_train_split=pretrain_train_split,
        pretrain_valid_split=pretrain_valid_split,
        pretrain_test_split=pretrain_test_split,
        pretrain_image_column=pretrain_image_column,
        pretrain_label_column=pretrain_label_column,
        pretrain_sample_limit=pretrain_sample_limit,
        pretrain_output_dir=pretrain_output_dir,
    )
