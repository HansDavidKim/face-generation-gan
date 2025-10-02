from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

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


class DataConfig:
    """Load dataset configuration from configs/datasets.toml using tomllib."""

    def __init__(self, config_path: Optional[Path] = None) -> None:
        base_dir = Path(__file__).resolve().parent
        self.config_path = config_path or base_dir / "configs" / "datasets.toml"
        self._values: Dict[str, Any] = self._load()

    def _load(self) -> Dict[str, Any]:
        if not self.config_path.exists():
            raise FileNotFoundError(f"datasets.toml not found at {self.config_path}")

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
            raise AttributeError(f"No dataset configuration named '{item}'") from exc


env_config = EnvConfig()
"""Singleton-like configuration object ready for import."""

data_config = DataConfig()
"""Singleton-like dataset configuration loaded from datasets.toml."""
