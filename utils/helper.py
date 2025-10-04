"""Helper utilities for facial GAN workflows."""

from __future__ import annotations

import os, random
import numpy as np
import torch

from pathlib import Path
from typing import Optional, Tuple

from config import env_config, data_config

_wandb_login_attempted = False
_wandb_login_succeeded = False
_hf_credentials_cached: Tuple[Optional[str], Optional[str]] | None = None


def login_kaggle():
    """Authenticate against Kaggle using credentials stored in .env."""
    username: Optional[str] = env_config.get("KAGGLE_USERNAME")
    key: Optional[str] = env_config.get("KAGGLE_KEY")

    if not username or not key:
        raise ValueError("Missing KAGGLE_USERNAME or KAGGLE_KEY in .env")

    os.environ['KAGGLE_USERNAME'] = username
    os.environ['KAGGLE_KEY'] = key


def ensure_wandb_login() -> bool:
    """Log into Weights & Biases using credentials from .env if available."""
    global _wandb_login_attempted, _wandb_login_succeeded

    if _wandb_login_attempted:
        return _wandb_login_succeeded

    _wandb_login_attempted = True

    try:
        import wandb  # type: ignore  # noqa: F401
    except ModuleNotFoundError:
        print("wandb requested but not installed; install wandb to enable logging.")
        _wandb_login_succeeded = False
        return False

    api_key = env_config.get("WANDB_KEY") or env_config.get("WANDB_API_KEY")
    if not api_key:
        # No key provided; rely on prior wandb login configuration.
        _wandb_login_succeeded = True
        return True

    os.environ.setdefault("WANDB_API_KEY", api_key)

    try:
        wandb.login(key=api_key, relogin=False)
    except Exception as exc:  # pragma: no cover - network dependent
        print(f"wandb login failed: {exc}")
        _wandb_login_succeeded = False
        return False

    _wandb_login_succeeded = True
    return True


def get_hf_credentials() -> Tuple[Optional[str], Optional[str]]:
    """Return cached Hugging Face (username, token) tuple from .env if available."""
    global _hf_credentials_cached

    if _hf_credentials_cached is not None:
        return _hf_credentials_cached

    username = env_config.get("HUGGINGFACE") or env_config.get("HF_USERNAME")
    token = env_config.get("HF_TOKEN")

    if username:
        username = username.strip()
    if token:
        token = token.strip()

    if token:
        os.environ.setdefault("HF_TOKEN", token)

    _hf_credentials_cached = (username, token)
    return _hf_credentials_cached

def fix_randomness(seed: int, deterministic: bool):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_dataset_list() -> list[str]:
    """Return the list of dataset identifiers to download."""

    datasets = data_config.get("dataset", {})
    return datasets.get("face_data", [])

def get_option_list() -> list[int]:
    datasets = data_config.get("dataset", {})
    return datasets.get("id_option", [])

def get_private_id_num_list() -> list[int]:
    datasets = data_config.get("dataset", {})
    return datasets.get("private_id", [])
