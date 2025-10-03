"""Helper utilities for facial GAN workflows."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from config import env_config, data_config


def login_kaggle():
    """Authenticate against Kaggle using credentials stored in .env."""
    username: Optional[str] = env_config.get("KAGGLE_USERNAME")
    key: Optional[str] = env_config.get("KAGGLE_KEY")

    if not username or not key:
        raise ValueError("Missing KAGGLE_USERNAME or KAGGLE_KEY in .env")

    os.environ['KAGGLE_USERNAME'] = username
    os.environ['KAGGLE_KEY'] = key

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