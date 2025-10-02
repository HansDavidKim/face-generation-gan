"""Helper utilities for facial GAN workflows."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from config import env_config


def login_kaggle():
    """Authenticate against Kaggle using credentials stored in .env."""
    username: Optional[str] = env_config.get("KAGGLE_USERNAME")
    key: Optional[str] = env_config.get("KAGGLE_KEY")

    if not username or not key:
        raise ValueError("Missing KAGGLE_USERNAME or KAGGLE_KEY in .env")

    os.environ['KAGGLE_USERNAME'] = username
    os.environ['KAGGLE_KEY'] = key
