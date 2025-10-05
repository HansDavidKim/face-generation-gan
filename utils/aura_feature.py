"""Utilities for extracting AuraFace embeddings."""

from __future__ import annotations

import threading
from functools import lru_cache
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from huggingface_hub import snapshot_download
from insightface.app import FaceAnalysis


_MODEL_REPO = "fal/AuraFace-v1"
_MODEL_LOCAL_DIR = Path("models/auraface")
_DOWNLOAD_LOCK = threading.Lock()


def _ensure_model_downloaded() -> Path:
    """Download AuraFace model weights if they are not present."""

    with _DOWNLOAD_LOCK:
        local_dir = _MODEL_LOCAL_DIR
        local_dir.mkdir(parents=True, exist_ok=True)

        snapshot_download(
            repo_id=_MODEL_REPO,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
        )
        return local_dir


@lru_cache(maxsize=1)
def _load_face_analysis(device: str) -> FaceAnalysis:
    """Load the AuraFace feature extractor once per process."""

    local_dir = _ensure_model_downloaded()

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    app = FaceAnalysis(name="auraface", providers=providers, root=str(local_dir.parent))

    ctx_id = 0 if device == "cuda" and torch.cuda.is_available() else -1
    app.prepare(ctx_id=ctx_id, det_size=(640, 640))
    return app


def extract_embedding(
    image_path: Path | str,
    *,
    device: Optional[str] = None,
) -> np.ndarray:
    """Return the AuraFace embedding for an image.

    Parameters
    ----------
    image_path: Path or str
        Path to an input image containing a single face.
    device: Optional[str]
        Preferred device (``"cuda"`` or ``"cpu"``). Defaults to ``"cuda"`` if
        available, otherwise CPU.

    Returns
    -------
    np.ndarray
        1D embedding vector. Raises ``RuntimeError`` if no face is detected.
    """

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    app = _load_face_analysis(device)

    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    faces = app.get(image_bgr)
    if not faces:
        raise RuntimeError(f"No face detected in image: {image_path}")

    # Select the largest detected face.
    largest_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    embedding = np.asarray(largest_face.normed_embedding, dtype=np.float32)
    return embedding

