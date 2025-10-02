"""Utility for preprocessing facial data before training the model."""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Iterable, Optional, Tuple

import kagglehub
import shutil
from PIL import Image, ImageFile, ImageOps, UnidentifiedImageError

from config import data_config
from utils.helper import login_kaggle

ImageFile.LOAD_TRUNCATED_IMAGES = True  # allow processing images with minor corruption

logging.getLogger("kagglehub.clients").setLevel(logging.WARNING)
logging.getLogger("kagglehub").setLevel(logging.ERROR)

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_SIZE = (256, 256)

try:  # Pillow >= 9
    RESAMPLING_FILTER = Image.Resampling.LANCZOS
except AttributeError:  # Pillow < 9
    RESAMPLING_FILTER = Image.LANCZOS


def get_dataset_list() -> list[str]:
    """Return the list of dataset identifiers to download."""

    datasets = data_config.get("dataset", {})
    return datasets.get("face_data", [])


def download_dataset(data_src: str, data_dest: str) -> None:
    """Authenticate to Kaggle, fetch a dataset, and copy it into ``data_dest``."""
    login_kaggle()

    print()
    print("=" * 50)
    print(f"Start Downloading Dataset :\n - {data_src}")
    print("=" * 50)

    path = kagglehub.dataset_download(data_src)

    print("Downloaded Path:", path)
    os.makedirs(data_dest, exist_ok=True)

    shutil.copytree(path, data_dest, dirs_exist_ok=True)
    print(f"Moved to {data_dest}")


def _iter_image_files(directory: Path) -> Iterable[Path]:
    """Yield image files with supported extensions under ``directory``."""

    for file_path in directory.rglob("*"):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
            continue
        yield file_path


def standardize_images(
    source_dir: Path,
    target_dir: Path,
    image_size: Tuple[int, int] = DEFAULT_SIZE,
    output_format: str = "PNG",
    convert_mode: str = "RGB",
) -> None:
    """Convert images into a unified size/format directory.

    Parameters
    ----------
    source_dir
        Directory containing the raw dataset.
    target_dir
        Destination where normalised images will be written.
    image_size
        Target size (width, height) applied with a center crop via ``ImageOps.fit``.
    output_format
        Save format (e.g. ``"PNG"`` or ``"JPEG"``). Determines the file extension.
    convert_mode
        Pillow image mode to convert before saving. ``"RGB"`` is typical for GAN inputs.
    """

    source_dir = Path(source_dir)
    target_dir = Path(target_dir)

    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    target_dir.mkdir(parents=True, exist_ok=True)

    output_suffix = f".{output_format.lower()}"
    processed = 0
    skipped = 0

    for image_path in _iter_image_files(source_dir):
        relative = image_path.relative_to(source_dir)
        destination = target_dir / relative.with_suffix(output_suffix)
        destination.parent.mkdir(parents=True, exist_ok=True)

        try:
            with Image.open(image_path) as img:
                if convert_mode:
                    img = img.convert(convert_mode)
                if image_size:
                    img = ImageOps.fit(img, image_size, method=RESAMPLING_FILTER)
                save_kwargs = {"format": output_format}
                upper_format = output_format.upper()
                if upper_format in {"JPEG", "JPG"}:
                    save_kwargs.setdefault("quality", 95)
                elif upper_format == "PNG":
                    save_kwargs.setdefault("optimize", True)
                img.save(destination, **save_kwargs)
                processed += 1
        except (UnidentifiedImageError, OSError) as exc:
            logging.warning("Skipping image %s: %s", image_path, exc)
            skipped += 1

    logging.info(
        "Standardized %d images from %s into %s (skipped %d).",
        processed,
        source_dir,
        target_dir,
        skipped,
    )


def preprocess_dataset(
    dataset_name: str,
    raw_root: Optional[Path] = None,
    processed_root: Optional[Path] = None,
    image_size: Tuple[int, int] = DEFAULT_SIZE,
    output_format: str = "PNG",
) -> Path:
    """Standardize images for a named dataset located under ``raw_root``.

    Returns the path to the processed dataset directory.
    """

    raw_root = Path(raw_root or "raw_data")
    processed_root = Path(processed_root or "data/processed")

    source_dir = raw_root / dataset_name
    target_dir = processed_root / dataset_name

    standardize_images(source_dir, target_dir, image_size=image_size, output_format=output_format)
    return target_dir


def process_celeba() -> Path:
    """Example entry point for processing the CelebA dataset with default settings."""

    return preprocess_dataset("celeba-dataset")


if __name__ == "__main__":
    for dataset in get_dataset_list():
        dataset_name = dataset.split("/")[-1]
        print(f"Processing dataset: {dataset_name}")
        preprocess_dataset(dataset_name)
