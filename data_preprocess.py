"""Utility for preprocessing facial data before training the model."""
from __future__ import annotations

import logging
import os
import random
import tarfile
from collections import defaultdict
from functools import lru_cache
from itertools import count
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional, Tuple

import kagglehub
import shutil
from PIL import Image, ImageFile, ImageOps, UnidentifiedImageError
from tqdm import tqdm

from config import data_config
from utils.helper import login_kaggle

ImageFile.LOAD_TRUNCATED_IMAGES = True  # allow processing images with minor corruption

logging.getLogger("kagglehub.clients").setLevel(logging.WARNING)
logging.getLogger("kagglehub").setLevel(logging.ERROR)

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_SIZE = (224, 224)

try:  # Pillow >= 9
    RESAMPLING_FILTER = Image.Resampling.LANCZOS
except AttributeError:  # Pillow < 9
    RESAMPLING_FILTER = Image.LANCZOS


def _unique_destination(directory: Path, stem: str, suffix: str) -> Path:
    """Return a destination path, adding a counter when ``stem`` already exists."""

    candidate = directory / f"{stem}{suffix}"
    if not candidate.exists():
        return candidate

    for idx in count(1):
        candidate = directory / f"{stem}_{idx}{suffix}"
        if not candidate.exists():
            return candidate


def _safe_extract_tar(tar: tarfile.TarFile, target_dir: Path) -> None:
    """Safely extract a tar archive while preventing path traversal."""

    target_dir = target_dir.resolve()
    for member in tar.getmembers():
        member_path = (target_dir / member.name).resolve()
        if not str(member_path).startswith(str(target_dir)):
            raise RuntimeError(f"Unsafe path detected in archive extraction: {member.name}")
    tar.extractall(path=target_dir)


IdentityResolver = Callable[[Path], Optional[str]]
IdentityResolverFactory = Callable[[Path], Optional[IdentityResolver]]
SourcePreparer = Callable[[Path], Path]


def _parent_directory_identity(image_path: Path) -> Optional[str]:
    """Use the immediate parent directory name as an identity label."""

    name = image_path.parent.name
    return name or None


@lru_cache(maxsize=None)
def _load_celeba_identity_map(identity_file: str) -> Dict[str, str]:
    """Load the identity mapping from the official CelebA identity file."""

    path = Path(identity_file)

    if not path.exists():
        raise FileNotFoundError(f"CelebA identity file not found: {identity_file}")

    mapping: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 2:
                logging.warning("Unexpected CelebA identity line: %s", line)
                continue
            image_name, identity = parts
            mapping[image_name] = identity
    return mapping


def _celeba_identity_resolver(source_dir: Path) -> IdentityResolver:
    identity_map = _load_celeba_identity_map(str(source_dir / "identity_CelebA.txt"))

    def resolver(image_path: Path) -> Optional[str]:
        return identity_map.get(image_path.name)

    return resolver


IDENTITY_RESOLVERS: Dict[str, Optional[IdentityResolverFactory]] = {
    "facescrub-full": lambda _source: _parent_directory_identity,
    "pubfig83": lambda _source: _parent_directory_identity,
    "celeba-dataset": _celeba_identity_resolver,
}


def _prepare_pubfig_source(source_dir: Path) -> Path:
    """Ensure the PubFig archive is extracted and return the directory with images."""

    archive_path = source_dir / "pubfig83.v1.tgz"
    extracted_root = source_dir / "pubfig83"

    if archive_path.exists() and not extracted_root.exists():
        logging.info("Extracting %s to %s", archive_path, extracted_root)
        with tarfile.open(archive_path, "r:gz") as tar:
            _safe_extract_tar(tar, source_dir)

    if extracted_root.exists():
        return extracted_root

    return source_dir


DATASET_PREPARERS: Dict[str, SourcePreparer] = {
    "pubfig83": _prepare_pubfig_source,
}


DATASET_MAX_IMAGES: Dict[str, int] = {
    "flickrfaceshq-dataset-ffhq": 30_000,
}


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
    output_format: str = "JPEG",
    convert_mode: str = "RGB",
    identity_resolver: Optional[IdentityResolver] = None,
    rng: Optional[random.Random] = None,
    max_images: Optional[int] = None,
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
    identity_resolver
        Optional callable returning a string identity label per image. When provided,
        identities are shuffled (seeded externally) and remapped to zero-based indices.
    rng
        Optional ``random.Random`` instance used for deterministic shuffling/sampling.
        When ``None``, a fresh generator is created as needed.
    max_images
        Optional cap on the number of images processed. Applies only when
        ``identity_resolver`` is ``None``.
    """

    source_dir = Path(source_dir)
    target_dir = Path(target_dir)

    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    target_dir.mkdir(parents=True, exist_ok=True)

    upper_format = output_format.upper()
    if upper_format in {"JPEG", "JPG"}:
        output_suffix = ".jpg"
    elif upper_format == "PNG":
        output_suffix = ".png"
    else:
        output_suffix = f".{output_format.lower()}"
    processed = 0
    skipped = 0

    key_fn = lambda path: path.relative_to(source_dir).as_posix()
    rng = rng or random.Random(0)

    all_paths = list(_iter_image_files(source_dir))
    all_paths.sort(key=key_fn)

    if not all_paths:
        logging.info("No images found in %s", source_dir)
        return

    if identity_resolver:
        total_paths = len(all_paths)
    else:
        if max_images is not None and len(all_paths) > max_images:
            all_paths = rng.sample(all_paths, k=max_images)
            all_paths.sort(key=key_fn)
        total_paths = len(all_paths)

    progress_desc = f"{source_dir.name} -> {target_dir.name}"

    with tqdm(total=total_paths, desc=progress_desc, unit="img") as progress:

        def process_single(image_path: Path, destination: Path) -> None:
            nonlocal processed, skipped
            try:
                with Image.open(image_path) as img:
                    if convert_mode:
                        img = img.convert(convert_mode)
                    if image_size:
                        img = ImageOps.fit(img, image_size, method=RESAMPLING_FILTER)
                    save_kwargs = {"format": output_format}
                    if upper_format in {"JPEG", "JPG"}:
                        save_kwargs.setdefault("quality", 95)
                    elif upper_format == "PNG":
                        save_kwargs.setdefault("optimize", True)
                    img.save(destination, **save_kwargs)
                    processed += 1
            except (UnidentifiedImageError, OSError) as exc:
                logging.warning("Skipping image %s: %s", image_path, exc)
                skipped += 1
            finally:
                progress.update(1)

        if identity_resolver:
            grouped: Dict[str, list[Path]] = defaultdict(list)
            for image_path in all_paths:
                try:
                    identity = identity_resolver(image_path)
                except Exception as exc:  # pragma: no cover - defensive
                    logging.warning("Failed to resolve identity for %s: %s", image_path, exc)
                    skipped += 1
                    progress.update(1)
                    continue

                if identity is None:
                    logging.warning("Identity resolver returned None for %s", image_path)
                    skipped += 1
                    progress.update(1)
                    continue

                grouped[str(identity)].append(image_path)

            identities = list(grouped.keys())
            rng.shuffle(identities)
            identity_to_index = {identity: idx for idx, identity in enumerate(identities)}

            for identity in identities:
                destination_dir = target_dir / str(identity_to_index[identity])
                destination_dir.mkdir(parents=True, exist_ok=True)
                for image_path in sorted(grouped[identity], key=key_fn):
                    destination = _unique_destination(destination_dir, image_path.stem, output_suffix)
                    process_single(image_path, destination)
        else:
            for image_path in all_paths:
                relative = image_path.relative_to(source_dir)
                destination = target_dir / relative.with_suffix(output_suffix)
                destination.parent.mkdir(parents=True, exist_ok=True)
                process_single(image_path, destination)

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
    output_format: str = "JPEG",
    seed: int = 42,
    max_images: Optional[int] = None,
) -> Path:
    """Standardize images for a named dataset located under ``raw_root``.

    Returns the path to the processed dataset directory.
    """

    raw_root = Path(raw_root or "raw_data")
    processed_root = Path(processed_root or "data")

    source_dir = raw_root / dataset_name
    target_dir = processed_root / dataset_name

    preparer = DATASET_PREPARERS.get(dataset_name)
    if preparer:
        source_dir = preparer(source_dir)

    resolver_factory = IDENTITY_RESOLVERS.get(dataset_name)
    identity_resolver = resolver_factory(source_dir) if resolver_factory else None

    max_images = max_images if max_images is not None else DATASET_MAX_IMAGES.get(dataset_name)

    rng = random.Random(seed)

    standardize_images(
        source_dir,
        target_dir,
        image_size=image_size,
        output_format=output_format,
        identity_resolver=identity_resolver,
        rng=rng,
        max_images=max_images,
    )
    return target_dir


if __name__ == "__main__":
    for dataset in get_dataset_list():
        dataset_name = dataset.split("/")[-1]
        print(f"Processing dataset: {dataset_name}")
        preprocess_dataset(dataset_name)
