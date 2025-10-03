### data_split.py

### Import Libraries
import os
import random
import shutil

from pathlib import Path

from tqdm import tqdm

from utils.helper import get_dataset_list, get_option_list, get_private_id_num_list

### ============== [ Plan for training ] ==============
### ---------------------------------------------------
### 1. Split Dataset into Private & Public
### 2. Train Classifiers
### 3. Train GAN for each public dataset

### ============== [ Split Option ] ============== 
### ----------------------------------------------
### 1. Random Identity Sampling
### 2. Cardinality-Aware Identity Sampling

RANDOM_IDENTITY = 0
CARD_IDENTITY = 1
NONE = 2

### Helper Function for counting total number of images in directory.
def _count_images(dir_path: str, extension: str='.jpg'):
    return sum(
        1 for entry in os.scandir(dir_path) 
        if entry.is_file() and entry.name.lower().endswith(extension)
    )
    

def _cardinality_sampling(data_path: str, pub_id: int):
    ### Constructing list for (id, num sample) tuple list
    data_root = Path(data_path)
    identity_dirs = sorted([p for p in data_root.iterdir() if p.is_dir()], key=lambda p: p.name)

    sample_nums = [
        (identity_dir.name, _count_images(str(identity_dir)))
        for identity_dir in identity_dirs
    ]

    sample_nums.sort(key=lambda x: x[1], reverse=True)
    dataset_name = data_root.name

    private_root = Path("private") / dataset_name
    public_root = Path("public") / dataset_name
    private_root.mkdir(parents=True, exist_ok=True)
    public_root.mkdir(parents=True, exist_ok=True)

    sampled_id = {identity for identity, _ in sample_nums[:pub_id]}
    private_index = 0

    progress = tqdm(identity_dirs, desc=f"Cardinality split {dataset_name}", unit="id")
    for source_dir in progress:
        identity = source_dir.name
        progress.set_postfix(identity=identity, private=private_index)

        if identity in sampled_id:
            destination_dir = private_root / str(private_index)
            shutil.copytree(source_dir, destination_dir, dirs_exist_ok=True)
            private_index += 1
        else:
            _copy_to_public(source_dir, public_root)

    progress.close()
                
def _random_sampling(data_path: str, pub_id: int, seed: int = 42):
    data_root = Path(data_path)
    identity_dirs = sorted([p for p in data_root.iterdir() if p.is_dir()], key=lambda p: p.name)

    rng = random.Random(seed)
    shuffled_dirs = identity_dirs.copy()
    rng.shuffle(shuffled_dirs)

    sampled_id = {identity_dir.name for identity_dir in shuffled_dirs[:pub_id]}

    dataset_name = data_root.name
    private_root = Path("private") / dataset_name
    public_root = Path("public") / dataset_name
    private_root.mkdir(parents=True, exist_ok=True)
    public_root.mkdir(parents=True, exist_ok=True)

    private_index = 0

    progress = tqdm(identity_dirs, desc=f"Random split {dataset_name}", unit="id")
    for source_dir in progress:
        identity = source_dir.name
        progress.set_postfix(identity=identity, private=private_index)

        if identity in sampled_id:
            destination_dir = private_root / str(private_index)
            shutil.copytree(source_dir, destination_dir, dirs_exist_ok=True)
            private_index += 1
        else:
            _copy_to_public(source_dir, public_root)

    progress.close()


def _copy_to_public(source_dir: Path, public_root: Path) -> None:
    for file_entry in source_dir.iterdir():
        if not file_entry.is_file():
            continue
        if file_entry.suffix.lower() != ".jpg":
            continue

        target_path = public_root / file_entry.name
        counter = 1
        while target_path.exists():
            target_path = public_root / f"{file_entry.stem}_{counter}{file_entry.suffix}"
            counter += 1

        shutil.copy(file_entry, target_path)

def split_dataset(data_path: str, pub_id: int, option: int, seed: int = 42):
    dataset_name = Path(data_path).name

    if option == RANDOM_IDENTITY:
        _random_sampling(data_path, pub_id, seed)
    elif option == CARD_IDENTITY:
        _cardinality_sampling(data_path, pub_id)
    elif option == NONE:
        shutil.copytree(
            data_path,
            f'public/{dataset_name}',
            dirs_exist_ok=True,
        )
    else:
        raise ValueError(f"Unknown split option: {option}")

def split_datasets(seed: int = 42):
    dataset_list = get_dataset_list()
    dataset_list = [f'data/{i.split("/")[1]}' for i in dataset_list]

    option_list = get_option_list()
    id_num_list = get_private_id_num_list()
    
    num_dataset = len(dataset_list)

    with tqdm(total=num_dataset, desc="Splitting datasets", unit="dataset") as progress:
        for i in range(num_dataset):
            dataset_path = dataset_list[i]
            dataset_name = Path(dataset_path).name

            progress.write(
                "\n" + "=" * 50 +
                f"\nStart Splitting Dataset :\n - {dataset_name}\n" +
                "=" * 50
            )

            split_dataset(
                dataset_path,
                id_num_list[i],
                option_list[i],
                seed
            )

            progress.update(1)

if __name__ == '__main__':
    split_datasets()
