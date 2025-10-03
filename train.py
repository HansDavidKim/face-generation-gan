### train.py

### Import Libraries
import os
import shutil
from pathlib import Path
import trackio

from config import data_config

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
    sample_nums = [
        (entry.name, _count_images(entry.path))
        for entry in os.scandir(data_root)
        if entry.is_dir()
    ]

    sample_nums.sort(key=lambda x: x[1], reverse=True)
    dataset_name = data_root.name

    private_root = Path("private") / dataset_name
    public_root = Path("public") / dataset_name
    private_root.mkdir(parents=True, exist_ok=True)
    public_root.mkdir(parents=True, exist_ok=True)

    sampled_id = {identity for identity, _ in sample_nums[:pub_id]}
    private_index = 0

    for entry in os.scandir(data_root):
        if not entry.is_dir():
            continue

        source_dir = Path(entry.path)

        if entry.name in sampled_id:
            destination_dir = private_root / str(private_index)
            shutil.copytree(source_dir, destination_dir, dirs_exist_ok=True)
            private_index += 1
        else:
            for file_entry in os.scandir(source_dir):
                if not file_entry.is_file():
                    continue
                if not file_entry.name.lower().endswith(".jpg"):
                    continue

                target_path = public_root / file_entry.name
                if target_path.exists():
                    stem = target_path.stem
                    suffix = target_path.suffix
                    counter = 1
                    while True:
                        candidate = public_root / f"{stem}_{counter}{suffix}"
                        if not candidate.exists():
                            target_path = candidate
                            break
                        counter += 1

                shutil.copy(file_entry.path, target_path)
                

def split_dataset(data_path: str, pub_id: int, option: int):
    if option == RANDOM_IDENTITY:
        pass
    elif option == CARD_IDENTITY:
        _cardinality_sampling(data_path, pub_id)
    pass

if __name__ == '__main__':
    # for testing _count_images
    _cardinality_sampling(data_path='data/celeba-dataset', pub_id = 1000)
