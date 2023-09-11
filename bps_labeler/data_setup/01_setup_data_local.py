
import json
import os
import pathlib
import pyprojroot
from typing import List

import numpy as np

root = pyprojroot.find_root(pyprojroot.has_dir(".git"))
SEED = 42
np.random.seed(SEED)


def setup_data(data_dir_str: str) -> None:
    """Splits the full dataset into a training set and a validation set.

    The training set will have images that will be used to train the model. Even if
    they are already labelled, we will use them as unlabelled data and use Lightly
    to label them. The validation set will be used to evaluate the model's performance.
    """
    data_dir = pathlib.Path(data_dir_str)
    files = os.listdir(data_dir)

    train_set_path = data_dir / "train_set"
    val_set_path = data_dir / "val_set"
    train_set_path.mkdir(parents=True, exist_ok=True)
    val_set_path.mkdir(parents=True, exist_ok=True)

    train_set = []
    val_set = []

    for file in files:
        filepath = data_dir / file

        # Split the data into train and validation sets
        if np.random.rand() < 0.95:
            new_path = train_set_path / filepath.name
            os.rename(filepath, new_path)
            train_set.append({"path": str(new_path)})
        else:
            new_path = val_set_path / filepath.name
            os.rename(filepath, new_path)
            val_set.append({"path": str(new_path)})

    with open("full_train.json", "w") as f:
        json.dump(train_set, f)
    with open("val.json", "w") as f:
        json.dump(val_set, f)

def main():
    data_dir = root / 'data_Gyhi_4hr'
    setup_data(data_dir)

if __name__ == "__main__":
    main()
