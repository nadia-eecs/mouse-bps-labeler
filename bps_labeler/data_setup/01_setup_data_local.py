
import json
import os
import pathlib
import pyprojroot
from typing import List
import shutil

import numpy as np

root = pyprojroot.find_root(pyprojroot.has_dir(".git"))
SEED = 42
np.random.seed(SEED)


def setup_data(data_dir_str: str) -> None:
    """Splits the full dataset into a training set and a validation set and
    places them in separate folders.

    The training set will have images that will be used to train the model. Even if
    they are already labelled, we will use them as unlabelled data and use Lightly
    to label them. The validation set will be used to evaluate the model's performance.
    """
    data_dir = pathlib.Path(data_dir_str)
    files = os.listdir(data_dir)

    # train_set_path = os.path.join(data_dir, "train_set")
    # val_set_path = os.path.join(data_dir, "val_set")
    train_set_path = pathlib.Path("train_set")
    val_set_path = pathlib.Path("val_set")

    # if train_set_path and val_set_path do not exist, create them
    train_set_path.mkdir(exist_ok=True)
    val_set_path.mkdir(exist_ok=True)

    # define the subdirectories for the tif and json files using pathlib
    train_set_tif_path = pathlib.Path(train_set_path, "data")
    train_set_json_path = pathlib.Path(train_set_path, "metadata")
    val_set_tif_path = pathlib.Path(val_set_path, "data")
    val_set_json_path = pathlib.Path(val_set_path, "metadata")

    # if the subdirectories do not exist, create them
    train_set_tif_path.mkdir(exist_ok=True)
    train_set_json_path.mkdir(exist_ok=True)
    val_set_tif_path.mkdir(exist_ok=True)
    val_set_json_path.mkdir(exist_ok=True)

    # if the directory already exists do not overwrite it
    # if os.path.exists(train_set_path):
    #     print("train_set_path already exists")
    # else:
    #     os.mkdir(train_set_path)
    # if os.path.exists(val_set_path):
    #     print("val_set_path already exists")
    # else: 
    #     os.mkdir(val_set_path)

    # defining the subdirectories for the tif and json files
    # train_set_tif_path = os.path.join(train_set_path, "data")
    # train_set_json_path = os.path.join(train_set_path, "metadata")
    # val_set_tif_path = os.path.join(val_set_path, "data")
    # val_set_json_path = os.path.join(val_set_path, "metadata")

    # if the directory already exists do nothing, else create it
    # if os.path.exists(train_set_tif_path):
    #     print("train_set_tif_path already exists")
    # else: 
    #     os.mkdir(train_set_tif_path)

    # if os.path.exists(train_set_json_path):
    #     print("train_set_json_path already exists")
    # else:
    #     os.mkdir(train_set_json_path)
    
    # if os.path.exists(val_set_tif_path):
    #     print("val_set_tif_path already exists")
    # else: os.mkdir(val_set_tif_path)

    # if os.path.exists(val_set_json_path):
    #     print("val_set_json_path already exists")
    # else:
    #     os.mkdir(val_set_json_path)

    for file in files:
        if file.endswith(".tif"):
            # split on the . in the filename to get the filestem
            filestem = file.split(".")[0]

            if np.random.rand() < 0.99:
                if os.path.exists(data_dir / f"{filestem}.json"):
                    shutil.move(data_dir / file, train_set_tif_path)
                    shutil.move(data_dir / f"{filestem}.json", train_set_json_path)
                else:
                    raise FileNotFoundError(f"{filestem}.json not found")
            else:
                if os.path.exists(data_dir / f"{filestem}.json"):
                    shutil.move(data_dir / file, val_set_tif_path)
                    shutil.move(data_dir / f"{filestem}.json", val_set_json_path)
                else:
                    raise FileNotFoundError(f"{filestem}.json not found")
    # create a json file in data_dir that records paths to all files in the train_set as a list of dictionaries
    # with a key called path and saved as full_train.json
    with open(data_dir / "full_train.json", "w") as f:
        json.dump([{"path": str(path)} for path in train_set_tif_path.glob("*.tif")], f)
   
    # create a json file in data_dir that records paths to all files in the val_set called
    # val.json
    with open(data_dir / "val.json", "w") as f:
        json.dump([{"path": str(path)} for path in val_set_tif_path.glob("*.tif")], f)
   
def main():
    data_dir = root / 'data_Gyhi_4hr'
    setup_data(data_dir)

if __name__ == "__main__":
    main()
