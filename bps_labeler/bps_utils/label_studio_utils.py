import json
import os
import pathlib
from typing import Dict, List, Tuple

import numpy as np
import numpy.typing as npt


from PIL import Image
import pyprojroot
root = pyprojroot.find_root(pyprojroot.has_dir(".git"))
#IMAGE_SIZE = 224  # Resize images


def read_label_element(label_element: Dict) -> Tuple[str, str]:
    """Parses labels from LabelStudio output data structure."""
    filepath = pathlib.Path("/" + label_element["image"].split("?d=")[-1])
    label = label_element["choice"]
    return filepath.name, label


def read_label_studio_annotation_file(filepath: str) -> Tuple[List[str], List[str]]:
    """Reads labels from LabelStudio output files."""
    # read the label file
    with open(filepath, "r") as json_file:
        data = json.load(json_file)
    return [read_label_element(label_element) for label_element in data]

# def prepare_training_data(annotation_filepath: str) -> None:
#     """Collects labels and filenames from LabelStudio output files.

#     Images still stays in directory `train_set`. `train.json` only contains paths to
#     samples to be used for training. For instance,
#     [{"path": "/path/image1.png", "label": "cloudy"}]

#     `train.json` will be picked up by the scripts for model training to load the
#     actual images.
#     """
#     samples = []
#     train_set_dir = pathlib.Path(os.path.join(root, 'data_Gyhi_4hr', "train_set", "data"))#"train_set")
#     print(f'train_set_dir: {train_set_dir}')
#     for filename, label in read_label_studio_annotation_file(annotation_filepath):
#         samples.append({"path": str(train_set_dir / filename), "label": label})
#     train_sample_json_path = os.path.join(root, 'data_Gyhi_4hr', "train_sample.json")

#     with open(train_sample_json_path, "w") as f:
#         json.dump(samples, f)


# def load_data(sample_json_path: str=os.path.join(root, 'data_Gyhi_4hr', "train_sample.json")) -> Tuple[List[Image.Image], List[str], List[str]]:
#     """Loads image data.

#     Paths to samples to be used for training are loaded from the json file created in
#     `prepare_training_data`.
#     """
#     with open(sample_json_path) as f:
#         sample_list = json.load(f)

#     all_images, all_labels = [], []
#     filenames = []

#     for sample in sample_list:
#         all_images.append(
#             Image.open(sample["path"])
#             .convert("RGB")
#             .resize((IMAGE_SIZE, IMAGE_SIZE), resample=Image.LANCZOS)
#         )
#         all_labels.append(sample["label"])
#         filenames.append(pathlib.Path(sample["path"]).name)
#     return all_images, all_labels, filenames

# def load_data_preds(sample_json_path: str=os.path.join(root, 'data_Gyhi_4hr', "train_sample.json")) -> Tuple[List[Image.Image], List[str]]:
#     """Loads image data.

#     Paths to samples to be used for training are loaded from the json file created in
#     `prepare_training_data`.
#     """
#     with open(sample_json_path) as f:
#         sample_list = json.load(f)

#     all_images = []#, all_labels = [], []
#     filenames = []

#     for sample in sample_list:
#         all_images.append(
#             Image.open(sample["path"])
#             .convert("RGB")
#             .resize((IMAGE_SIZE, IMAGE_SIZE), resample=Image.LANCZOS)
#         )
#         #print(f'sample["path"]: {sample["path"]}')
#         #all_labels.append(sample["label"])
#         filenames.append(pathlib.Path(sample["path"]).name)
#     return all_images, filenames #all_labels, filenames

# def dump_lightly_predictions(filenames: List[str], predictions: npt.NDArray) -> None:
#     """Dumps model predictions in the Lightly Prediction format.

#     Each input image has its own prediction file. The filename is `<image_name>.json`.
#     """
#     root = pathlib.Path("lightly_predictions")
#     os.mkdir(root)
#     for filename, prediction in zip(filenames, predictions):
#         with open(str(root / pathlib.Path(filename).stem) + ".json", "w") as f:
#             pred_list = prediction.tolist()
#             # Normalise probabilities again because of precision loss in `to_list`.
#             pred_sum = sum(pred_list)
#             json.dump(
#                 {
#                     "file_name": filename,
#                     "predictions": [
#                         {
#                             "category_id": int(np.argmax(prediction)),
#                             "probabilities": [p / pred_sum for p in pred_list],
#                         }
#                     ],
#                 },
#                 f,
#             )