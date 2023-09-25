import csv
import json
import pyprojroot
root = pyprojroot.find_root(pyprojroot.has_dir(".git"))
import os
import shutil
import pathlib
from typing import List
import tqdm as tqdm
import cv2
import numpy as np
import numpy.typing as npt

def convert_16bit_tif_to_8bit_tif_to_jpg(
    tif_file_path: str,
    tif_file_name: str,
    jpg_file_save_path: str,
    jpg_file_save_name: str
    ) -> None:
    """
    Converts a 16-bit tif image to an 8-bit tif image to a jpg.

    Args:
        tif_file_path (str): The path to the 16-bit tif image.
        tif_file_name (str): The name of the 16-bit tif image.
        jpg_file_save_path (str): The path to save the jpg image.
        jpg_file_save_name (str): The name of the 8-bit jpg image.
    
    Returns:
        None
    """
    # open the 16-bit tif image as a numpy array
    tif_file_full_path = os.path.join(tif_file_path, tif_file_name)
    tif_image = cv2.imread(tif_file_full_path, cv2.IMREAD_ANYDEPTH)

    # using opencv, normalize then scale to 255 and convert to uint8
    tif_image_8bit = cv2.normalize(tif_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # save the jpg image
    jpg_file_save_full_path = os.path.join(jpg_file_save_path, jpg_file_save_name)
    cv2.imwrite(jpg_file_save_full_path, tif_image_8bit, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

def is_jpg_file(filepath: str) -> bool:
    """
    Checks if a file is a jpg file.
    Args:
        filepath (str): The path to the file in full.
    Returns:
        bool: True if the file is a jpg file, False otherwise.
    """
    with open(filepath, "rb") as file:
        file_signature = file.read(3)
        return file_signature == b"\xFF\xD8\xFF"

def generate_meta_json_per_from_csv(
    csv_file_name: str = 'filtered_meta.csv',
    csv_file_path: str = os.path.join(root, "data_Gyhi_4hr"),
    json_file_path: str = os.path.join(root, "data_Gyhi_4hr"),
    s3_bucket_data_dir: str = 'data'
    ) -> None :
    """
    Generates a metadata JSON file containing metadata for each image
    for use with Lightly datasource usage from metadata CSV file.

    Args:
        csv_file_name (str): The name of the CSV file containing the metadata.
        csv_file_path (str): The path to the CSV file containing the metadata.
        json_file_path (str): The path to the directory where the JSON file will
        be saved.
    
    Returns:
        None
    """
    # Open the csv file containing the metadata for each image as a row.
    csv_full_path = os.path.join(csv_file_path, csv_file_name)
    with open(csv_full_path, 'r') as csv_file:
        # Parse the CSV file
        csv_reader = csv.DictReader(csv_file)

        # Iterate over each row in the CSV file
        for row in csv_reader:
            # Create a dictionary for the metadata entry
            filestem = os.path.splitext(row['filename'])[0]
            metadata_entry = {
                'file_name': f'{s3_bucket_data_dir}/{filestem}.jpg',
                'type': 'image',
                'metadata': {
                    'dose_Gy': float(row['dose_Gy']),
                    'particle_type': row['particle_type'],
                    'hr_post_exposure': int(row['hr_post_exposure'])
                }
            }

            # Generate the metadata JSON file name
            json_fname = os.path.splitext(row['filename'])[0]
            json_file_name = json_fname + '.json'

            # Write the metadata entry to the JSON file
            json_full_path = os.path.join(json_file_path, json_file_name)
            print(f'json_full_path: {json_full_path}')
            with open(json_full_path, 'w') as json_file:
                json.dump(metadata_entry, json_file, indent=4)

def generate_lightly_schema_json(path_to_save_file: str = os.path.join(root, "data_Gyhi_4hr")):
    """
    This function generates the Lightly schema JSON file that is used to
    configure the Lightly dataset to read custom metadata from the individual
    .tif files.

    Args:
        path_to_save_file (str): The path to the directory where the JSON file will
        be saved.
    Returns:
        None
    """
    # Define the schema list
    schema = [
        {
            "name": "Filename",
            "path": "file_name",
            "defaultValue": "undefined",
            "valueDataType": "CATEGORICAL_STRING"
        },
        {
            "name": "Type",
            "path": "type",
            "defaultValue": "undefined",
            "valueDataType": "CATEGORICAL_STRING"
        },
        {
            "name": "Dose (Gy)",
            "path": "dose_Gy",
            "defaultValue": 0.0,
            "valueDataType": "NUMERIC_FLOAT"
        },
        {
            "name": "Particle Type",
            "path": "particle_type",
            "defaultValue": "nothing",
            "valueDataType": "CATEGORICAL_STRING"
        },
        {
            "name": "Hours Post Exposure",
            "path": "hr_post_exposure",
            "defaultValue": 0,
            "valueDataType": "NUMERIC_INT"
        }
    ]
    json_file_name = 'schema.json'
    json_full_path = os.path.join(path_to_save_file, json_file_name)
    with open(json_full_path, 'w') as json_file:
        json.dump(schema, json_file, indent=4)

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
    train_set_path = pathlib.Path(os.path.join(data_dir, "train_set"))
    print(f'train_set_path: {train_set_path}')
    val_set_path = pathlib.Path(os.path.join(data_dir, "val_set"))
    print(f'val_set_path: {val_set_path}')

    # if train_set_path and val_set_path do not exist, create them
    train_set_path.mkdir(exist_ok=True)
    val_set_path.mkdir(exist_ok=True)

    # define the subdirectories for the jpg and json files using pathlib
    train_set_jpg_path = pathlib.Path(train_set_path, "data")
    train_set_json_path = pathlib.Path(train_set_path, "metadata")
    val_set_jpg_path = pathlib.Path(val_set_path, "data")
    val_set_json_path = pathlib.Path(val_set_path, "metadata")

    # if the subdirectories do not exist, create them
    train_set_jpg_path.mkdir(exist_ok=True)
    train_set_json_path.mkdir(exist_ok=True)
    val_set_jpg_path.mkdir(exist_ok=True)
    val_set_json_path.mkdir(exist_ok=True)

    for file in files:
        if file.endswith(".jpg"):
            # split on the . in the filename to get the filestem
            filestem = file.split(".")[0]

            if np.random.rand() < 0.99:
                if os.path.exists(data_dir / f"{filestem}.json"):
                    shutil.move(data_dir / file, train_set_jpg_path)
                    shutil.move(data_dir / f"{filestem}.json", train_set_json_path)
                else:
                    raise FileNotFoundError(f"{filestem}.json not found")
            else:
                if os.path.exists(data_dir / f"{filestem}.json"):
                    shutil.move(data_dir / file, val_set_jpg_path)
                    shutil.move(data_dir / f"{filestem}.json", val_set_json_path)
                else:
                    raise FileNotFoundError(f"{filestem}.json not found")
    # create a json file in data_dir that records paths to all files in the train_set as a list of dictionaries
    # with a key called path and saved as full_train.json
    with open(data_dir / "full_train.json", "w") as f:
        json.dump([{"path": str(path)} for path in train_set_jpg_path.glob("*.jpg")], f)
   
    # create a json file in data_dir that records paths to all files in the val_set called
    # val.json
    with open(data_dir / "val.json", "w") as f:
        json.dump([{"path": str(path)} for path in val_set_jpg_path.glob("*.jpg")], f)

def dump_lightly_predictions(filenames: List[str], predictions: npt.NDArray, predictions_dir: str) -> None:
    """Dumps model predictions in the Lightly Prediction format.

    Each input image has its own prediction file. The filename is `<image_name>.json`.
    """
    # set path based on predictions_dir
    pred_path = pathlib.Path(os.path.join(root, predictions_dir))
    print(f'pred_path: {pred_path}')
    #pathlib.Path("lightly_predictions") #pass this in as an input parameter
    # if pred_path does not exist, create it
    pred_path.mkdir(exist_ok=True)
    # os.mkdir(pred_path)
    for filename, prediction in zip(filenames, predictions):
        print(f'filename: {filename}, prediction: {prediction}')
        preds_fname = str(pred_path / pathlib.Path(filename).stem) + ".json"
        print(f'preds_fname: {preds_fname}')
        with open(preds_fname, "w") as f:
            pred_list = prediction.tolist()
            # Normalise probabilities again because of precision loss in `to_list`.
            pred_sum = sum(pred_list)
            json.dump(
                {
                    "file_name": filename,
                    "predictions": [
                        {
                            "category_id": int(np.argmax(prediction)),
                            "probabilities": [p / pred_sum for p in pred_list],
                        }
                    ],
                },
                f,
            )
