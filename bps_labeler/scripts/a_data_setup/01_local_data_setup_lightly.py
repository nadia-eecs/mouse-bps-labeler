"""
This script is used to convert the 16-bit tif images to 8-bit tif images
and generate a metadata JSON file containing metadata for each image based
on the metadata CSV file. Finally, it generates a Lightly schema JSON file
for use with the Lightly platform. It then splits the full dataset into a
training set and a validation set and places them in separate folders within
the data directory. It also creates subdirectories for the tif and json
metadata files. The training set will have images that will be used to train
the model after labeling. The validation set will be used to evaluate the
model's performance. It is also unlabelled but we'd like to hold it out for
evaluation.
"""

import pyprojroot
root = pyprojroot.find_root(pyprojroot.has_dir(".git"))
import sys
sys.path.append(str(root))
from bps_labeler.bps_utils.data_utils import (
    convert_16bit_tif_to_8bit_tif_to_jpg,
    generate_meta_json_per_from_csv,
    generate_lightly_schema_json,
    setup_data
)
import os
import numpy as np

def main():

    data_dir = os.path.join(root, 'data_Gyhi_4hr')
    for file in os.listdir(data_dir):
        if file.endswith(".tif"):
            filestem = os.path.splitext(file)[0]
            jpg_file_name = f'{filestem}.jpg'
            # call the function to convert the 16-bit tif to 8-bit tif to jpg
            convert_16bit_tif_to_8bit_tif_to_jpg(tif_file_path=data_dir,
                                          tif_file_name=file,
                                          jpg_file_save_path=data_dir,
                                          jpg_file_save_name=jpg_file_name
                                          )
    
    csv_file_name = 'filtered_meta.csv'

    generate_meta_json_per_from_csv(csv_file_name=csv_file_name,
                                    csv_file_path=data_dir,
                                    json_file_path=data_dir)
    generate_lightly_schema_json(path_to_save_file=data_dir)
    SEED = 42
    np.random.seed(SEED)
    setup_data(data_dir)

if __name__ == "__main__":
    main()