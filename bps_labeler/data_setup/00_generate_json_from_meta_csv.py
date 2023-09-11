import csv
import json
import pyprojroot
root = pyprojroot.find_root(pyprojroot.has_dir(".git"))
import os
import tqdm as tqdm

def generate_meta_json_per_from_csv(
    csv_file_name: str = 'filtered_meta.csv',
    csv_file_path: str = os.path.join(root, "data_Gyhi_4hr"),
    json_file_path: str = os.path.join(root, "data_Gyhi_4hr")
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
            metadata_entry = {
                'file_name': row['filename'],
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
            with open(json_full_path, 'w') as json_file:
                json.dump(metadata_entry, json_file, indent=4)

def main():
    csv_file_name: str = 'filtered_meta.csv'
    csv_file_path: str = os.path.join(root, "data_Gyhi_4hr")
    json_file_path: str = os.path.join(root, "data_Gyhi_4hr")
    generate_meta_json_per_from_csv(csv_file_name=csv_file_name,
                                    csv_file_path=csv_file_path,
                                    json_file_path=json_file_path)

if __name__ == "__main__":
    main()