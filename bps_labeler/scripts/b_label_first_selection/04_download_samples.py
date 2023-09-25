import pyprojroot
root = pyprojroot.find_root(pyprojroot.has_dir(".git"))
import os
import sys
sys.path.append(str(root))
from bps_labeler.bps_utils.lightly_utils import (
    create_lightly_client,
    get_latest_tag,
    export_filenames_and_urls,
    download_files
)
import pathlib
from dotenv import load_dotenv
load_dotenv(os.path.join(root,".env"))

def main():
    token = os.environ.get("MY_LIGHTLY_TOKEN")
    lightly_dataset_name = "nasa-bps-microscopy"

    client = create_lightly_client(token, lightly_dataset_name)
    latest_tag = get_latest_tag(client)
    # filename_url_mappings is a list of entries with their filenames and read URLs.
    # # For example, [{"fileName": "image1.png", "readUrl": "https://..."}]
    filename_url_mappings = export_filenames_and_urls(client, latest_tag.id)
  
    data_dir = pathlib.Path(os.path.join(root, 'data_Gyhi_4hr'))
    
    output_path = pathlib.Path(os.path.join(data_dir, "samples_for_labeling"))
    print(f'output_path to save samples for labeling: {output_path}')
    output_path.mkdir(exist_ok=True)

    for entry in filename_url_mappings:
        read_url = entry["readUrl"]
        filename = entry["fileName"]
        print(f"Downloading {filename}")
        download_files(read_url, filename, output_path)

if __name__ == "__main__":
    main()




