import boto
from botocore import UNSIGNED
from botocore.config import Config
import pyprojroot
import sys
from tqdm import tqdm
sys.path.append(str(pyprojroot.here()))

from bps_utils.s3_utils import copy_s3_to_s3_directory_files 

def main():
    """
    This function retrieves individual files from a specific aws s3 bucket blob/file path and
    saves the files of interest to another s3 bucket.
    """

    # NASA BPS Public Data Bucket source
    source_bucket_name = "nasa-bps-training-data"
    source_s3_path = 'Microscopy/train'
    source_meta_csv_path = f'{source_s3_path}/metadata.csv'
    source_s3_client = boto3.client('s3', config=Config(signature_version=UNSIGNED))

    # Personal Public Data Bucket destination
    dest_bucket_name = "ai4ls-bps-training-data"
    dest_s3_path = 'data'
    dest_s3_client = boto3.client('s3', config=Config(signature_version=UNSIGNED))

    # Copy files from source to destination
    copy_s3_to_s3_directory_files(source_bucket_name, dest_bucket_name, source_s3_path)


if __name__ == "__main__":
    main()