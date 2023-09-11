import boto3
from botocore import UNSIGNED
from botocore.config import Config
import io
from io import BytesIO
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import pyprojroot
import sys
from tqdm import tqdm
sys.path.append(str(pyprojroot.here()))


def get_bytesio_from_s3(
    s3_client: boto3.client, bucket_name: str, file_path: str
) -> BytesIO:
    """
    This function retrieves individual files from a specific aws s3 bucket blob/file path as
    a BytesIO object to enable the user to not have to save the file to their local machine.

    args:
        s3_client (boto3.client): s3 client should be configured for open UNSIGNED signature.
        bucket_name (str): name of bucket from AWS open source registry.
        file_path (str): blob/file path name from aws including file name and extension.

    returns:
        BytesIO: BytesIO object from the file contents
    """
    # use the S3 client to read the contents of the file into memory
    response = s3_client.get_object(Bucket=bucket_name, Key=file_path)
    file_contents = response["Body"].read()

    # create a BytesIO object from the file contents
    file_buffer = BytesIO(file_contents)
    return file_buffer


def get_file_from_s3(
    s3_client: boto3.client, bucket_name: str, s3_file_path: str, local_file_path: str
) -> str:
    """
    This function retrieves individual files from a specific aws s3 bucket blob/file path and
    saves the files of interest to a local filepath on the user's machine.

    args:
      s3_client (boto3.client): s3 client should be configured for open UNSIGNED signature.
      bucket_name (str): name of bucket from AWS open source registry.
      s3_file_path (str): blob/file path name from aws
      local_file_path (str): file path for user's local directory.

    returns:
      str: local file path with naming convention of the file that was downloaded from s3 bucket
    """
    # If time: add in error handling for string formatting
    
    # os.makedirs(os.path.join(sys.path[0], local_file_path), exist_ok=True)
    os.makedirs(local_file_path, exist_ok=True)

    # Create local file path with file having the same name as the file in the s3 bucket
    new_file_path = f"{local_file_path}/{s3_file_path.split('/')[-1]}"

    # Download file
    s3_client.download_file(bucket_name, s3_file_path, new_file_path)
    return new_file_path


def save_tiffs_local_from_s3(
    s3_client: boto3.client,
    bucket_name: str,
    s3_path: str,
    local_fnames_meta_path: str,
    save_file_path: str,
) -> None:
    """
    This function retrieves tiff files from a locally stored csv file containing specific aws s3 bucket
    blob/file paths and saves the files of interest the same filepath on the user's machine following
    the same naming convention as the files from s3.

    args:
      s3_client (boto3.client): s3 client should be configured for open UNSIGNED signature.
      bucket_name (str): name of bucket from AWS open source registry.
      s3_path (str): blob/file directory where files of interest reside in s3 from AWS
      local_fnames_meta_path (str): file path for user's local directory containing the csv file containing the blob/file paths
      save_file_path (str): file path for user's local directory where files of interest will be saved
    returns:
      None
    """

    # Get s3_file_paths from local_fnames_meta_path csv file
    df = pd.read_csv(local_fnames_meta_path)
    s3_file_paths = df["filename"].values.tolist()

    # Download files because the meta.csv file entries do not contain the full paths
    for s3_file_path in tqdm(s3_file_paths):
        s3_file_path_full = f"{s3_path}/{s3_file_path}"
        get_file_from_s3(s3_client, bucket_name, s3_file_path_full, save_file_path)
        
def copy_s3_to_s3_directory_files(source_bucket: str, dest_bucket: str, source_dir_name: str) -> None:
    """
    This function copies an object from one s3 bucket to another s3 bucket.
    args:
        source_bucket (str): name of source bucket
        dest_bucket (str): name of destination bucket
        source_dir_name (str): name of source directory
    returns:
        None
    """
    print(f'Copying files from {source_bucket}/{source_dir_name} to {dest_bucket}')
    s3 = boto3.resource('s3', config=Config(signature_version=UNSIGNED))
    source_bucket = s3.Bucket(source_bucket)
    dest_bucket = s3.Bucket(dest_bucket)
    
    for obj in source_bucket.objects.filter(Prefix = f'{source_dir_name}/'):
        file_key = obj.key
        dest_key = file_key.replace(f'{source_dir_name}/', '') # remove source directory from destination key
        dest_path_key = f'{dest_bucket.name}/{dest_key}'
        print(f'Copying {file_key} to {dest_path_key}')

        # Copy object to destination bucket with the same name
        dest_bucket.copy({'Bucket': source_bucket, 'Key': file_key}, dest_path_key)


