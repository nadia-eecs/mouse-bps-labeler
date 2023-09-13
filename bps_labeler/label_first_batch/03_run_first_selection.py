"""
This module creates and configures a dataset on Lightly.
In this code we assume that the AWS bucket contains two directories:
- lightly: where Lightly will read and write data for use with the Lightly platform
- data: where the raw data is stored
"""
from dotenv import load_dotenv
from lightly.api import ApiWorkflowClient
from lightly.openapi_generated.swagger_client import (
    DatasetType,
    DatasourcePurpose
)
import pyprojroot
root = pyprojroot.find_root(pyprojroot.has_dir(".git"))
import os

# Load environment variables from .env file
load_dotenv(os.path.join(root,".env"))

# Configure the client to use the Lightly API with thedatase ID
# Create the Lightly client to connect to the API.
client = ApiWorkflowClient(token=os.environ.get("MY_LIGHTLY_TOKEN"))

client.create_dataset(
    dataset_name="nasa-bps-microscopy",
    dataset_type=DatasetType.IMAGES  # can be DatasetType.VIDEOS when working with videos
)
my_dataset_id = client.dataset_id
print(f'my_dataset_id: {my_dataset_id}')

# Configure the Input Datasource (AWS Bucket) where Lightly will read the raw input data from.
# Lighly requires list and read access to the Input Datasource.
print(f'my s3 resource path: {os.environ.get("MY_S3_RESOURCE_PATH")}')
client.set_s3_delegated_access_config(
    resource_path=os.environ.get("MY_S3_RESOURCE_PATH"),
    region=os.environ.get("MY_S3_REGION"),
    role_arn=os.environ.get("MY_S3_ROLE_ARN"),
    external_id=os.environ.get("MY_S3_EXTERNAL_ID"),
    purpose=DatasourcePurpose.INPUT,
)

# Note: Lightly is agnostic to nested folder structures and can only access
# files that are in the path of the input datasource.

# Configure the Lightly Datasource where Lightly can read and write from.
# In this example, the Lightly bucket will point to a different diectory in
# the same AWS bucket as the Input Datasource.
client.set_s3_delegated_access_config(
    resource_path=os.environ.get("MY_S3_LIGHTLY_PATH"),
    region=os.environ.get("MY_S3_REGION"),
    role_arn=os.environ.get("MY_S3_ROLE_ARN"),
    external_id=os.environ.get("MY_S3_EXTERNAL_ID"),
    purpose=DatasourcePurpose.LIGHTLY,
)

# Create a Lightly Worker run to select the first batch of 50 samples based on image embeddings and balanced on
# the metadata value in the .json file called "particle_type"
scheduled_run_id = client.schedule_compute_worker_run(
    selection_config={
        "n_samples": 50,
        "strategies": [
            # select the first 50 samples such that the distribution of the metadata value "particle_type" is balanced
            {
                "input": {
                    "type": "METADATA",
                    "key": "particle_type",
                },
                "strategy": {
                    "type": "BALANCE",
                    "target": {
                        "Fe": 0.5,
                        "X-ray": 0.5
                    }
                },
            },
            # and that the samples are diverse in terms of image embeddings

            {
                "input": {
                    "type": "EMBEDDINGS",
                },
                "strategy": {
                    "type": "DIVERSITY",
                },
            },
        ]
    }
)
for run_info in client.compute_worker_run_info_generator(
    scheduled_run_id=scheduled_run_id
):
    print(
        f"Lightly Worker run is now in state='{run_info.state}' with message='{run_info.message}'"
    )

if run_info.ended_successfully():
    print("SUCCESS")
else:
    print("FAILURE")

