"""
This module creates and configures a dataset on Lightly.
In this code we assume that the AWS bucket contains two directories:
- lightly: where Lightly will read and write data for use with the Lightly platform
- data: where the raw data is stored
"""
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
# Create a new dataset on the Lightly Platform.
client.create_dataset(
    dataset_name="nasa-bps-microscopy",
    dataset_type=DatasetType.IMAGES  # can be DatasetType.VIDEOS when working with videos
)
my_dataset_id = client.dataset_id
print(my_dataset_id)

# Configure the Input Datasource (AWS Bucket) where Lightly will read the raw input data from.
# Lighly requires list and read access to the Input Datasource.
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

# Verify Datasource permissions
permissions = client.list_datasource_permissions()
# Check Lightly access permissions
try:
    assert permissions.can_list
    assert permissions.can_read
    assert permissions.can_write
    assert permissions.can_overwrite
except AssertionError:
    print("Datasources are missing permissions. Potential errors are:", permissions.errors)
