import pyprojroot
import os
import requests
from dotenv import load_dotenv
from lightly.api import ApiWorkflowClient
from lightly.openapi_generated.swagger_client import (
    DatasetType,
    DatasourcePurpose
)

def load_environment_variables():
    """
    Loads environment variables from the `.env` file.
    """
    root = pyprojroot.find_root(pyprojroot.has_dir(".git"))
    load_dotenv(os.path.join(root, ".env"))

def configure_lightly_client():
    """
    Configures the Lightly client to connect to the API
    """
    token = os.environ.get("MY_LIGHTLY_TOKEN")
    client = ApiWorkflowClient(token=token)
    return client

def create_lightly_dataset(client: ApiWorkflowClient, datasetname: str) -> None:
    """
    Creates a Lightly dataset if it does not exist yet.
    """
    dataset_name = datasetname
    dataset_type = DatasetType.IMAGES
    client.create_dataset(dataset_name=dataset_name, dataset_type=dataset_type)

def set_lightly_dataset(client: ApiWorkflowClient, datasetname: str) -> None:
    """
    Sets the Lightly dataset to the datasetname.
    """
    dataset_name = datasetname
    client.set_dataset_id_by_name(dataset_name=dataset_name)

def configure_input_datasource(client: ApiWorkflowClient) -> None:
    """
    Fetches environment variables to configure the AWS S3 datasource to
    read raw input data from. Datasource must be configured with appropriate
    IAM delegated access.
    """
    resource_path = os.environ.get("S3_RESOURCE_PATH")
    region = os.environ.get("S3_REGION")
    role_arn = os.environ.get("S3_ROLE_ARN")
    external_id = os.environ.get("S3_EXTERNAL_ID")
    purpose = DatasourcePurpose.INPUT
    client.set_s3_delegated_access_config(
        resource_path=resource_path,
        region=region,
        role_arn=role_arn,
        external_id=external_id,
        purpose=purpose
    )

def configure_lightly_datasource(client: ApiWorkflowClient):
    """
    Fetches environment variables to configure a portion of the AWS S3 datasource
    for Lightly to read and write from. Datasource must be configured with appropriate
    IAM delegated access.
    """
    resource_path = os.environ.get("S3_LIGHTLY_PATH")
    region = os.environ.get("S3_REGION")
    role_arn = os.environ.get("S3_ROLE_ARN")
    external_id = os.environ.get("S3_EXTERNAL_ID")
    purpose = DatasourcePurpose.LIGHTLY
    client.set_s3_delegated_access_config(
        resource_path=resource_path,
        region=region,
        role_arn=role_arn,
        external_id=external_id,
        purpose=purpose
    )

def run_lightly_worker(client: ApiWorkflowClient, num_samples: int) -> None:
    """
    Runs a Lightly worker to select samples from the Lightly datasource to
    ensure diverse sampling of embeddings and balanced sampling of the metadata
    value "particle_type".

    Args:
        client (ApiWorkflowClient): The Lightly client to connect to the API.
        num_samples (int): The number of samples to select.
    """
    selection_config = {
        "n_samples": num_samples,
        "strategies": [
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
    scheduled_run_id = client.schedule_compute_worker_run(selection_config=selection_config)
    for run_info in client.compute_worker_run_info_generator(scheduled_run_id=scheduled_run_id):
        print(f"Lightly Worker run is now in state='{run_info.state}' with message='{run_info.message}'")

    if run_info.ended_successfully():
        print("SUCCESS")
    else:
        print("FAILURE")

def run_lightly_worker_active_learning(client: ApiWorkflowClient, num_samples: int) -> None:
    """
    Runs a Lightly worker to select samples using active learning.

    Args:
        client (ApiWorkflowClient): The Lightly client to connect to the API.
        num_samples (int): The number of samples to select.
    """
    scheduled_run_id = client.schedule_compute_worker_run(
        worker_config={
            "datasource": {
                "process_all": True,
            },
            "enable_training": False,
        },
        selection_config={
            "n_samples": num_samples,
            "strategies": [
                {
                    "input": {
                        "type": "SCORES",
                        "task": "bps-classification",
                        "score": "uncertainty_entropy",
                    },
                    "strategy": {"type": "WEIGHTS"},
                }
            ],
        },
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

def create_lightly_client(token:str, dataset_name:str) -> ApiWorkflowClient:
    """
    Creates a Lightly client to connect to the API.

    Args:
        token (str): The Lightly API token.
        dataset_name (str): The name of the dataset.

    Returns:
        ApiWorkflowClient: The Lightly client to connect to the API.
    """
    # Create the Lightly client to connect to the API.
    client = ApiWorkflowClient(token=token)
    client.set_dataset_id_by_name(dataset_name=dataset_name)
    return client

def get_latest_tag(client: ApiWorkflowClient) -> str:
    """
    Gets the latest tag from the Lightly API.

    Args:
        client (ApiWorkflowClient): The Lightly client to connect to the API.
    Returns:
        str: The latest tag.
    """ 
    # Get the latest tag from the Lightly API.
    return client.get_all_tags()[0]

def export_filenames_and_urls(client: ApiWorkflowClient, tag_id: str) -> None:
    """
    Exports the filenames and URLs of the samples in the Lightly API by tag.

    Args:
        client (ApiWorkflowClient): The Lightly client to connect to the API.
        tag_id (str): The tag ID.
    """ 
    return client.export_filenames_and_read_urls_by_tag_id(tag_id=tag_id)

def download_files(read_url: str, filename: str, output_path: str) -> None:
    """
    Downloads the files from the Lightly API.

    Args:
        read_url (str): The URL to read the file from.
        filename (str): The name of the file.
        output_path (str): The path to the directory where the file will be saved.
    
    Returns:
        None
    """
    response = requests.get(read_url, stream=True)
    with open(os.path.join(output_path, filename), 'wb') as f:
        for data in response.iter_content():
            f.write(data)