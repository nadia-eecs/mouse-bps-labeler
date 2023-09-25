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
import sys
sys.path.append(str(root))
from bps_labeler.bps_utils.lightly_utils import (
    load_environment_variables,
    configure_lightly_client,
    set_lightly_dataset,
    configure_input_datasource,
    configure_lightly_datasource,
    run_lightly_worker_active_learning
)

# Main function to execute the steps
def main():
    n_samples = 50
    lightly_dataset_name = "nasa-bps-microscopy"
    load_environment_variables()
    client = configure_lightly_client()
    set_lightly_dataset(client, datasetname=lightly_dataset_name)
    configure_input_datasource(client)
    configure_lightly_datasource(client)
    run_lightly_worker_active_learning(client, num_samples=n_samples)

if __name__ == "__main__":
    main()