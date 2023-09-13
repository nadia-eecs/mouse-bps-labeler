#!/bin/bash

local_dir="data_Gyhi_4hr"
local_train_dir="train_set"
local_train_data_dir="data"
local_train_meta_dir="metadata"
dest_bucket_name="ai4ls-bps-training-data"
destination_s3_data_dir="data"
destination_s3_metadata_dir="lightly/.lightly/metadata"


# Upload the train_set/data to the S3 destination bucket in the data directory
aws s3 cp ${local_dir}/train_set/data s3://${dest_bucket_name}/${destination_s3_data_dir} --recursive --exclude "*" --include "*.tif"

# Upload the train_set/metadata to the S3 destination bucket in the lightly/.lightly/metadata directory
aws s3 cp ${local_dir}/train_set/metadata s3://${dest_bucket_name}/${destination_s3_metadata_dir} --recursive --exclude "*" --include "*.json"