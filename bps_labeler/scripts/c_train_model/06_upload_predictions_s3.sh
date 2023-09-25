#!/bin/bash

local_predictions_dir="lightly_predictions"
dest_bucket_name="ai4ls-bps-training-data"
destination_s3_predictions_dir="lightly/.lightly/predictions/bps-classification"

# Upload the lightly_predictions to the S3 destination bucket in the lightly/.lightly/predictions directory
aws s3 cp ${local_predictions_dir} s3://${dest_bucket_name}/${destination_s3_predictions_dir} --recursive --exclude "*" --include "*.json"