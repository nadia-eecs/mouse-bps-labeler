#!/bin/bash

# Set the bucket name and file name
bucket_name="nasa-bps-training-data"
source_file_name="meta.csv"
source_dir="Microscopy/train"
source_csv_fpath="Microscopy/train/${source_file_name}"
local_dir="data_Gyhi_4hr"
# dest_bucket_name="ai4ls-bps-training-data"
# destination_s3_dir="data"

# Download the meta.csv file if it doesn't exist
if [ ! -f "${local_dir}/${source_file_name}" ]; then
  aws s3 cp s3://${bucket_name}/${source_csv_fpath} ${local_dir}/${source_file_name}
fi

# Filter the file names based on dose_Gy and hr_post_exposure using awk
awk -F ',' '$2 >= 0.82 && $4 == 4 { print $1 }' ${local_dir}/${source_file_name} > ${local_dir}/filtered_files.txt

# Save the filtered meta.csv file
awk -F "," 'NR==1 || ($2 >= 0.82 && $4 == 4)' ${local_dir}/${source_file_name} > ${local_dir}/filtered_meta.csv


# Download the filtered files if they don't exist
while IFS= read -r line; do
  if [ ! -f "${local_dir}/${line}" ]; then
    aws s3 cp s3://${bucket_name}/${source_dir}/${line} ${local_dir}
  fi
done < ${local_dir}/filtered_files.txt

# # Upload the filtered files to S3 destination bucket
# aws s3 cp ../../${local_dir} s3://${dest_bucket_name}/${destination_s3_dir} --recursive --exclude "*" --include "*.tif"


