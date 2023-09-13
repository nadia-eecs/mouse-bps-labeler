# 09/10/2023
- [X] Setup AWS Datasource with delegated access IAM role for Lightly
- [X] Setup Lightly Datasource under `lightly` directory in `ai4ls-bps-microscopy-data`
- [X] Created `data` directory in `ai4ls-bps-microscopy-data`
- [ ] Create Scripts to download subset of data locally
  - [X] `00_download_Gyhi_4hr_from_s3_source.sh`
  - [ ] Write filtered meta.csv and separate rows into .json metadata files
  - [ ] Do a train, val split on the data and organize with directory structure required by Lightly platform
- [ ] Label a small validation dataset using Label Studio directly (~100 samples if possible)
- [ ] Revisit `bps_labeler/01_setup_data_local.py` to divide training and validaton set for Lightly Active Learning Pipeline
- [ ] Incorporate file metadata json to Lightly for use and display in Lightly platform
- [ ] Delete full `Gy=hi`, `hr_post_exposure=4` from aws bucket and reupload the training dataset only
- [ ] Incorporate the label in the format that `01_setup_data_local.py` under the key: `label` for the validation set
- [ ] Profit :moneybag:

# 09/11/2023-09/12/2023
- [ ] Create Scripts to download subset of data locally
  - [X] `00_download_Gyhi_4hr_from_s3_source.sh`
  - [X] Write filtered meta.csv and separate rows into .json metadata files
  - [X] Do a train, val split on the data and organize with directory structure required by Lightly platform
- [ ] Label a small validation dataset using Label Studio directly (~100 samples if possible)
- [X] Revisit `bps_labeler/01_setup_data_local.py` to divide training and validaton set for Lightly Active Learning Pipeline
- [X] Incorporate file metadata json to Lightly for use and display in Lightly platform
- [X] Delete full `Gy=hi`, `hr_post_exposure=4` from aws bucket and reupload the training dataset only
- [ ] Incorporate the label in the format that `01_setup_data_local.py` under the key: `label` for the validation set
- [ ] Profit :moneybag:

# 09/12/2023
- [X] Completed data setup with final `-2_upload_training_set_Gyhi_4hr_to_s3_dest.sh`
- [ ] Write a script to start the lightly worker with the token information taken from .env file
- [ ] Troubleshoot `03_run_first_selection.py` for Lightly dataset that already exists and also for proper taking of tokens from .env AWS S3
- [ ] Configure the sampling strategy to select balanced on the particle_type as well diversity of embeddings for self supervised learning