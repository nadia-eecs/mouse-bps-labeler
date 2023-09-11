# 09/10/2023
- [ ] Setup AWS Datasource with delegated access IAM role for Lightly
- [ ] Setup Lightly Datasource under `lightly` directory in `ai4ls-bps-microscopy-data`
- [ ] Created `data` directory in `ai4ls-bps-microscopy-data`
- [ ] Label a small validation dataset using Label Studio directly (~100 samples if possible)
- [ ] Revisit `bps_labeler/01_setup_data_local.py` to divide training and validaton set for Lightly Active Learning Pipeline
- [ ] Incorporate file metadata json to Lightly for use and display in Lightly platform
- [ ] Delete full `Gy=hi`, `hr_post_exposure=4` from aws bucket and reupload the training dataset only
- [ ] Incorporate the label in the format that `01_setup_data_local.py` under the key: `label` for the validation set
- [ ] Profit :moneybag: