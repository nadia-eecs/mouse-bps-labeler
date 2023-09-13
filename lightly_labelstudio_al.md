## Tutorial
This tutorial demonstrates a complete workflow of training a machine learning model with the aid of Active Learning using [Lightly](https://www.lightly.ai) and [Label Studio](https://labelstud.io).

Assume we have a new unlabelled dataset and want to train a new model. We do not want to label all samples because not all of them are valuable. Lightly can help select a good subset of samples to kick off labeling and model training. The loop is as follows:

1. Lightly chooses a subset of the unlabelled samples.
1. This subset is labeled using Label Studio.
1. A machine learning model is trained on the labeled data and generates predictions for the entire dataset.
1. Lightly consumes predictions and performs Active Learning to choose the next batch of samples to be labeled.
1. This new batch of samples is labeled in Label Studio.
1. The machine learning model is re-trained on the enriched labeled dataset and to achieve better performance.


Let's get started!

## 0. Installation and Requirements
Make sure you have an account for the [Lightly Web App](https://app.lightly.ai). 
You also need to know your API token which is shown under your `USERNAME` -> `Preferences`.

Clone this repo and install all Python package requirements in the `requirements.txt` file, e.g. with pip.
```bash
git clone https://github.com/lightly-ai/Lightly_LabelStudio_AL.git
cd Lightly_LabelStudio_AL
pip install -r requirements.txt
```


## 1. Prepare data
We want to train a classifier to predict whether the microscopy image contains a linear arrangement of 53BP1 accumulation on chromatin surrounding DNA damage,
or irradiation induced foci. We use this dataset: [Biological and Physical Sciences (BPS) Microscopy Benchmark Training Dataset](https://aws.amazon.com/marketplace/pp/prodview-6eq625wnwk4b6#overview).

### 1.1 `00_download_Gyhi_4hr_from_s3_source.sh` To Download High Radiation Dosage and 4 Hour Post Exposure Conditions of the BPS Data
Run `00_download_Gyhi_4hr_from_s3_source.sh` to download a subset of the publicly available BPS Mouse Microscopy data locally. In this example, we will be using the portion of the dataset for high radiation (Gy) exposure levels and 4 hours post-radiation and will store the information in directory called `data_Gyhi_4hr`

```sh
chmod +x bps_labeler/data_setup/00_download_Gyhi_4hr_from_s3_source.sh
./bps_labeler/data_setup/00_download_Gyhi_4hr_from_s3_source.sh
```

After downloading the data, you will see the data directory as follows:

```
data_Gyhi_4hr/
├── meta.csv
├── filtered_files.txt
├── filtered_meta.csv
├── P242_73665006707-A6_003_013_proj.tif
├── P242_73665006707-A6_008_034_proj.tif
├── P242_73665006707-A6_009_007_proj.tif
...
```
Since metadata is available in `filtered_meta.csv` we would like to separate out the metadata for use with Lightly which requires a specific format. [Lightly metadata format](https://docs.lightly.ai/docs/work-with-metadata). In other words, all rows in the csv file must be written as a json with the same file stem as the `.tif` images.

### 1.2 `00_generate_json_from_meta_csv.py` To Extract Metadata from CSV to JSON
Run `00_generate_json_from_meta_csv.py` to generate individual json files with the same filestem as the images containing information with respect to the filename, the radiation exposure, the particle, and the post exposure time period. These files will also save to `data_Gyhi_4hr` and you will be able to see the data directory as follows:

```sh
python bps_labeler/data_setup/00_generate_json_from_meta_csv.py
```


```
data_Gyhi_4hr/
├── meta.csv
├── filtered_files.txt
├── filtered_meta.csv
├── P242_73665006707-A6_003_013_proj.json
├── P242_73665006707-A6_003_013_proj.tif
├── P242_73665006707-A6_008_034_proj.json
├── P242_73665006707-A6_008_034_proj.tif
├── P242_73665006707-A6_009_007_proj.json
├── P242_73665006707-A6_009_007_proj.tif
...
```
### 1.3 `01_setup_data_local.py` For Training and Validation Split
Run `01_setup_data_local.py` to split the files into a the directories `train_set` and `val_set` and migrate the files and their corresponding metadata to subdirectories `data` and `metadata`. 

```sh
python bps_labeler/data_setup/01_setup_data_local.py
```

The parent directory will remain `data_Gyhi_4hr` and you will be able to see the data directory as follows:
  
```
data_Gyhi_4hr/
├── train_set/
│   ├── data/
│   │   ├── P242_73665006707-A6_003_013_proj.tif
│   │   ├── P242_73665006707-A6_008_034_proj.tif
│   │   ├── P242_73665006707-A6_009_007_proj.tif
│   │   ...
│   └── meta/
│       ├── P242_73665006707-A6_003_013_proj.json
│       ├── P242_73665006707-A6_008_034_proj.json
│       ├── P242_73665006707-A6_009_007_proj.json
│       ...
├── val_set/
│   ├── data/
│   │   ├── P242_73665006707-G3_005_027_proj.tif
│   │   ...
│   └── meta/
│       ├── P242_73665006707-G3_005_027_proj.json
│       ...
├── filtered_files.txt
├── filtered_meta.csv
├── full_train.json
├── meta.csv
└── val.json
```

After this, note the following files and directories in the current directory:
* `train_set`: Directory that contains all samples to be used for training the model. Here we pretend that these samples are all unlabelled.
* `val_set`: Directory that contains all samples to be used for model validation. Samples are labeled.
* `full_train.json`: JSON file that records paths to all files in `train_set`.
* `val.json`: JSON file that records paths of all files in `val_set`.

These will be used in the following steps.

### 1.2 `02_upload_training_set_Gyhi_4hr_to_s3_dest.sh` Upload Training Samples to Cloud Storage
In this tutorial, samples are stored in the cloud, and Lightly Worker will read the samples from the cloud data source. For details, please refer to [Set Up Your First Dataset](https://docs.lightly.ai/docs/set-up-your-first-dataset). Here we use Amazon S3 as an example.

Under your S3 bucket, create two directories: `data` and `lightly`. We will upload all training samples to `data`. For example, run the [AWS CLI tool](https://aws.amazon.com/cli/):

```sh
chmod +x bps_labeler/data_setup/02_upload_training_set_Gyhi_4hr_to_s3.sh
./bps_labeler/data_setup/02_upload_training_set_Gyhi_4hr_to_s3.sh
```

After uploading the samples, your S3 bucket should look like
```
s3://bucket/
├── lightly/
│   └── .lightly/
│       └── meta/
│           ├── P242_73665006707-A6_003_013_proj.json
│           ├── P242_73665006707-A6_008_034_proj.json
│           ├── P242_73665006707-A6_009_007_proj.json
│           ...
└── data/
    ├── P242_73665006707-A6_003_013_proj.json
    ├── P242_73665006707-A6_008_034_proj.json
    ├── P242_73665006707-A6_009_007_proj.json
    ├── ...
```
########################################################################################## HERE ##########
## 2. Select the first batch of samples for labeling

Now, with all unlabelled data samples in your training dataset, we want to select a good subset, label them, and train our classification model with them. Lightly can do this selection for you in a simple way. The script [run_first_selection.py](./source/run_first_selection.py) does the job for you. You need to first set up Lightly Worker on your machine and put the correct configuration values in the script. Please refer to [Install Lightly](https://docs.lightly.ai/docs/install-lightly) and [Set Up Your First Dataset](https://docs.lightly.ai/docs/set-up-your-first-dataset) for more details.

Run the script after your worker is ready:

```sh
python source/run_first_selection.py
```

In this script, Lightly Worker first creates a dataset named `weather-classification`, selects 30 samples based on embeddings of the training samples, and records them in this dataset. These 30 samples are the ones that we are going to label in the first round. You can see the selected samples in the [Web App](https://app.lightly.ai/).

![First selection.](tutorial/images/init-selection.png)

## 3. Label the selected samples to train a classifier

We do this using the open source labeling tool **Label Studio**, which is a browser-based tool hosted on your machine.
You have already installed it and can run it from the command line. It will need access to your local files. We will first download the selected samples, import them in Label Studio, label them, and export the annotations.

_Curious to get started with Label Studio? Check out [this tutorial](https://labelstud.io/blog/zero-to-one-getting-started-with-label-studio/) for help getting started!_

#### 3.1 Download the selected samples

We can download the selected samples from the Lightly Platform. The [download_samples.py](./source/download_samples.py) script will do everything for you and download the samples to a local directory called `samples_for_labelling`.

```sh
python source/download_samples.py
```

Lightly Worker created a tag for the selected samples. This script pulls information about samples in this tag and downloads the samples.

#### 3.2 Run LabelStudio

Now we can launch LabelStudio.

```sh
export LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true && label-studio start
```

You should see it in your browser. Create an account and log in.

#### 3.3 Configure Storage

Create a new project called "weather-classification".
Then, head to `Settings` -> `Cloud Storage` -> `Add Source Storage` -> `Storage Type`: `Local files`.
Set the `Absolute local path` to the absolute path of directory `samples_for_labelling`.
Enable the option `Treat every bucket object as a source file`.
Then click `Add Storage`. It will show you that you have added a storage.
Now click on `Sync Storage` to finally load the 30 images.

![Configuration of local file input.](tutorial/images/ls-add-storage.png)

##### 3.4 Configure the labeling interface

In the `Settings` -> `Labeling Interface` in the `Code`insert
```xml
<View>
    <Image name="image" value="$image"/>
        <Choices name="choice" toName="image">
        <Choice value="cloudy"/>
        <Choice value="rain"/>
        <Choice value="shine" />
        <Choice value="sunrise" />
    </Choices>
</View>
```
![Configuration of Labeling Interface.](tutorial/images/ls-interface.png)

It tells Label Studio that there is an image classification task with 4 distinct choices.

If you want someone else to help you label the images, navigate to `Settings`->`Instructions` and add some instructions.

#### 3.5 Labelling

Now if you click on your project again, you see 30 tasks and the corresponding images.
Click on `Label All Tasks` and get those 30 images labeled.

**Pro Tip!** Use the keys `1`, `2`, `3`, `4`, on your keyboard as hotkeys to be faster!

#### 3.6 Export labels

Export the labels via `Export` in the format `JSON-MIN`.
Rename the file to `annotation-0.json` and place that in the root directory of this repository.

## 4. Train a model and prepare for active learning

We can train a classification model with the 30 labeled samples. The [train_model_1.py](./source/train_model_1.py) script loads samples from `annotation-0.json` and performs this task.

```sh
python source/train_model_1.py
```

The following steps are performed in this script:
* Load the annotations and the labeled images.
* Load the validation set.
* Train a simple model as in [model.py](./source/model.py).
* Make predictions for all samples for training, including unlabeled samples.
* Dump the predictions in [Lightly Prediction format](https://docs.lightly.ai/docs/prediction-format#prediction-format) into directory `lightly_predictions`.

We can see that the model performance is not good:
```
Training Acc:  60.000           Validation Acc:  19.027
```

It is okay for now. We will improve this. Predictions will be used for active learning.

#### Upload predictions to data source

Lightly Worker also does active learning for you based on predictions. It consumes predictions stored in the data source. We need to place the predictions we just acquired in the data source. For detailed information, please refer to [Predictions Folder Structure](https://docs.lightly.ai/docs/prediction-format#predictions-folder-structure). Here we still use the AWS S3 bucket as an example.

In the `lightly` directory you created earlier in your S3 bucket, you will have a subdirectory `.lightly/predictions` where predictions are kept. You need the following additional files. You can create these files directly by copying the code blocks below.

##### tasks.json
```json
["weather-classification"]
```

We only have one task here, and let's name it as `weather-classification`.

##### schema.json
```json
{
    "task_type": "classification",
    "categories": [
        {
            "id": 0,
            "name": "cloudy"
        },
        {
            "id": 1,
            "name": "rain"
        },
        {
            "id": 2,
            "name": "shine"
        },
        {
            "id": 3,
            "name": "sunrise"
        }
    ]
}
```

Place these files in the `lightly` directory in your bucket along with predictions in the directory `lightly_prediction`.
After uploading these files, your S3 bucket should look like
```
s3://bucket/
├── lightly/
│   └── .lightly/
│       └── predictions/
│           ├── tasks.json
│           └── weather-classification/
│               ├── schema.json
│               ├── cloudy1.json
│               ├── cloudy2.json
│               ├── ...
└── data/
    ├── cloudy1.jpg
    ├── cloudy2.jpg
    ├── ...
```

where files like `cloudy1.json` and `cloudy2.json` are prediction files in `lightly_prediction`.

## 5. Select and label new samples

With the predictions, Lightly Worker can perform active learning and select new samples for us. The [run_second_selection.py](./source/run_second_selection.py) script does the job.

```sh
python source/run_second_selection.py
```

This time, Lightly Worker goes through all training samples again and selects another 30 samples based on active learning scores computed from the predictions we uploaded in the previous step. For more details, please refer to [Selection Scores](https://docs.lightly.ai/docs/selection#scores) and [Active Learning Scorer](https://docs.lightly.ai/docs/active-learning-scorers).

You can see the results in the Web App.

![Second selection.](./tutorial/images/second-selection.png)


#### Label new samples

You can repeat step 3 to label new samples. To import new samples, go to `Settings` -> `Cloud Storage` and then click `Sync Storage` on the Source Cloud Storage you created earlier. A message `Synced 30 task(s)` should show up.

![Sync Storage.](./tutorial/images/ls-sync-storage.png)

Then, you can go back to the project page and label the new samples. After finishing annotating the samples, export the annotations again. Rename the file to `annotation-1.json` and place that in the root directory of this repository.

## 6. Train a new model with the new samples
Very similar to the script in step 4, script [train_model_2.py](source/train_model_2.py) loads samples from `annotation-1.json` and trains the classification model again with all 60 labeled samples now.

```sh
python source/train_model_2.py
```

The model indeed does better this time on the validation set:
```
Training Acc:  90.000           Validation Acc:  44.248
```

## Celebrate! You've improved your model with the help of Label Studio and Lightly! 
