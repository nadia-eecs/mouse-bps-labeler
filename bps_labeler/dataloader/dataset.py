import os
import pytorch_lightning as pl
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import json
from datetime import datetime
from typing import Dict, List
from PIL import Image
import pathlib
import pyprojroot
root = pyprojroot.find_root(pyprojroot.has_dir(".git"))
from bps_labeler.bps_utils.label_studio_utils import read_label_studio_annotation_file
import numpy as np

class BPSTracksDataset(Dataset):
    """ Dataset class for BPSTracks data from Label Studio for
    use with PyTorch DataLoader."""
    def __init__(self,sample_list: List[Dict], transform=None):
        self.sample_list = sample_list
        self.transform = transform
        self.classes = ["track", "no track"]
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        sample = self.sample_list[idx]
        image = Image.open(sample["path"]).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # Check if 'label is present, else return a default dummy label since it will
        # be ignored anyway.
        if "label" in sample:
            label = sample["label"]
            numerical_label = self.class_to_idx[label]

            # One hot encode the label
            one_hot_label = np.zeros(len(self.classes))
            one_hot_label[numerical_label] = 1
            one_hot_label = torch.tensor(one_hot_label, dtype=torch.float32)
        else:
            label = "dummy"
            # placeholder for dummy label
            one_hot_label = torch.tensor([0, 0], dtype=torch.float32)
        
        filename = pathlib.Path(sample["path"]).name
        return image, one_hot_label, filename

class BPSTracksDataModule(pl.LightningDataModule):
    """ PyTorch Lightning DataModule class for BPSTracksDataset."""
    def __init__(self, annotation_fpath: str, full_train_fpath: str, batch_size: int, train_path:str, image_size: int, num_workers: int):
        super().__init__()
        self.annotation_filepath = annotation_fpath
        self.full_train_json_path = full_train_fpath
        self.batch_size = batch_size
        self.sample_list = []
        self.train_path = train_path
        self.image_size = image_size
        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size), interpolation=Image.LANCZOS),
            transforms.ToTensor(),
        ])
        self.num_workers = num_workers

    def prepare_data(self):
        """Collects labels and filenames from LabelStudio output files.

        Images still stays in directory `train_set`. `train.json` only contains paths to
        samples to be used for training. For instance,
        [{"path": "/path/image1.png", "label": "cloudy"}]

        `train.json` will be picked up by the scripts for model training to load the
        actual images.
        """
        for filename, label in read_label_studio_annotation_file(self.annotation_filepath):
            self.sample_list.append({"path": os.path.join(self.train_path, filename), "label": label})
        now = datetime.now().strftime("%Y%m%d%H")
        fname = f"{now}_train.json"
        print(f"Saving {fname} to {self.train_path}.")

        with open(fname, "w") as f:
            json.dump(self.sample_list, f)
            print(f"{fname} is saved successfully to {self.train_path}.")
        
    def setup(self, stage=None):
        """ Instantiates the dataset based on the stage: training or active."""
        if stage == "fit" or stage is None:
            self.train_dataset = BPSTracksDataset(self.sample_list, self.transform)
        elif stage == "active_learn":
            self.full_train_list = json.load(open(self.full_train_json_path))
            self.active_learn_dataset = BPSTracksDataset(self.full_train_list, self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
    
    def active_learn_dataloader(self):
        return DataLoader(self.active_learn_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
