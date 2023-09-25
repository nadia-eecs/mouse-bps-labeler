import datetime
import gc
from typing import List, Tuple, Mapping
import os
import json
import pathlib

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.models as models
import pytorch_lightning as pl
import wandb
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from tqdm import tqdm

class ResNet50Classifier(pl.LightningModule):
    def __init__(self, num_classes: int, pred_path: str, lr: float = 0.01, momentum: float = 0.5, decay: float = 0.01):
        super().__init__()
        self.num_classes = num_classes
        self.pred_path = pred_path
        self.lr = lr
        self.momentum = momentum
        self.decay = decay

        # Load a pretrained ResNet-50 model
        self.resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        # Replace the final fully connected layer for classification
        num_ftrs = self.resnet50.fc.in_features
        self.resnet50.fc = nn.Linear(num_ftrs, self.num_classes)

    def forward(self, x):
        x = self.resnet50(x)
        # Apply softmax to get probabilities
        x = F.softmax(x, dim=1)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.decay)
        return optimizer

    def training_step(self, batch, batch_idx):
        image, label, filename = batch
        output = self(image)
        loss = F.cross_entropy(output, label)
        wandb.log({"train_loss" : loss})

        return loss
    
    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        wandb.log({"avg_train_epoch_loss": avg_loss})

    def predict_active_learning(self, dataloader: DataLoader) -> List[np.ndarray]:
        self.resnet50.eval()
        predictions = []
        filenames = []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Predicting on unlabeled data"):
                image, _, fname = batch
                image = image.to(self.device)
                output = self.resnet50(image)
                # Apply softmax to get probabilities
                output = F.softmax(output, dim=1)
                predictions.append(output.cpu().detach().numpy())

                # unpack fname from tuple b/c of batch size
                fname_list = [f for f in fname]
                filenames.extend(fname_list)

        #self.save_predictions(filenames, predictions)
        return np.concatenate(predictions), filenames
