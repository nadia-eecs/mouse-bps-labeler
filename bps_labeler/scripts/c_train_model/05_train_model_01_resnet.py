""" Train ResNet50 model on BPS Tracks dataset and save model predictions to resample unlabeled data for active learning
through the Lightly API.
@Author: Nadia Ahmed
Adapted from https://github.com/lightly-ai/Lightly_LabelStudio_AL
"""
import os
import sys
import pyprojroot
root = pyprojroot.find_root(pyprojroot.has_dir(".git"))
sys.path.append(str(root))
import wandb
import torch
import pytorch_lightning as pl
from datetime import datetime
from bps_labeler.bps_utils.data_utils import dump_lightly_predictions
from bps_labeler.dataloader.dataset import BPSTracksDataModule
from bps_labeler.model.resnet50 import ResNet50Classifier
from bps_labeler.bps_utils.bps_tracks_config import BPSTracksConfig
import pickle


def main():
    # Load configuration options
    config = BPSTracksConfig()
    pl.seed_everything(config.seed)

    wandb.init(
        project=config.wandb_project_name,
        dir=config.save_wandb_dir,
        config=
        {
            "architecture":"resnet50",
            "learning_rate":config.lr,
            "batch_size":config.batch_size,
            "epochs":config.epochs
        }
)
    
    # Instantiate data module
    bps_tracks_dm = BPSTracksDataModule(
        annotation_fpath=os.path.join(config.ls_annotation_dir, config.ls_annotation_fname),
        full_train_fpath=os.path.join(config.data_dir, 'full_train.json'),
        batch_size=config.batch_size,
        train_path=config.train_dir,
        image_size=config.image_size,
        num_workers=config.num_workers
        )
    
    # collect labels and filenames from Label Studio output files
    bps_tracks_dm.prepare_data()
    # setup data for training from annotations file
    bps_tracks_dm.setup(config.train_stage)


    # Instantiate model
    model = ResNet50Classifier(
        config.num_classes,
        config.save_pred_dir,
        config.lr,
        config.momentum,
        config.decay,
        )
     
    # Instantiate trainer
    trainer = pl.Trainer(
        gpus=1,
        max_epochs=config.epochs,
        accelerator=config.accelerator
        )
    
    # Train model
    trainer.fit(model, bps_tracks_dm)

    # Save model weights
    now = datetime.now().strftime("%Y%m%d%H%M%S")
    fname = f"{now}_resnet50.pth"

    torch.save(model.state_dict(), os.path.join(config.save_model_dir, fname))

    # Active Learn using trained model on sample on the full training set
    bps_tracks_dm.setup(config.active_learning_stage)

    # fetch predictions
    predictions, filenames= model.predict_active_learning(bps_tracks_dm.active_learn_dataloader())
    pickle.dump(predictions, open(os.path.join(config.save_pred_dir, 'predictions.pkl'), 'wb'))
    pickle.dump(filenames, open(os.path.join(config.save_pred_dir, 'filenames.pkl'), 'wb'))

    # dump predictions
    dump_lightly_predictions(filenames, predictions, config.save_pred_dir)



if __name__ == "__main__":
    main()