""" Retrain ResNet50 model on additionally labeled BPS Tracks dataset to see loss behavior.
@Author: Nadia Ahmed
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
from bps_labeler.dataloader.dataset import BPSTracksDataModule
from bps_labeler.model.resnet50 import ResNet50Classifier
from bps_labeler.bps_utils.bps_tracks_config import BPSTracksConfig



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
    ls_annotation_fname = 'annotation-1.json'
    
    # Instantiate data module
    bps_tracks_dm = BPSTracksDataModule(
        annotation_fpath=os.path.join(config.ls_annotation_dir, ls_annotation_fname),
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

if __name__ == "__main__":
    main()