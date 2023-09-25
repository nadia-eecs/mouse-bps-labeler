import os
import sys
import pyprojroot
root = pyprojroot.find_root(pyprojroot.has_dir(".git"))
sys.path.append(str(root))
import torch
from dataclasses import dataclass


@dataclass
class BPSTracksConfig:
    """ Configuration options for BPSTracks data retrieved from Label Studio."""
    data_dir: str = os.path.join(root, 'data_Gyhi_4hr')
    train_dir: str = os.path.join(data_dir, 'train_set', 'data')
    ls_annotation_dir: str = os.path.join(data_dir, 'ls_annotations')
    ls_annotation_fname: str = 'annotation-0.json'
    save_model_dir: str = os.path.join(root, 'model_weights')
    save_pred_dir: str = os.path.join(root, 'lightly_predictions')
    save_wandb_dir: str = os.path.join(root, 'wandb')
    wandb_project_name: str = 'bps_labeler'
    image_size: int = 224
    num_classes: int = 2
    batch_size: int = 32
    lr: float = 0.01
    momentum: float = 0.5
    decay: float = 0.01
    epochs: int = 5
    num_workers: int = 12
    accelerator: str = 'gpu' if torch.cuda.is_available() else 'cpu'
    seed: int = 42
    train_stage: str = 'fit'
    active_learning_stage: str = 'active_learn'