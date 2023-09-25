""" Utilities for reading LabelStudio output files.
Taken from https://github.com/lightly-ai/Lightly_LabelStudio_AL
"""
import json
import os
import pathlib
from typing import Dict, List, Tuple

import numpy as np
import numpy.typing as npt


from PIL import Image
import pyprojroot
root = pyprojroot.find_root(pyprojroot.has_dir(".git"))
#IMAGE_SIZE = 224  # Resize images


def read_label_element(label_element: Dict) -> Tuple[str, str]:
    """Parses labels from LabelStudio output data structure."""
    filepath = pathlib.Path("/" + label_element["image"].split("?d=")[-1])
    label = label_element["choice"]
    return filepath.name, label


def read_label_studio_annotation_file(filepath: str) -> Tuple[List[str], List[str]]:
    """Reads labels from LabelStudio output files."""
    # read the label file
    with open(filepath, "r") as json_file:
        data = json.load(json_file)
    return [read_label_element(label_element) for label_element in data]
