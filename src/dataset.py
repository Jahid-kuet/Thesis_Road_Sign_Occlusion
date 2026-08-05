from pathlib import Path

import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset


class TrafficSignDataset(Dataset):
    """
    Custom PyTorch Dataset for Traffic Sign Classification
    """

    def __init__(self, csv_file, project_root, transform=None):

        self.data = pd.read_csv(csv_file)

        self.project_root = Path(project_root)

        self.transform = transform

    def __len__(self):

        return len(self.data)

    def __getitem__(self, index):

        row = self.data.iloc[index]

        image_path = self.project_root / row["relative_path"]

        image = Image.open(image_path).convert("RGB")

        label = int(row["label"])

        if self.transform is not None:
            image = self.transform(image)

        return image, label