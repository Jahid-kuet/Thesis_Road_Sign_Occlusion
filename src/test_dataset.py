from pathlib import Path

from dataset import TrafficSignDataset
from transforms import train_transform

PROJECT_ROOT = Path(__file__).resolve().parent.parent

dataset = TrafficSignDataset(

    csv_file=PROJECT_ROOT / "metadata" / "train.csv",

    project_root=PROJECT_ROOT,

    transform=train_transform

)

image, label = dataset[0]

print(image.shape)

print(label)

print(image.min().item())

print(image.max().item())