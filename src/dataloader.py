from pathlib import Path

from torch.utils.data import DataLoader

from dataset import TrafficSignDataset
from transforms import train_transform, test_transform
from config import BATCH_SIZE, NUM_WORKERS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
METADATA_DIR = PROJECT_ROOT / "metadata"


def get_dataloaders():

    train_dataset = TrafficSignDataset(
        csv_file=METADATA_DIR / "train.csv",
        project_root=PROJECT_ROOT,
        transform=train_transform
    )

    validation_dataset = TrafficSignDataset(
        csv_file=METADATA_DIR / "validation.csv",
        project_root=PROJECT_ROOT,
        transform=test_transform
    )

    test_dataset = TrafficSignDataset(
        csv_file=METADATA_DIR / "test.csv",
        project_root=PROJECT_ROOT,
        transform=test_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    return train_loader, validation_loader, test_loader


if __name__ == "__main__":

    train_loader, validation_loader, test_loader = get_dataloaders()

    images, labels = next(iter(train_loader))

    print(images.shape)
    print(labels.shape)