from pathlib import Path
import csv
import random
import time

import torch
from torch import nn
from tqdm import tqdm

from config import (
    LEARNING_RATE,
    NUM_CLASSES,
    NUM_EPOCHS,
    RANDOM_SEED,
)
from custom_cnn import CustomCNN
from dataloader import get_dataloaders


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
BEST_MODEL_PATH = MODELS_DIR / "custom_cnn_best.pth"
LAST_MODEL_PATH = MODELS_DIR / "custom_cnn_last.pth"
HISTORY_PATH = RESULTS_DIR / "training_history.csv"


def set_seed(seed: int) -> None:
    """Make training as reproducible as possible."""

    random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Use GPU when available, otherwise fall back to CPU."""

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_training_epoch(model, data_loader, criterion, optimizer, device):
    """Train the model for one epoch and return loss and accuracy."""

    model.train()

    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    progress_bar = tqdm(data_loader, desc="Training", leave=False)

    for images, labels in progress_bar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        predictions = torch.argmax(outputs, dim=1)
        correct_predictions += (predictions == labels).sum().item()
        total_samples += batch_size

        progress_bar.set_postfix(loss=loss.item())

    epoch_loss = running_loss / total_samples
    epoch_accuracy = correct_predictions / total_samples

    return epoch_loss, epoch_accuracy


def run_validation_epoch(model, data_loader, criterion, device):
    """Evaluate the model for one epoch and return loss and accuracy."""

    model.eval()

    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        progress_bar = tqdm(data_loader, desc="Validation", leave=False)

        for images, labels in progress_bar:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            predictions = torch.argmax(outputs, dim=1)
            correct_predictions += (predictions == labels).sum().item()
            total_samples += batch_size

            progress_bar.set_postfix(loss=loss.item())

    epoch_loss = running_loss / total_samples
    epoch_accuracy = correct_predictions / total_samples

    return epoch_loss, epoch_accuracy


def save_history(history):
    """Save training history to a CSV file."""

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with HISTORY_PATH.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([
            "epoch",
            "train_loss",
            "train_accuracy",
            "validation_loss",
            "validation_accuracy",
        ])

        for row in history:
            writer.writerow(row)


def main():
    """Train the custom CNN on the traffic sign dataset."""

    set_seed(RANDOM_SEED)
    device = get_device()

    train_loader, validation_loader, _ = get_dataloaders()

    model = CustomCNN(num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    history = []
    best_validation_accuracy = -1.0
    start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")

        train_loss, train_accuracy = run_training_epoch(
            model=model,
            data_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        validation_loss, validation_accuracy = run_validation_epoch(
            model=model,
            data_loader=validation_loader,
            criterion=criterion,
            device=device,
        )

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Train Accuracy: {train_accuracy:.4f}")
        print(f"Validation Loss: {validation_loss:.4f}")
        print(f"Validation Accuracy: {validation_accuracy:.4f}")
        print()

        history.append([
            epoch + 1,
            train_loss,
            train_accuracy,
            validation_loss,
            validation_accuracy,
        ])

        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            torch.save(model.state_dict(), BEST_MODEL_PATH)

    torch.save(model.state_dict(), LAST_MODEL_PATH)
    save_history(history)

    training_time = time.time() - start_time

    print(f"Best Validation Accuracy: {best_validation_accuracy:.4f}")
    print(f"Training Time: {training_time:.2f} seconds")
    print(f"Model Save Location: {BEST_MODEL_PATH}")
    print(f"Last Model Save Location: {LAST_MODEL_PATH}")
    print(f"Training History Location: {HISTORY_PATH}")


if __name__ == "__main__":
    main()