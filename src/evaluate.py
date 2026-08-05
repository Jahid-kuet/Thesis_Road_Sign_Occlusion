from pathlib import Path
import time

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from config import NUM_CLASSES, PROJECT_ROOT
from custom_cnn import CustomCNN
from dataloader import get_dataloaders


REPO_ROOT = PROJECT_ROOT.parent
METADATA_DIR = REPO_ROOT / "metadata"
MODELS_DIR = REPO_ROOT / "models"
RESULTS_DIR = REPO_ROOT / "results"
BEST_MODEL_PATH = MODELS_DIR / "custom_cnn_best.pth"
PREDICTIONS_PATH = RESULTS_DIR / "test_predictions.csv"
REPORT_PATH = RESULTS_DIR / "classification_report.csv"
CONFUSION_MATRIX_PATH = RESULTS_DIR / "confusion_matrix.csv"
CONFUSION_MATRIX_IMAGE_PATH = RESULTS_DIR / "confusion_matrix.png"
PER_CLASS_ACCURACY_PATH = RESULTS_DIR / "per_class_accuracy.csv"
SUMMARY_PATH = RESULTS_DIR / "evaluation_summary.txt"


def get_device() -> torch.device:
    """Select GPU when available, otherwise use CPU."""

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_class_names() -> list[str]:
    """Load class names in label order from the metadata mapping file."""

    mapping_path = METADATA_DIR / "class_mapping.csv"
    mapping_frame = pd.read_csv(mapping_path)
    mapping_frame = mapping_frame.sort_values("label")
    return mapping_frame["class_name"].tolist()


def load_best_model(device: torch.device) -> CustomCNN:
    """Create the model and load the saved best checkpoint."""

    model = CustomCNN(num_classes=NUM_CLASSES)

    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Best model file not found: {BEST_MODEL_PATH}"
        )

    state_dict = torch.load(BEST_MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model


def run_inference(model, test_loader, device):
    """Run inference on the test set and collect predictions."""

    test_dataset = test_loader.dataset
    test_metadata = test_dataset.data.reset_index(drop=True)

    all_true_labels = []
    all_predicted_labels = []
    prediction_rows = []

    current_index = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            predicted_labels = torch.argmax(outputs, dim=1)

            batch_size = labels.size(0)
            batch_metadata = test_metadata.iloc[
                current_index:current_index + batch_size
            ]
            current_index += batch_size

            true_labels_cpu = labels.cpu().numpy()
            predicted_labels_cpu = predicted_labels.cpu().numpy()

            all_true_labels.extend(true_labels_cpu.tolist())
            all_predicted_labels.extend(predicted_labels_cpu.tolist())

            for offset, (_, metadata_row) in enumerate(batch_metadata.iterrows()):
                predicted_label = int(predicted_labels_cpu[offset])
                true_label = int(metadata_row["label"])

                prediction_rows.append(
                    {
                        "image_id": int(metadata_row["image_id"]),
                        "filename": metadata_row["filename"],
                        "true_label": true_label,
                        "predicted_label": predicted_label,
                        "true_class_name": metadata_row["class_name"],
                        "predicted_class_name": None,
                        "condition": metadata_row["condition"],
                        "correct": true_label == predicted_label,
                    }
                )

    return all_true_labels, all_predicted_labels, prediction_rows


def add_predicted_class_names(prediction_frame, label_to_class_name):
    """Map predicted labels to human-readable class names."""

    prediction_frame["predicted_class_name"] = prediction_frame[
        "predicted_label"
    ].map(label_to_class_name)
    return prediction_frame


def save_confusion_matrix(confusion_matrix_values, class_names):
    """Save the confusion matrix as both CSV and PNG."""

    confusion_matrix_frame = pd.DataFrame(
        confusion_matrix_values,
        index=class_names,
        columns=class_names,
    )
    confusion_matrix_frame.to_csv(CONFUSION_MATRIX_PATH)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        confusion_matrix_frame,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=True,
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_IMAGE_PATH, dpi=300, bbox_inches="tight")
    plt.close()


def save_classification_report(true_labels, predicted_labels, class_names):
    """Create and save the classification report as a CSV file."""

    report_dict = classification_report(
        true_labels,
        predicted_labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    accuracy_value = report_dict.pop("accuracy")
    report_frame = pd.DataFrame(report_dict).T
    report_frame.loc["accuracy"] = {
        "precision": accuracy_value,
        "recall": accuracy_value,
        "f1-score": accuracy_value,
        "support": len(true_labels),
    }
    report_frame = report_frame[["precision", "recall", "f1-score", "support"]]
    report_frame.to_csv(REPORT_PATH)

    return report_frame, accuracy_value


def save_per_class_accuracy(prediction_frame, class_names):
    """Compute and save accuracy for each class."""

    per_class_rows = []

    for class_name in class_names:
        class_frame = prediction_frame[prediction_frame["true_class_name"] == class_name]
        number_of_test_images = len(class_frame)

        if number_of_test_images > 0:
            accuracy_value = class_frame["correct"].mean()
        else:
            accuracy_value = 0.0

        per_class_rows.append(
            {
                "class_name": class_name,
                "accuracy": accuracy_value,
                "number_of_test_images": number_of_test_images,
            }
        )

    per_class_frame = pd.DataFrame(per_class_rows)
    per_class_frame.to_csv(PER_CLASS_ACCURACY_PATH, index=False)

    return per_class_frame


def calculate_condition_accuracy(prediction_frame, condition_name):
    """Calculate accuracy for a specific condition subset."""

    condition_frame = prediction_frame[prediction_frame["condition"] == condition_name]

    if len(condition_frame) == 0:
        return 0.0

    return condition_frame["correct"].mean()


def save_summary(
    overall_accuracy,
    opened_accuracy,
    closed_accuracy,
    macro_precision,
    macro_recall,
    macro_f1_score,
    number_of_test_images,
    evaluation_time,
):
    """Write a plain-text evaluation summary."""

    summary_text = (
        f"Overall Test Accuracy: {overall_accuracy:.4f}\n"
        f"Opened Accuracy: {opened_accuracy:.4f}\n"
        f"Closed Accuracy: {closed_accuracy:.4f}\n"
        f"Macro Precision: {macro_precision:.4f}\n"
        f"Macro Recall: {macro_recall:.4f}\n"
        f"Macro F1-score: {macro_f1_score:.4f}\n"
        f"Number of Test Images: {number_of_test_images}\n"
        f"Evaluation Time: {evaluation_time:.2f} seconds\n"
        f"Best Model Path: {BEST_MODEL_PATH}\n"
    )

    SUMMARY_PATH.write_text(summary_text, encoding="utf-8")


def main():
    """Evaluate the trained CustomCNN on the test dataset."""

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=================================================")
    print("Evaluation Started")
    print("=================================================")
    print()
    print("Loading model...")

    device = get_device()
    class_names = load_class_names()
    label_to_class_name = dict(enumerate(class_names))
    model = load_best_model(device=device)

    print("Running inference...")
    _, _, test_loader = get_dataloaders()

    evaluation_start_time = time.time()
    true_labels, predicted_labels, prediction_rows = run_inference(
        model=model,
        test_loader=test_loader,
        device=device,
    )
    evaluation_time = time.time() - evaluation_start_time

    print("Computing metrics...")

    prediction_frame = pd.DataFrame(prediction_rows)
    prediction_frame = add_predicted_class_names(
        prediction_frame,
        label_to_class_name,
    )
    prediction_frame.to_csv(PREDICTIONS_PATH, index=False)

    overall_accuracy = accuracy_score(true_labels, predicted_labels)
    report_frame, _ = save_classification_report(
        true_labels=true_labels,
        predicted_labels=predicted_labels,
        class_names=class_names,
    )

    macro_precision = float(report_frame.loc["macro avg", "precision"])
    macro_recall = float(report_frame.loc["macro avg", "recall"])
    macro_f1_score = float(report_frame.loc["macro avg", "f1-score"])

    confusion_matrix_values = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=list(range(NUM_CLASSES)),
    )
    save_confusion_matrix(confusion_matrix_values, class_names)

    save_per_class_accuracy(prediction_frame, class_names)
    opened_accuracy = calculate_condition_accuracy(prediction_frame, "opened")
    closed_accuracy = calculate_condition_accuracy(prediction_frame, "closed")

    save_summary(
        overall_accuracy=overall_accuracy,
        opened_accuracy=opened_accuracy,
        closed_accuracy=closed_accuracy,
        macro_precision=macro_precision,
        macro_recall=macro_recall,
        macro_f1_score=macro_f1_score,
        number_of_test_images=len(prediction_frame),
        evaluation_time=evaluation_time,
    )

    print("Saving files...")
    print()
    print(f"Overall Accuracy: {overall_accuracy:.4f}")
    print(f"Opened Accuracy: {opened_accuracy:.4f}")
    print(f"Closed Accuracy: {closed_accuracy:.4f}")
    print(f"Macro F1-score: {macro_f1_score:.4f}")
    print()
    print("All output file locations")
    print(f"Predictions: {PREDICTIONS_PATH}")
    print(f"Classification Report: {REPORT_PATH}")
    print(f"Confusion Matrix CSV: {CONFUSION_MATRIX_PATH}")
    print(f"Confusion Matrix PNG: {CONFUSION_MATRIX_IMAGE_PATH}")
    print(f"Per-class Accuracy: {PER_CLASS_ACCURACY_PATH}")
    print(f"Evaluation Summary: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()