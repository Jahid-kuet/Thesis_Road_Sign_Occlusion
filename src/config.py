from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

BATCH_SIZE = 32
NUM_WORKERS = 0
NUM_CLASSES = 12

IMAGE_SIZE = 224

LEARNING_RATE = 0.001

RANDOM_SEED = 42

NUM_EPOCHS = 30

TRAIN_CSV = PROJECT_ROOT / "metadata" / "train.csv"
VALIDATION_CSV = PROJECT_ROOT / "metadata" / "validation.csv"
TEST_CSV = PROJECT_ROOT / "metadata" / "test.csv"

DATASET_ROOT = PROJECT_ROOT / "dataset_top12"

DEVICE = "cuda"