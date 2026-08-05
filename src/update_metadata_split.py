from pathlib import Path
import pandas as pd

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

METADATA_DIR = PROJECT_ROOT / "metadata"

MASTER_CSV = METADATA_DIR / "dataset_metadata.csv"
TRAIN_CSV = METADATA_DIR / "train.csv"
VAL_CSV = METADATA_DIR / "validation.csv"
TEST_CSV = METADATA_DIR / "test.csv"

# ============================================================
# LOAD FILES
# ============================================================

master_df = pd.read_csv(MASTER_CSV)
train_df = pd.read_csv(TRAIN_CSV)
val_df = pd.read_csv(VAL_CSV)
test_df = pd.read_csv(TEST_CSV)

# ============================================================
# CREATE SPLIT COLUMN
# ============================================================

master_df["split"] = ""

# ============================================================
# ASSIGN SPLITS USING image_id
# ============================================================

master_df.loc[
    master_df["image_id"].isin(train_df["image_id"]),
    "split"
] = "train"

master_df.loc[
    master_df["image_id"].isin(val_df["image_id"]),
    "split"
] = "validation"

master_df.loc[
    master_df["image_id"].isin(test_df["image_id"]),
    "split"
] = "test"

# ============================================================
# VERIFY
# ============================================================

print("=" * 60)
print("Split Counts")
print("=" * 60)

print(master_df["split"].value_counts())

# Check if any image was not assigned
missing = master_df[master_df["split"] == ""]

print("\nUnassigned Images :", len(missing))

# ============================================================
# SAVE
# ============================================================

master_df.to_csv(MASTER_CSV, index=False)

print("\nUpdated:")
print(MASTER_CSV)