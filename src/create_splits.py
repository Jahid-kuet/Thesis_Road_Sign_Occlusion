from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

METADATA_DIR = PROJECT_ROOT / "metadata"

INPUT_CSV = METADATA_DIR / "dataset_metadata.csv"

TRAIN_CSV = METADATA_DIR / "train.csv"
VAL_CSV = METADATA_DIR / "validation.csv"
TEST_CSV = METADATA_DIR / "test.csv"

RANDOM_SEED = 42

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv(INPUT_CSV)

print("=" * 70)
print("Original Dataset")
print("=" * 70)

print(f"Total Images : {len(df)}")
print(f"Classes      : {df['class_name'].nunique()}")

# ============================================================
# CREATE STRATIFICATION LABEL
# ============================================================

df["stratify_label"] = (
    df["class_name"] + "_" + df["condition"]
)

# ============================================================
# FIRST SPLIT
# 70% Train
# 30% Temp
# ============================================================

train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    stratify=df["stratify_label"],
    random_state=RANDOM_SEED,
    shuffle=True
)

# ============================================================
# SECOND SPLIT
# Temp -> Validation + Test
# ============================================================

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df["stratify_label"],
    random_state=RANDOM_SEED,
    shuffle=True
)

# ============================================================
# REMOVE STRATIFICATION COLUMN
# ============================================================

train_df = train_df.drop(columns=["stratify_label"])
val_df = val_df.drop(columns=["stratify_label"])
test_df = test_df.drop(columns=["stratify_label"])

# ============================================================
# SAVE CSV
# ============================================================

train_df.to_csv(TRAIN_CSV, index=False)

val_df.to_csv(VAL_CSV, index=False)

test_df.to_csv(TEST_CSV, index=False)

# ============================================================
# PRINT SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("Split Summary")
print("=" * 70)

print(f"Training Images   : {len(train_df)}")
print(f"Validation Images : {len(val_df)}")
print(f"Testing Images    : {len(test_df)}")

print("\nPercentages")

print(f"Train : {len(train_df)/len(df)*100:.2f}%")
print(f"Val   : {len(val_df)/len(df)*100:.2f}%")
print(f"Test  : {len(test_df)/len(df)*100:.2f}%")

print("\nCSV Files Saved")

print(TRAIN_CSV)

print(VAL_CSV)

print(TEST_CSV)

# ============================================================
# VERIFY CLASS DISTRIBUTION
# ============================================================

print("\n")
print("=" * 70)
print("Class Distribution")
print("=" * 70)

summary = pd.DataFrame({

    "Train":
        train_df["class_name"].value_counts(),

    "Validation":
        val_df["class_name"].value_counts(),

    "Test":
        test_df["class_name"].value_counts()

}).fillna(0).astype(int)

print(summary.sort_index())

# ============================================================
# VERIFY CONDITION DISTRIBUTION
# ============================================================

print("\n")
print("=" * 70)
print("Opened / Closed Distribution")
print("=" * 70)

condition_summary = pd.DataFrame({

    "Train":
        train_df["condition"].value_counts(),

    "Validation":
        val_df["condition"].value_counts(),

    "Test":
        test_df["condition"].value_counts()

})

print(condition_summary)

print("\n")
print("=" * 70)
print("Splitting Completed Successfully")
print("=" * 70)