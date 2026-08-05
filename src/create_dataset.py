import shutil
from pathlib import Path
from collections import defaultdict
from PIL import Image
import pandas as pd

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = PROJECT_ROOT / "dataset"
OUTPUT_DIR = PROJECT_ROOT / "dataset_top12"
METADATA_DIR = PROJECT_ROOT / "metadata"

TOP_K = 12

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

# ============================================================
# CREATE OUTPUT FOLDERS
# ============================================================

OUTPUT_DIR.mkdir(exist_ok=True)
METADATA_DIR.mkdir(exist_ok=True)

# ============================================================
# STEP 1 : COUNT IMAGES PER CLASS
# ============================================================

class_counts = {}

for class_dir in DATASET_DIR.iterdir():

    if not class_dir.is_dir():
        continue

    total = 0

    for condition in ["opened", "closed"]:

        condition_dir = class_dir / condition

        if not condition_dir.exists():
            continue

        total += len([
            f for f in condition_dir.iterdir()
            if f.suffix.lower() in IMAGE_EXTENSIONS
        ])

    class_counts[class_dir.name] = total

# ============================================================
# STEP 2 : SELECT TOP 12 CLASSES
# ============================================================

sorted_classes = sorted(
    class_counts.items(),
    key=lambda x: x[1],
    reverse=True
)

selected_classes = [c[0] for c in sorted_classes[:TOP_K]]

print("=" * 70)
print("Selected Classes")
print("=" * 70)

for i, cls in enumerate(selected_classes):

    print(f"{i:2d} -> {cls} ({class_counts[cls]} images)")

# ============================================================
# STEP 3 : CLASS MAPPING
# ============================================================

class_mapping = {}

for idx, cls in enumerate(selected_classes):

    class_mapping[cls] = idx

mapping_df = pd.DataFrame({
    "label": list(class_mapping.values()),
    "class_name": list(class_mapping.keys())
})

mapping_df.to_csv(
    METADATA_DIR / "class_mapping.csv",
    index=False
)

# ============================================================
# STEP 4 : COPY IMAGES
# ============================================================

metadata_rows = []

statistics = defaultdict(lambda: {
    "opened": 0,
    "closed": 0,
    "total": 0
})

image_id = 1

for cls in selected_classes:

    source_class = DATASET_DIR / cls
    target_class = OUTPUT_DIR / cls

    for condition in ["opened", "closed"]:

        source_condition = source_class / condition

        if not source_condition.exists():
            continue

        target_condition = target_class / condition
        target_condition.mkdir(parents=True, exist_ok=True)

        files = sorted(source_condition.iterdir())

        for file in files:

            if file.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            destination = target_condition / file.name

            shutil.copy2(file, destination)

            try:

                img = Image.open(destination)
                width, height = img.size

            except Exception:

                width = None
                height = None

            metadata_rows.append({

                "image_id": image_id,

                "label": class_mapping[cls],

                "class_name": cls,

                "condition": condition,

                "width": width,

                "height": height,

                "filename": file.name,

                "relative_path":
                    str(destination.relative_to(PROJECT_ROOT))

            })

            statistics[cls][condition] += 1
            statistics[cls]["total"] += 1

            image_id += 1

# ============================================================
# STEP 5 : DATASET METADATA
# ============================================================

metadata_df = pd.DataFrame(metadata_rows)

metadata_df.to_csv(

    METADATA_DIR / "dataset_metadata.csv",

    index=False

)

# ============================================================
# STEP 6 : CLASS STATISTICS
# ============================================================

stats_rows = []

for cls in selected_classes:

    stats_rows.append({

        "label": class_mapping[cls],

        "class_name": cls,

        "opened": statistics[cls]["opened"],

        "closed": statistics[cls]["closed"],

        "total": statistics[cls]["total"]

    })

stats_df = pd.DataFrame(stats_rows)

stats_df.to_csv(

    METADATA_DIR / "class_statistics.csv",

    index=False

)

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")

print("=" * 70)
print("DATASET CREATED SUCCESSFULLY")
print("=" * 70)

print(f"Selected Classes : {len(selected_classes)}")
print(f"Total Images     : {len(metadata_df)}")

print("\nSaved files")

print("dataset_top12/")
print("metadata/dataset_metadata.csv")
print("metadata/class_mapping.csv")
print("metadata/class_statistics.csv")

print("=" * 70)