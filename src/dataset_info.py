import os
from collections import Counter
from PIL import Image

# -------------------------------------------------
# Change if necessary
# -------------------------------------------------
DATASET_PATH = r"C:\Users\HP\OneDrive\Desktop\Thesis\masud sir\RoadSignResearch\dataset"

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")

total_images = 0
total_opened = 0
total_closed = 0

class_summary = {}
image_formats = Counter()
image_sizes = Counter()
corrupted = []

print("=" * 70)
print("DATASET SUMMARY")
print("=" * 70)

for sign_class in sorted(os.listdir(DATASET_PATH)):

    class_path = os.path.join(DATASET_PATH, sign_class)

    if not os.path.isdir(class_path):
        continue

    opened_count = 0
    closed_count = 0

    for condition in ["opened", "closed"]:

        condition_path = os.path.join(class_path, condition)

        if not os.path.exists(condition_path):
            continue

        for file in os.listdir(condition_path):

            if not file.lower().endswith(IMAGE_EXTENSIONS):
                continue

            image_path = os.path.join(condition_path, file)

            try:
                with Image.open(image_path) as img:
                    image_formats[img.format] += 1
                    image_sizes[img.size] += 1

            except Exception:
                corrupted.append(image_path)

            if condition == "opened":
                opened_count += 1
                total_opened += 1
            else:
                closed_count += 1
                total_closed += 1

    total = opened_count + closed_count
    total_images += total

    class_summary[sign_class] = {
        "opened": opened_count,
        "closed": closed_count,
        "total": total
    }

    print(f"{sign_class:<30} Opened: {opened_count:4}  Closed: {closed_count:4}  Total: {total:4}")

print("\n" + "=" * 70)

print(f"Number of classes : {len(class_summary)}")
print(f"Opened images     : {total_opened}")
print(f"Closed images     : {total_closed}")
print(f"Total images      : {total_images}")

sorted_classes = sorted(
    class_summary.items(),
    key=lambda x: x[1]["total"],
    reverse=True
)

print("\nClasses sorted by total images\n")

for cls, info in sorted_classes:
    print(
        f"{cls:<30} "
        f"{info['total']:4}"
    )

print(f"\nLargest class  : {sorted_classes[0][0]} ({sorted_classes[0][1]['total']} images)")
print(f"Smallest class : {sorted_classes[-1][0]} ({sorted_classes[-1][1]['total']} images)")

print("\nImage Formats")
for fmt, count in image_formats.items():
    print(f"{fmt}: {count}")

print("\nMost Common Image Sizes")
for size, count in image_sizes.most_common(10):
    print(f"{size} : {count}")

print(f"\nCorrupted Images : {len(corrupted)}")

# -------------------------------------------------
# Save report
# -------------------------------------------------

os.makedirs("../results", exist_ok=True)

report = "../results/dataset_summary.txt"

with open(report, "w", encoding="utf-8") as f:

    f.write("DATASET SUMMARY\n\n")

    for cls, info in class_summary.items():
        f.write(
            f"{cls:30} "
            f"Opened={info['opened']:4} "
            f"Closed={info['closed']:4} "
            f"Total={info['total']:4}\n"
        )

    f.write("\n")
    f.write(f"Classes : {len(class_summary)}\n")
    f.write(f"Opened  : {total_opened}\n")
    f.write(f"Closed  : {total_closed}\n")
    f.write(f"Total   : {total_images}\n")

print("\nReport saved successfully.")