from torchvision import transforms

# ============================================================
# ImageNet normalization
# ============================================================

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ============================================================
# Training transforms
# ============================================================

train_transform = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.RandomAffine(
        degrees=0,
        translate=(0.10, 0.10)
    ),

    transforms.ColorJitter(
        contrast=0.3
    ),

    transforms.RandomResizedCrop(
        size=224,
        scale=(0.80, 1.00)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD
    )

])

# ============================================================
# Validation / Test transforms
# ============================================================

test_transform = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD
    )

])