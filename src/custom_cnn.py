import torch
import torch.nn as nn


class CustomCNN(nn.Module):

    def __init__(self, num_classes=12):

        super().__init__()

        # ===========================
        # Feature Extraction
        # ===========================

        self.features = nn.Sequential(

            # Block 1
            nn.Conv2d(
                in_channels=3,
                out_channels=16,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # Block 2
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # Block 3
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # Block 4
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)

        )

        # After four MaxPools:
        # 224
        # ↓
        # 112
        # ↓
        # 56
        # ↓
        # 28
        # ↓
        # 14

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                128 * 14 * 14,
                256
            ),

            nn.ReLU(),

            nn.Linear(
                256,
                128
            ),

            nn.ReLU(),

            nn.Linear(
                128,
                num_classes
            )

        )

    def forward(self, x):

        x = self.features(x)

        x = self.classifier(x)

        return x


# ==========================================================
# TEST
# ==========================================================

if __name__ == "__main__":

    model = CustomCNN(num_classes=12)

    x = torch.randn(4, 3, 224, 224)

    y = model(x)

    print("=" * 60)

    print(model)

    print("=" * 60)

    print("Input Shape :", x.shape)

    print("Output Shape:", y.shape)