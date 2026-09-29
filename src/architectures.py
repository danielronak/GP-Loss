"""Model factories.

Contract (Spec §6.6): a zero-argument factory returning a freshly initialized
nn.Module that maps the fixed input shape to raw logits (batch, num_classes).

Only the Phase 0 model lives here for now. The Arch-A/B/C zoo is built after
Phase 0 has a recorded PASS/FAIL (Kickoff §0).
"""
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """The Phase 1 MNIST SimpleCNN (2 conv + 2 FC), generalized in/out sizes."""

    def __init__(self, num_classes=10, in_channels=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.flatten(1)
        return self.fc2(F.relu(self.fc1(x)))


class TinyMLP(nn.Module):
    """Very small model for CPU unit tests."""

    def __init__(self, num_classes=10, in_features=28 * 28, hidden=32):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(in_features, hidden), nn.ReLU(),
                                 nn.Linear(hidden, num_classes))

    def forward(self, x):
        return self.net(x)


MODEL_REGISTRY = {
    "simple_cnn": lambda: SimpleCNN(num_classes=10),
    "tiny_mlp": lambda: TinyMLP(num_classes=10),
}
