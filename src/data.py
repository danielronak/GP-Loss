"""Datasets held as device-resident tensors.

At MNIST scale a DataLoader (PIL decode + collate per batch) costs far more
than the forward pass itself. Here the whole dataset is loaded once as a
normalized tensor on the GPU and batches are index slices.
"""
from dataclasses import dataclass

import torch

_STATS = {
    "mnist": (0.1307, 0.3081),
    "fashion_mnist": (0.2860, 0.3530),
}


@dataclass
class TensorData:
    x: torch.Tensor
    y: torch.Tensor
    num_classes: int

    def __post_init__(self):
        # Spec §7 pitfall 1: labels must be contiguous 0..num_classes-1.
        if self.y.numel():
            lo, hi = int(self.y.min()), int(self.y.max())
            assert lo >= 0 and hi < self.num_classes, (
                f"labels span [{lo}, {hi}] but num_classes={self.num_classes}; remap labels first")

    def __len__(self):
        return self.x.size(0)

    def to(self, device):
        return TensorData(self.x.to(device), self.y.to(device), self.num_classes)

    def subset(self, indices):
        return TensorData(self.x[indices], self.y[indices], self.num_classes)

    def take(self, n, seed=0):
        """Fixed random subset of size n (all of it if n is None or >= len)."""
        if n is None or n >= len(self):
            return self
        g = torch.Generator().manual_seed(seed)
        idx = torch.randperm(len(self), generator=g)[:n].to(self.x.device)
        return self.subset(idx)


def load_torchvision(name, root, device):
    """Return (train, test) TensorData for 'mnist' or 'fashion_mnist'."""
    import torchvision

    cls = {"mnist": torchvision.datasets.MNIST,
           "fashion_mnist": torchvision.datasets.FashionMNIST}[name]
    mean, std = _STATS[name]
    out = []
    for train in (True, False):
        ds = cls(root=root, train=train, download=True)
        x = (ds.data.float().div(255.0).sub(mean).div(std)).unsqueeze(1)
        y = ds.targets.long()
        out.append(TensorData(x, y, 10).to(device))
    return tuple(out)


def train_val_split(train, val_size, split_seed=0):
    """Fixed train/val split. The official test set is never used for fitness."""
    g = torch.Generator().manual_seed(split_seed)
    perm = torch.randperm(len(train), generator=g).to(train.x.device)
    return train.subset(perm[val_size:]), train.subset(perm[:val_size])


def make_synthetic(n, num_classes=10, shape=(1, 28, 28), seed=0, noise=1.0):
    """Learnable toy data (class-dependent means) for CPU tests."""
    g = torch.Generator().manual_seed(seed)
    means = torch.randn(num_classes, *shape, generator=g)
    y = torch.randint(0, num_classes, (n,), generator=g)
    x = means[y] + noise * torch.randn(n, *shape, generator=g)
    return TensorData(x, y, num_classes)


def tensors_from_loader(loader, num_classes, device="cpu"):
    """Convert any (x, y) DataLoader/iterable into TensorData (for user models)."""
    xs, ys = zip(*[(x, y) for x, y in loader])
    return TensorData(torch.cat(xs), torch.cat(ys).long(), num_classes).to(device)


def iterate_minibatches(data, batch_size, generator=None, shuffle=True):
    n = len(data)
    if shuffle:
        idx = torch.randperm(n, generator=generator).to(data.x.device)
    else:
        idx = torch.arange(n, device=data.x.device)
    for i in range(0, n, batch_size):
        j = idx[i:i + batch_size]
        yield data.x[j], data.y[j]
