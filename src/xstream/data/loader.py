"""ImageNet data loading with synthetic fallback."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _imagenet_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class SyntheticDataset(Dataset):
    """Random tensors mimicking ImageNet for when real data is unavailable."""

    def __init__(self, num_images: int = 100, image_size: int = 224, num_classes: int = 1000):
        self.num_images = num_images
        self.image_size = image_size
        self.num_classes = num_classes

    def __len__(self) -> int:
        return self.num_images

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        gen = torch.Generator().manual_seed(idx)
        image = torch.randn(3, self.image_size, self.image_size, generator=gen)
        label = idx % self.num_classes
        return image, label


def get_imagenet_val_loader(
    root: str = "/data/imagenet/val",
    num_images: int = 100,
    image_size: int = 224,
    seed: int = 42,
) -> DataLoader:
    """
    Load a deterministic subset of ImageNet validation set.
    Falls back to synthetic data if root doesn't exist.
    """
    try:
        dataset = datasets.ImageFolder(root, transform=_imagenet_transform(image_size))
        gen = torch.Generator().manual_seed(seed)
        indices = torch.randperm(len(dataset), generator=gen)[:num_images].tolist()
        subset = Subset(dataset, indices)
        pin_memory = torch.cuda.is_available()
        return DataLoader(subset, batch_size=1, shuffle=False, num_workers=4, pin_memory=pin_memory)
    except (FileNotFoundError, RuntimeError):
        print(f"[WARN] ImageNet not found at {root}, using synthetic data")
        return get_synthetic_loader(num_images, image_size)


def get_synthetic_loader(
    num_images: int = 100, image_size: int = 224
) -> DataLoader:
    """Generate synthetic random data for testing."""
    dataset = SyntheticDataset(num_images, image_size)
    return DataLoader(dataset, batch_size=1, shuffle=False)
