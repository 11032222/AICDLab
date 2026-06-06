from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision import transforms
from torchvision.transforms import InterpolationMode

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def build_transforms(
    image_size: int,
    train: bool,
    use_randaugment: bool = False,
    use_random_erasing: bool = False,
    resize_size: int | None = None,
    interpolation: InterpolationMode = InterpolationMode.BILINEAR,
) -> transforms.Compose:
    if train:
        transform_list = [
            transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0), interpolation=interpolation),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.03),
        ]
        if use_randaugment:
            transform_list.append(transforms.RandAugment())
    else:
        resize_size = resize_size or int(round(image_size * 256 / 224))
        transform_list = [
            transforms.Resize(resize_size, interpolation=interpolation),
            transforms.CenterCrop(image_size),
        ]

    transform_list.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    if train and use_random_erasing:
        transform_list.append(transforms.RandomErasing(p=0.25, scale=(0.02, 0.12), ratio=(0.3, 3.3)))
    return transforms.Compose(transform_list)


class ImageCsvDataset(Dataset):
    def __init__(self, csv_path: str | Path, transform: transforms.Compose | None = None) -> None:
        self.csv_path = Path(csv_path)
        self.frame = pd.read_csv(self.csv_path)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.frame.iloc[index]
        image_path = Path(row["image_path"])
        with Image.open(image_path) as image:
            image = image.convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, int(row["label"])


def load_class_counts(csv_path: str | Path) -> torch.Tensor:
    frame = pd.read_csv(csv_path)
    counts = frame["label"].value_counts().sort_index()
    return torch.tensor(counts.tolist(), dtype=torch.float32)


def build_weighted_sampler(csv_path: str | Path) -> WeightedRandomSampler:
    frame = pd.read_csv(csv_path)
    class_counts = frame["label"].value_counts().sort_index()
    class_weights = 1.0 / class_counts
    sample_weights = frame["label"].map(class_weights).to_numpy(dtype="float64")
    weight_tensor = torch.tensor(sample_weights, dtype=torch.double)
    return WeightedRandomSampler(weights=weight_tensor, num_samples=len(weight_tensor), replacement=True)


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
