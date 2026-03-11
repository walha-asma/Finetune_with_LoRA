"""
Dataset loader for small concept learning dataset.
Loads dataset.json and returns PyTorch DataLoader.
"""

import json
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class ImagePromptDataset(Dataset):
    def __init__(self, dataset_json="data/dataset.json", image_size=512):
        self.samples = []

        dataset_json = Path(dataset_json)
        assert dataset_json.exists(), f"{dataset_json} not found"

        with open(dataset_json, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Flatten categories
        for category, items in data.items():
            for item in items:
                self.samples.append(item)

        assert len(self.samples) > 0, "Dataset is empty"

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

        print(f"Loaded dataset with {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]

        image = Image.open(item["image"]).convert("RGB")
        image = self.transform(image)

        prompt = item["prompt"]

        return {
            "image": image,
            "prompt": prompt
        }


def get_dataloader(
    dataset_json="data/dataset.json",
    batch_size=1,
    shuffle=True,
    num_workers=2
):
    dataset = ImagePromptDataset(dataset_json)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
