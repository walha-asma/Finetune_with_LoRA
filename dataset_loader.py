import json
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

BASE_DIR = Path("data")


class ImagePromptDataset(Dataset):
    def __init__(self, dataset_json, image_size=512):
        dataset_json = Path(dataset_json)
        assert dataset_json.exists(), f"{dataset_json} not found"

        with open(dataset_json, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Handle {"split": ..., "count": ..., "data": [...]} wrapper format
        if isinstance(data, list):
            self.samples = data
        elif isinstance(data, dict) and "data" in data:
            self.samples = data["data"]
        else:
            self.samples = list(data.values())

        assert len(self.samples) > 0, "Dataset is empty"

        self.base_dir = BASE_DIR
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        print(f"Loaded {len(self.samples)} samples from {dataset_json}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        image_path = self.base_dir / item["filepath"]
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        return {
            "image": image,
            "prompt": item["prompt"],
            "text": item.get("text", "")
        }


def get_dataloader(dataset_json, batch_size=1, shuffle=True, num_workers=2):
    dataset = ImagePromptDataset(dataset_json)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True)


def get_train_dataloader(batch_size=1):
    return get_dataloader(BASE_DIR / "train.json", batch_size=batch_size, shuffle=True)


def get_val_dataloader(batch_size=1):
    return get_dataloader(BASE_DIR / "val.json", batch_size=batch_size, shuffle=False)


def get_test_dataloader(batch_size=1):
    return get_dataloader(BASE_DIR / "test.json", batch_size=batch_size, shuffle=False)