from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

logging.getLogger("torch.utils.flop_counter").setLevel(logging.ERROR)

import lightning as L
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class PokemonDataModule(L.LightningDataModule):
    """
    Lightning DataModule for Pokémon JPG images.

    - Loads all .jpg files from `data/pokemon_jpg/pokemon_jpg/`
    - Resizes to 128x128
    - Converts to tensor and normalizes pixel values to [-1, 1]
    - Saves the fully preprocessed dataset tensor to disk
    """

    def __init__(
        self,
        data_dir: str = "data/pokemon_jpg/pokemon_jpg",
        output_dir: str = "data/processed",
        output_file: str = "pokemon_128_normalized.pt",
        batch_size: int = 32,
        num_workers: int = 0,
        resize_size: int = 128,
        pin_memory: bool = True,
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_file = output_file
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.resize_size = resize_size
        self.pin_memory = pin_memory

        # Mean/std 0.5 maps [0, 1] -> [-1, 1]
        self.transform = transforms.Compose(
            [
                transforms.Resize((resize_size, resize_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
        )

        self.dataset: Optional[datasets.ImageFolder] = None
        self.saved_path: Optional[Path] = None

    def prepare_data(self) -> None:
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def setup(self, stage: Optional[str] = None) -> None:
        if self.dataset is None:
            # ImageFolder requires class subdirectories; this works for this dataset layout.
            self.dataset = datasets.ImageFolder(
                root=str(self.data_dir.parent),
                transform=self.transform,
                is_valid_file=lambda p: p.lower().endswith(".jpg"),
            )

    def train_dataloader(self) -> DataLoader:
        if self.dataset is None:
            self.setup()

        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def save_preprocessed_dataset(self) -> Path:
        """
        Iterates over the dataset and saves preprocessed tensors + labels to disk.
        """
        if self.dataset is None:
            self.setup()

        assert self.dataset is not None  # for type checkers
        dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

        all_images = []
        all_labels = []

        for images, labels in dataloader:
            all_images.append(images)
            all_labels.append(labels)

        images_tensor = torch.cat(all_images, dim=0)
        labels_tensor = torch.cat(all_labels, dim=0)

        save_path = self.output_dir / self.output_file
        torch.save(
            {
                "images": images_tensor,  # shape: [N, 3, 128, 128], values in [-1, 1]
                "labels": labels_tensor,  # shape: [N]
                "class_to_idx": self.dataset.class_to_idx,
                "idx_to_class": {v: k for k, v in self.dataset.class_to_idx.items()},
            },
            save_path,
        )

        self.saved_path = save_path
        return save_path


def main() -> None:
    dm = PokemonDataModule(
        data_dir="data/pokemon_jpg/pokemon_jpg",
        output_dir="data/processed",
        output_file="pokemon_128_normalized.pt",
        batch_size=64,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    dm.prepare_data()
    dm.setup()

    save_path = dm.save_preprocessed_dataset()
    print(f"Saved preprocessed dataset to: {save_path}")


if __name__ == "__main__":
    main()
