from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms



def build_dataframe(cfg: Dict) -> pd.DataFrame:
    metadata_csv = Path(cfg["metadata_csv"])
    if not metadata_csv.exists():
        raise FileNotFoundError(f"Missing metadata_csv: {metadata_csv}")

    df = pd.read_csv(metadata_csv)

    required_cols = {"sample_id", "feature_path", "label"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"metadata.csv missing required columns: {missing}")

    df = df[df["label"].isin([0, 1])].copy()
    df = df.reset_index(drop=True)

    if len(df) == 0:
        raise ValueError("No labeled samples found in metadata.csv for ResNet pipeline.")

    return df


class MelNPYDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        feature_path = row["feature_path"]
        label = int(row["label"])

        mel = np.load(feature_path).astype(np.float32)  # expected [H, W]

        if mel.ndim != 2:
            raise ValueError(
                f"Expected mel feature to be 2D [H, W], got shape {mel.shape} from {feature_path}"
            )

        # your preprocessing already normalizes to [0, 1]
        mel = np.clip(mel, 0.0, 1.0)

        mel_img = (mel * 255.0).round().astype(np.uint8)

        # duplicate grayscale to RGB for ImageNet-pretrained ResNet
        mel_rgb = np.stack([mel_img, mel_img, mel_img], axis=-1)
        image = Image.fromarray(mel_rgb)

        if self.transform is not None:
            image = self.transform(image)

        return {
            "image": image,
            "label": torch.tensor(label, dtype=torch.float32),
            "feature_path": feature_path,
            "sample_id": row.get("sample_id", ""),
        }


def build_transforms(cfg: Dict):
    image_size = int(cfg.get("image_size", 224))

    train_tfms = [
        transforms.Resize((image_size, image_size)),
    ]

    if bool(cfg.get("train_horizontal_flip", False)):
        train_tfms.append(transforms.RandomHorizontalFlip(p=0.5))

    if bool(cfg.get("train_vertical_flip", False)):
        train_tfms.append(transforms.RandomVerticalFlip(p=0.5))

    train_tfms.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    if bool(cfg.get("train_random_erasing", False)):
        train_tfms.append(transforms.RandomErasing(p=0.25))

    val_tfms = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    train_tfms = transforms.Compose(train_tfms)

    return train_tfms, val_tfms


def build_dataloaders_from_splits(
    cfg: Dict,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    batch_size = int(cfg["batch_size"])
    num_workers = int(cfg.get("num_workers", 4))
    pin_memory = bool(cfg.get("pin_memory", True))
    drop_last_train = bool(cfg.get("drop_last_train", False))

    train_tfms, eval_tfms = build_transforms(cfg)

    train_ds = MelNPYDataset(train_df, transform=train_tfms)
    val_ds = MelNPYDataset(val_df, transform=eval_tfms)
    test_ds = MelNPYDataset(test_df, transform=eval_tfms)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last_train,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
