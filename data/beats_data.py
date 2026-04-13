from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader


def extract_label_from_filename(file_path: Path):
    stem = file_path.stem
    last_part = stem.split("_")[-1]
    if last_part not in {"0", "1"}:
        return None
    return int(last_part)


def build_dataframe(cfg: Dict) -> pd.DataFrame:
    use_precomputed_embeddings = bool(cfg.get("use_precomputed_embeddings", False))

    if use_precomputed_embeddings:
        metadata_csv = Path(cfg["precomputed_embeddings_metadata_csv"])
        if not metadata_csv.exists():
            raise FileNotFoundError(f"Missing precomputed embeddings metadata CSV: {metadata_csv}")

        df = pd.read_csv(metadata_csv)

        required_cols = {"sample_id", "label", "embedding_path"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Precomputed embeddings metadata missing columns: {missing}")

        df = df[df["label"].isin([0, 1])].copy().reset_index(drop=True)

        if len(df) == 0:
            raise ValueError("No labeled rows found in precomputed embeddings metadata.")

        return df

    raw_data_dir = Path(cfg["raw_data_dir"])
    if not raw_data_dir.exists():
        raise FileNotFoundError(f"Missing raw_data_dir: {raw_data_dir}")

    files = sorted(raw_data_dir.rglob("*.aif"))

    debug_num_files = int(cfg.get("debug_num_files", 0))
    if debug_num_files > 0:
        files = files[:debug_num_files]

    rows = []
    for f in tqdm(files, desc="Scanning BEATs audio files"):
        label = extract_label_from_filename(f)
        if label is None:
            continue

        rows.append(
            {
                "audio_path": str(f),
                "label": int(label),
                "file_name": f.name,
                "sample_id": f.stem,
            }
        )

    df = pd.DataFrame(rows)

    if len(df) == 0:
        raise ValueError("No labeled .aif files found for BEATs pipeline.")

    if df["sample_id"].duplicated().any():
        dupes = df[df["sample_id"].duplicated(keep=False)]["sample_id"].tolist()[:10]
        raise ValueError(f"Duplicate sample_id values found in audio scan. Examples: {dupes}")

    return df


class WhaleAudioDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        sample_rate: int,
        max_audio_seconds: float,
        normalize_peak: bool = False,
    ):
        self.df = dataframe.reset_index(drop=True)
        self.sample_rate = int(sample_rate)
        self.max_audio_len = int(float(max_audio_seconds) * self.sample_rate)
        self.normalize_peak = bool(normalize_peak)

    def __len__(self):
        return len(self.df)

    def _load_audio(self, path: str) -> np.ndarray:
        wav, sr = librosa.load(path, sr=self.sample_rate, mono=True)
        wav = wav.astype(np.float32)

        if self.normalize_peak:
            peak = np.max(np.abs(wav))
            if peak > 1e-8:
                wav = wav / peak

        if len(wav) > self.max_audio_len:
            wav = wav[:self.max_audio_len]
        elif len(wav) < self.max_audio_len:
            pad_len = self.max_audio_len - len(wav)
            wav = np.pad(wav, (0, pad_len), mode="constant")

        return wav

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        wav = self._load_audio(row["audio_path"])
        label = float(row["label"])

        return {
            "audio": torch.tensor(wav, dtype=torch.float32),
            "label": torch.tensor(label, dtype=torch.float32),
            "audio_path": row["audio_path"],
            "sample_id": row["sample_id"],
        }


class WhaleEmbeddingDataset(torch.utils.data.Dataset):
    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        emb = np.load(row["embedding_path"]).astype(np.float32)
        label = float(row["label"])

        return {
            "inputs": torch.tensor(emb, dtype=torch.float32),
            "label": torch.tensor(label, dtype=torch.float32),
            "embedding_path": row["embedding_path"],
            "sample_id": row["sample_id"],
        }


def build_dataloaders_from_splits(
    cfg: Dict,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
):
    batch_size = int(cfg["batch_size"])
    num_workers = int(cfg.get("num_workers", 0))
    pin_memory = bool(cfg.get("pin_memory", True))
    drop_last_train = bool(cfg.get("drop_last_train", False))

    use_precomputed_embeddings = bool(cfg.get("use_precomputed_embeddings", False))

    if use_precomputed_embeddings:
        train_ds = WhaleEmbeddingDataset(train_df)
        val_ds = WhaleEmbeddingDataset(val_df)
        test_ds = WhaleEmbeddingDataset(test_df)
    else:
        sample_rate = int(cfg["sample_rate"])
        max_audio_seconds = float(cfg["max_audio_seconds"])
        normalize_peak = bool(cfg.get("normalize_peak", False))

        train_ds = WhaleAudioDataset(train_df, sample_rate, max_audio_seconds, normalize_peak)
        val_ds = WhaleAudioDataset(val_df, sample_rate, max_audio_seconds, normalize_peak)
        test_ds = WhaleAudioDataset(test_df, sample_rate, max_audio_seconds, normalize_peak)

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
