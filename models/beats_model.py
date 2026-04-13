import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


class PrecomputedBEATsHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        metadata_csv = cfg.get("precomputed_embeddings_metadata_csv", "")
        if not metadata_csv:
            raise ValueError("Missing precomputed_embeddings_metadata_csv in config.")

        df = pd.read_csv(metadata_csv)
        if len(df) == 0:
            raise ValueError("Empty precomputed embeddings metadata CSV.")

        example_path = df.iloc[0]["embedding_path"]
        example_emb = np.load(example_path).astype(np.float32)
        embed_dim = int(example_emb.shape[0])

        self.dropout = nn.Dropout(float(cfg.get("dropout", 0.2)))
        self.classifier = nn.Linear(embed_dim, 1)

    def forward(self, inputs):
        x = self.dropout(inputs)
        logits = self.classifier(x).squeeze(1)
        return logits


class BEATsBinaryHead(nn.Module):
    def __init__(self, cfg, BEATs, BEATsConfig):
        super().__init__()

        checkpoint = torch.load(cfg["beats_checkpoint_path"], map_location="cpu")
        beats_cfg = BEATsConfig(checkpoint["cfg"])

        self.backbone = BEATs(beats_cfg)
        self.backbone.load_state_dict(checkpoint["model"])

        for p in self.backbone.parameters():
            p.requires_grad = False

        embed_dim = None
        for attr_name in ["encoder_embed_dim", "embed_dim"]:
            if hasattr(self.backbone.cfg, attr_name):
                embed_dim = getattr(self.backbone.cfg, attr_name)
                break

        if embed_dim is None:
            raise ValueError("Could not infer BEATs embedding dimension.")

        self.pooling = cfg.get("pooling", "mean")
        self.dropout = nn.Dropout(float(cfg.get("dropout", 0.2)))
        self.classifier = nn.Linear(embed_dim, 1)

    def forward(self, audio):
        feats = self.backbone.extract_features(audio)[0]  # [B, T, D]

        if self.pooling == "mean":
            pooled = feats.mean(dim=1)
        elif self.pooling == "cls_like_last":
            pooled = feats[:, -1, :]
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

        pooled = self.dropout(pooled)
        logits = self.classifier(pooled).squeeze(1)
        return logits


def import_beats_from_repo(beats_repo_dir: str):
    repo_dir = str(Path(beats_repo_dir).resolve())
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)

    from BEATs import BEATs, BEATsConfig
    return BEATs, BEATsConfig


def build_model(cfg):
    if bool(cfg.get("use_precomputed_embeddings", False)):
        return PrecomputedBEATsHead(cfg)

    BEATs, BEATsConfig = import_beats_from_repo(cfg["beats_repo_dir"])
    return BEATsBinaryHead(cfg, BEATs, BEATsConfig)
