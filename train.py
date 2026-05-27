import time
import json
import random
import argparse
from pathlib import Path
from typing import Any, Dict, Tuple, List, Optional
from data import beats_data, resnet18_data
from models import beats_model, resnet18_model
import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import wandb

from dataclasses import dataclass



@dataclass
class CheckpointState:
    model: nn.Module
    optimizer: Optional[torch.optim.Optimizer]
    scheduler: Any
    start_epoch: int
    best_score: float
    best_threshold: float

# ============================================================
# CONFIG / UTILS
# ============================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="General frozen-backbone binary trainer")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=["beats", "resnet18"],
        help="Which model family to use",
    )
    return parser.parse_args()


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def ensure_dirs(cfg: Dict[str, Any]) -> None:
    ensure_dir(cfg["output_dir"])
    ensure_dir(cfg["checkpoint_dir"])
    if "split_dir" in cfg:
        ensure_dir(cfg["split_dir"])
    if "shared_split_dir" in cfg:
        ensure_dir(cfg["shared_split_dir"])


def save_json(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    best_score: float,
    threshold: float,
    cfg: Dict[str, Any],
) -> None:
    ckpt = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "best_score": best_score,
        "best_threshold": threshold,
        "config": cfg,
    }
    torch.save(ckpt, path)


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer = None,
    scheduler: Any = None,
    map_location: str = "cpu",
) -> CheckpointState:
    ckpt = torch.load(path, map_location=map_location)

    model.load_state_dict(ckpt["model_state_dict"])

    if optimizer is not None and ckpt.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    return CheckpointState(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        start_epoch=ckpt.get("epoch", -1) + 1,
        best_score=ckpt.get("best_score", -1.0),
        best_threshold=ckpt.get("best_threshold", 0.5),
    )


def get_device(cfg: Dict[str, Any]) -> str:
    if cfg.get("device", "auto") == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return cfg["device"]



DATA_BUILDERS = {
    "beats": (
        beats_data.build_dataframe,
        beats_data.build_dataloaders_from_splits,
    ),
    "resnet18": (
        resnet18_data.build_dataframe,
        resnet18_data.build_dataloaders_from_splits,
    ),
}


MODEL_BUILDERS = {
    "beats": beats_model.build_model,
    "resnet18": resnet18_model.build_model,
}


def get_builders(model_type: str):
    if model_type not in DATA_BUILDERS or model_type not in MODEL_BUILDERS:
        raise ValueError(
            f"Unknown model_type: {model_type}. "
            f"Available options: {list(MODEL_BUILDERS.keys())}"
        )

    build_dataframe, build_dataloaders_from_splits = DATA_BUILDERS[model_type]
    build_model = MODEL_BUILDERS[model_type]

    return build_dataframe, build_dataloaders_from_splits, build_model


# ============================================================
# SPLITS
# ============================================================




def build_or_load_shared_splits(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Build one canonical split by sample_id and label.
    This split is shared by all model types.
    """

    shared_split_dir = Path(cfg["shared_split_dir"])
    shared_split_dir.mkdir(parents=True, exist_ok=True)

    train_ids_csv = shared_split_dir / "train_ids.csv"
    val_ids_csv = shared_split_dir / "val_ids.csv"
    test_ids_csv = shared_split_dir / "test_ids.csv"

    if train_ids_csv.exists() and val_ids_csv.exists() and test_ids_csv.exists():
        train_ids = pd.read_csv(train_ids_csv)
        val_ids = pd.read_csv(val_ids_csv)
        test_ids = pd.read_csv(test_ids_csv)
        return train_ids, val_ids, test_ids

    canonical_source = cfg.get("canonical_split_source", "resnet_metadata")

    if canonical_source != "resnet_metadata":
        raise ValueError(f"Unsupported canonical_split_source: {canonical_source}")

    metadata_csv = Path(cfg["metadata_csv"])
    if not metadata_csv.exists():
        raise FileNotFoundError(f"Missing metadata_csv for canonical split creation: {metadata_csv}")

    df = pd.read_csv(metadata_csv)

    required_cols = {"sample_id", "label"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"metadata.csv missing required columns: {missing}")

    df = df[df["label"].isin([0, 1])].copy()
    df = df[["sample_id", "label"]].drop_duplicates().reset_index(drop=True)

    if len(df) == 0:
        raise ValueError("No labeled rows found in canonical split source.")

    test_size = float(cfg["test_size"])
    val_size = float(cfg["val_size"])
    seed = int(cfg["seed"])

    train_val_ids, test_ids = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=df["label"],
    )

    val_relative = val_size / (1.0 - test_size)

    train_ids, val_ids = train_test_split(
        train_val_ids,
        test_size=val_relative,
        random_state=seed,
        stratify=train_val_ids["label"],
    )

    train_ids.to_csv(train_ids_csv, index=False)
    val_ids.to_csv(val_ids_csv, index=False)
    test_ids.to_csv(test_ids_csv, index=False)

    return train_ids, val_ids, test_ids


def apply_shared_split(full_df: pd.DataFrame, split_ids_df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    """
    Filter a model-specific dataframe by the shared sample_id split.
    """
    if "sample_id" not in full_df.columns:
        raise ValueError(f"{split_name}: full dataframe must contain 'sample_id'")
    if "sample_id" not in split_ids_df.columns:
        raise ValueError(f"{split_name}: split ids dataframe must contain 'sample_id'")

    merged = full_df.merge(
        split_ids_df[["sample_id"]],
        on="sample_id",
        how="inner",
    ).copy()

    if len(merged) != len(split_ids_df):
        missing_count = len(split_ids_df) - len(merged)
        raise ValueError(
            f"Split '{split_name}' mismatch: expected {len(split_ids_df)} ids, "
            f"matched {len(merged)} rows. Missing {missing_count} samples in this pipeline."
        )

    return merged.reset_index(drop=True)

# ============================================================
# LOSS / METRICS
# ============================================================

def build_loss(train_df: pd.DataFrame, device: str, cfg: Dict[str, Any]) -> nn.Module:
    use_weighted_loss = bool(cfg.get("use_weighted_loss", True))
    if not use_weighted_loss:
        return nn.BCEWithLogitsLoss()

    class_counts = train_df["label"].value_counts().to_dict()
    n_neg = float(class_counts.get(0, 1))
    n_pos = float(class_counts.get(1, 1))

    pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)], dtype=torch.float32, device=device)
    print(f"Using weighted BCEWithLogitsLoss with pos_weight={pos_weight.item():.6f}")
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)


def logits_to_probs(logits: torch.Tensor) -> torch.Tensor:
    # one-logit binary classification
    return torch.sigmoid(logits.view(-1))


def find_best_f1_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    min_thr: float = 0.05,
    max_thr: float = 0.95,
    num_steps: int = 181,
) -> Dict[str, float]:
    thresholds = np.linspace(min_thr, max_thr, num_steps)

    best = {
        "threshold": 0.5,
        "f1": -1.0,
        "precision": 0.0,
        "recall": 0.0,
        "accuracy": 0.0,
    }

    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(np.int64)

        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best["f1"]:
            best["threshold"] = float(thr)
            best["f1"] = float(f1)
            best["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
            best["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
            best["accuracy"] = float(accuracy_score(y_true, y_pred))

    return best


def recall_at_fixed_fpr(y_true: np.ndarray, y_prob: np.ndarray, target_fpr: float) -> float:
    try:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        valid = np.where(fpr <= target_fpr)[0]
        if len(valid) == 0:
            return 0.0
        return float(np.max(tpr[valid]))
    except Exception:
        return float("nan")


def compute_binary_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(np.int64)

    metrics = {}

    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    metrics["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    metrics["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    metrics["f1"] = float(f1_score(y_true, y_pred, zero_division=0))

    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    except Exception:
        metrics["roc_auc"] = float("nan")

    try:
        metrics["pr_auc"] = float(average_precision_score(y_true, y_prob))
    except Exception:
        metrics["pr_auc"] = float("nan")

    metrics["recall_at_fpr_1pct"] = recall_at_fixed_fpr(y_true, y_prob, 0.01)
    metrics["recall_at_fpr_5pct"] = recall_at_fixed_fpr(y_true, y_prob, 0.05)

    best_f1 = find_best_f1_threshold(
        y_true=y_true,
        y_prob=y_prob,
        min_thr=0.05,
        max_thr=0.95,
        num_steps=181,
    )

    metrics["best_f1"] = best_f1["f1"]
    metrics["best_f1_threshold"] = best_f1["threshold"]
    metrics["best_f1_precision"] = best_f1["precision"]
    metrics["best_f1_recall"] = best_f1["recall"]
    metrics["best_f1_accuracy"] = best_f1["accuracy"]

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    metrics["tn"] = int(cm[0, 0])
    metrics["fp"] = int(cm[0, 1])
    metrics["fn"] = int(cm[1, 0])
    metrics["tp"] = int(cm[1, 1])

    return metrics


# ============================================================
# LOOP HELPERS
# ============================================================

def batch_to_device(batch: Any, device: str):
    """
    Supports either:
      - dict batches, where tensors are under keys
      - tuple batches like (x, y)
    """
    if isinstance(batch, dict):
        out = {}
        for k, v in batch.items():
            if torch.is_tensor(v):
                out[k] = v.to(device, non_blocking=True)
            else:
                out[k] = v
        return out

    if isinstance(batch, (tuple, list)):
        out = []
        for v in batch:
            if torch.is_tensor(v):
                out.append(v.to(device, non_blocking=True))
            else:
                out.append(v)
        return out

    raise TypeError(f"Unsupported batch type: {type(batch)}")


def unpack_batch_for_model(batch: Any) -> Tuple[Any, torch.Tensor]:
    """
    Standardize batch output.

    Allowed:
      dict with one of:
        {"audio": ..., "label": ...}
        {"image": ..., "label": ...}
        {"inputs": ..., "label": ...}
      tuple/list:
        (inputs, labels)
    """
    if isinstance(batch, dict):
        if "audio" in batch:
            x = batch["audio"]
        elif "image" in batch:
            x = batch["image"]
        elif "inputs" in batch:
            x = batch["inputs"]
        else:
            raise KeyError("Batch dict must contain 'audio', 'image', or 'inputs'.")

        y = batch["label"].float().view(-1)
        return x, y

    if isinstance(batch, (tuple, list)) and len(batch) >= 2:
        x = batch[0]
        y = batch[1].float().view(-1)
        return x, y

    raise TypeError("Unsupported batch structure.")


def get_amp_dtype(cfg: Dict[str, Any]):
    amp_dtype = str(cfg.get("amp_dtype", "float16")).lower()
    if amp_dtype == "bfloat16":
        return torch.bfloat16
    return torch.float16


def forward_model(model: nn.Module, inputs: Any) -> torch.Tensor:
    logits = model(inputs)

    # expected shape [B] or [B,1]
    if logits.ndim == 2 and logits.shape[1] == 1:
        logits = logits[:, 0]
    elif logits.ndim != 1:
        raise ValueError(f"Expected model to return [B] or [B,1], got {tuple(logits.shape)}")

    return logits


# ============================================================
# TRAIN / EVAL
# ============================================================

def run_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    use_amp: bool,
    scaler: torch.amp.GradScaler,
    is_train: bool,
    epoch: int,
    total_epochs: int,
    fixed_threshold: float,
) -> Dict[str, float]:
    if is_train:
        model.train()
        desc = f"Train {epoch + 1}/{total_epochs}"
    else:
        model.eval()
        desc = f"Eval  {epoch + 1}/{total_epochs}"

    running_loss = 0.0
    n_samples = 0

    all_targets: List[float] = []
    all_probs: List[float] = []

    batch_times = []
    total_forward_samples = 0

    amp_enabled = use_amp and device.startswith("cuda")
    amp_dtype = torch.float16

    pbar = tqdm(loader, desc=desc, leave=False)

    context = torch.enable_grad if is_train else torch.no_grad

    with context():
        for batch in pbar:
            batch = batch_to_device(batch, device)
            inputs, targets = unpack_batch_for_model(batch)

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            start_time = time.perf_counter()

            with torch.amp.autocast(
                device_type="cuda",
                enabled=amp_enabled,
                dtype=amp_dtype,
            ):
                logits = forward_model(model, inputs)
                loss = criterion(logits, targets)

            if is_train:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            probs = logits_to_probs(logits)

            elapsed = time.perf_counter() - start_time
            batch_times.append(elapsed)
            total_forward_samples += int(targets.shape[0])

            bs = int(targets.shape[0])
            running_loss += float(loss.item()) * bs
            n_samples += bs

            all_targets.extend(targets.detach().float().cpu().numpy().tolist())
            all_probs.extend(probs.detach().float().cpu().numpy().tolist())

            pbar.set_postfix(loss=f"{loss.item():.4f}")

    y_true = np.array(all_targets, dtype=np.int64)
    y_prob = np.array(all_probs, dtype=np.float32)

    metrics = compute_binary_metrics(
        y_true=y_true,
        y_prob=y_prob,
        threshold=fixed_threshold,
    )
    metrics["loss"] = running_loss / max(n_samples, 1)

    if len(batch_times) > 0:
        total_time = max(sum(batch_times), 1e-8)
        metrics["samples_per_sec"] = float(total_forward_samples / total_time)
        metrics["sec_per_sample"] = float(total_time / max(total_forward_samples, 1))
    else:
        metrics["samples_per_sec"] = float("nan")
        metrics["sec_per_sample"] = float("nan")

    return metrics


@torch.no_grad()
def evaluate_on_test(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: str,
    use_amp: bool,
    threshold: float,
) -> Dict[str, float]:
    model.eval()

    running_loss = 0.0
    n_samples = 0

    all_targets: List[float] = []
    all_probs: List[float] = []

    batch_times = []
    total_forward_samples = 0

    amp_enabled = use_amp and device.startswith("cuda")
    amp_dtype = torch.float16

    pbar = tqdm(loader, desc="Test", leave=False)

    for batch in pbar:
        batch = batch_to_device(batch, device)
        inputs, targets = unpack_batch_for_model(batch)

        start_time = time.perf_counter()

        with torch.amp.autocast(
            device_type="cuda",
            enabled=amp_enabled,
            dtype=amp_dtype,
        ):
            logits = forward_model(model, inputs)
            loss = criterion(logits, targets)

        probs = logits_to_probs(logits)

        elapsed = time.perf_counter() - start_time
        batch_times.append(elapsed)
        total_forward_samples += int(targets.shape[0])

        bs = int(targets.shape[0])
        running_loss += float(loss.item()) * bs
        n_samples += bs

        all_targets.extend(targets.detach().float().cpu().numpy().tolist())
        all_probs.extend(probs.detach().float().cpu().numpy().tolist())

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    y_true = np.array(all_targets, dtype=np.int64)
    y_prob = np.array(all_probs, dtype=np.float32)

    metrics = compute_binary_metrics(
        y_true=y_true,
        y_prob=y_prob,
        threshold=threshold,
    )
    metrics["loss"] = running_loss / max(n_samples, 1)

    if len(batch_times) > 0:
        total_time = max(sum(batch_times), 1e-8)
        metrics["samples_per_sec"] = float(total_forward_samples / total_time)
        metrics["sec_per_sample"] = float(total_time / max(total_forward_samples, 1))
    else:
        metrics["samples_per_sec"] = float("nan")
        metrics["sec_per_sample"] = float("nan")

    return metrics


# ============================================================
# WANDB
# ============================================================

def maybe_init_wandb(cfg: Dict[str, Any]) -> None:
    if not cfg.get("use_wandb", True):
        return

    wandb.init(
        project=cfg["wandb_project"],
        name=cfg["wandb_run_name"],
        config=cfg,
    )


def maybe_log_wandb(log_dict: Dict[str, Any], use_wandb: bool) -> None:
    if use_wandb:
        wandb.log(log_dict)


# ============================================================
# MAIN
# ============================================================

def main():
    args = parse_args()
    cfg = load_yaml(args.config)
    cfg["model_type"] = args.model_type

    set_seed(int(cfg["seed"]))
    ensure_dirs(cfg)

    device = get_device(cfg)
    cfg["resolved_device"] = device

    print(f"Using device: {device}")
    print(f"Model type: {args.model_type}")
    print(f"Config: {args.config}")

    build_dataframe, build_dataloaders_from_splits, build_model = get_builders(args.model_type)

    shared_train_ids, shared_val_ids, shared_test_ids = build_or_load_shared_splits(cfg)
    
    full_df = build_dataframe(cfg)
    
    train_df = apply_shared_split(full_df, shared_train_ids, "train")
    val_df = apply_shared_split(full_df, shared_val_ids, "val")
    test_df = apply_shared_split(full_df, shared_test_ids, "test")
    
    train_loader, val_loader, test_loader = build_dataloaders_from_splits(
        cfg=cfg,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
    )
    
    model = build_model(cfg)
    model = model.to(device)

    # Safety check: backbone should already be frozen in model file.
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"\nTrainable params: {n_trainable:,} / {n_total:,}")

    criterion = build_loss(train_df, device, cfg)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=float(cfg["lr"]),
        weight_decay=float(cfg["weight_decay"]),
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(cfg["epochs"]),
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(bool(cfg.get("use_amp", True)) and device.startswith("cuda")),
    )

    start_epoch = 0
    best_score = -1.0
    best_threshold = float(cfg.get("default_eval_threshold", 0.5))

    maybe_init_wandb(cfg)

    resume_ckpt = str(cfg.get("resume_checkpoint", "")).strip()
    if resume_ckpt and Path(resume_ckpt).exists():
        print(f"Resuming from checkpoint: {resume_ckpt}")
    
        state = load_checkpoint(
            path=resume_ckpt,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            map_location=device,
        )
    
        model = state.model
        optimizer = state.optimizer
        scheduler = state.scheduler
        start_epoch = state.start_epoch
        best_score = state.best_score
        best_threshold = state.best_threshold
    
        print(
            f"Resumed at epoch={start_epoch}, "
            f"best_score={best_score:.6f}, "
            f"best_threshold={best_threshold:.4f}"
        )

    best_metric_name = str(cfg.get("best_metric_name", "val_best_f1"))
    use_amp = bool(cfg.get("use_amp", True))
    

    print(f"  Train matched: {len(train_df)}")
    print(f"  Val matched  : {len(val_df)}")
    print(f"  Test matched : {len(test_df)}")
    print("\nClass counts after applying shared split:")
    print("Train:\n", train_df["label"].value_counts().sort_index())
    print("Val:\n", val_df["label"].value_counts().sort_index())
    print("Test:\n", test_df["label"].value_counts().sort_index())

    print("\nStarting training...\n")

    for epoch in range(start_epoch, int(cfg["epochs"])):
        current_eval_threshold = best_threshold if bool(cfg.get("use_best_threshold_during_training", False)) else 0.5

        train_metrics = run_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            use_amp=use_amp,
            scaler=scaler,
            is_train=True,
            epoch=epoch,
            total_epochs=int(cfg["epochs"]),
            fixed_threshold=current_eval_threshold,
        )

        val_metrics = run_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            optimizer=None,
            device=device,
            use_amp=use_amp,
            scaler=scaler,
            is_train=False,
            epoch=epoch,
            total_epochs=int(cfg["epochs"]),
            fixed_threshold=current_eval_threshold,
        )

        scheduler.step()
        lr_now = optimizer.param_groups[0]["lr"]

        # Use the best threshold discovered on validation this epoch.
        val_best_threshold = float(val_metrics["best_f1_threshold"])



        log_dict = {
            "epoch": epoch,
            "lr": lr_now,

            "train/loss": train_metrics["loss"],
            "train/accuracy": train_metrics["accuracy"],
            "train/precision": train_metrics["precision"],
            "train/recall": train_metrics["recall"],
            "train/f1": train_metrics["f1"],
            "train/roc_auc": train_metrics["roc_auc"],
            "train/pr_auc": train_metrics["pr_auc"],
            "train/recall_at_fpr_1pct": train_metrics["recall_at_fpr_1pct"],
            "train/recall_at_fpr_5pct": train_metrics["recall_at_fpr_5pct"],
            "train/samples_per_sec": train_metrics["samples_per_sec"],
            "train/sec_per_sample": train_metrics["sec_per_sample"],

            "val/loss": val_metrics["loss"],
            "val/accuracy@0.5": val_metrics["accuracy"],
            "val/precision@0.5": val_metrics["precision"],
            "val/recall@0.5": val_metrics["recall"],
            "val/f1@0.5": val_metrics["f1"],
            "val/roc_auc": val_metrics["roc_auc"],
            "val/pr_auc": val_metrics["pr_auc"],
            "val/recall_at_fpr_1pct": val_metrics["recall_at_fpr_1pct"],
            "val/recall_at_fpr_5pct": val_metrics["recall_at_fpr_5pct"],
            "val/best_f1": val_metrics["best_f1"],
            "val/best_f1_threshold": val_metrics["best_f1_threshold"],
            "val/best_f1_precision": val_metrics["best_f1_precision"],
            "val/best_f1_recall": val_metrics["best_f1_recall"],
            "val/best_f1_accuracy": val_metrics["best_f1_accuracy"],
            "val/tn@0.5": val_metrics["tn"],
            "val/fp@0.5": val_metrics["fp"],
            "val/fn@0.5": val_metrics["fn"],
            "val/tp@0.5": val_metrics["tp"],
            "val/samples_per_sec": val_metrics["samples_per_sec"],
            "val/sec_per_sample": val_metrics["sec_per_sample"],
        }

        maybe_log_wandb(log_dict, bool(cfg.get("use_wandb", True)))

        print(
            f"Epoch [{epoch+1}/{cfg['epochs']}] "
            f"train_loss={train_metrics['loss']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_auc={val_metrics['roc_auc']:.4f} "
            f"val_pr_auc={val_metrics['pr_auc']:.4f} "
            f"val_r@1%fpr={val_metrics['recall_at_fpr_1pct']:.4f} "
            f"val_best_f1={val_metrics['best_f1']:.4f} "
            f"thr={val_best_threshold:.3f}"
        )

        # Save last
        save_checkpoint(
            path=str(Path(cfg["checkpoint_dir"]) / "last.pt"),
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            best_score=best_score,
            threshold=best_threshold,
            cfg=cfg,
        )

        current_score = None
        if best_metric_name == "val_best_f1":
            current_score = val_metrics["best_f1"]
        elif best_metric_name == "val_pr_auc":
            current_score = val_metrics["pr_auc"]
        elif best_metric_name == "val_roc_auc":
            current_score = val_metrics["roc_auc"]
        elif best_metric_name == "val_recall_at_fpr_1pct":
            current_score = val_metrics["recall_at_fpr_1pct"]
        elif best_metric_name == "val_recall_at_fpr_5pct":
            current_score = val_metrics["recall_at_fpr_5pct"]
        else:
            raise ValueError(f"Unsupported best_metric_name: {best_metric_name}")

        if current_score > best_score:
            best_score = float(current_score)
            best_threshold = float(val_best_threshold)

            save_checkpoint(
                path=str(Path(cfg["checkpoint_dir"]) / "best.pt"),
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_score=best_score,
                threshold=best_threshold,
                cfg=cfg,
            )

            print(f"New best model saved. best_score={best_score:.6f}, best_threshold={best_threshold:.4f}")

    # Final test using best checkpoint and best validation threshold
    best_ckpt_path = str(Path(cfg["checkpoint_dir"]) / "best.pt")
    print(f"\nLoading best checkpoint for final test: {best_ckpt_path}")

    state = load_checkpoint(
        path=best_ckpt_path,
        model=model,
        optimizer=None,
        scheduler=None,
        map_location=device,
    )
    
    model = state.model
    best_score_loaded = state.best_score
    best_threshold_loaded = state.best_threshold

    test_metrics = evaluate_on_test(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        use_amp=use_amp,
        threshold=best_threshold_loaded,
    )

    final_results = {
        "best_val_score": best_score_loaded,
        "best_val_threshold": best_threshold_loaded,
        "test/loss": test_metrics["loss"],
        "test/accuracy": test_metrics["accuracy"],
        "test/precision": test_metrics["precision"],
        "test/recall": test_metrics["recall"],
        "test/f1": test_metrics["f1"],
        "test/roc_auc": test_metrics["roc_auc"],
        "test/pr_auc": test_metrics["pr_auc"],
        "test/recall_at_fpr_1pct": test_metrics["recall_at_fpr_1pct"],
        "test/recall_at_fpr_5pct": test_metrics["recall_at_fpr_5pct"],
        "test/best_f1": test_metrics["best_f1"],
        "test/best_f1_threshold": test_metrics["best_f1_threshold"],
        "test/tn": test_metrics["tn"],
        "test/fp": test_metrics["fp"],
        "test/fn": test_metrics["fn"],
        "test/tp": test_metrics["tp"],
        "test/samples_per_sec": test_metrics["samples_per_sec"],
        "test/sec_per_sample": test_metrics["sec_per_sample"],
    }

    print("\nFinal test results:")
    for k, v in final_results.items():
        if isinstance(v, float):
            print(f"{k}: {v:.6f}")
        else:
            print(f"{k}: {v}")

    save_json(str(Path(cfg["output_dir"]) / "final_test_results.json"), final_results)

    if cfg.get("use_wandb", True):
        wandb.log(final_results)
        wandb.finish()


if __name__ == "__main__":
    main()
