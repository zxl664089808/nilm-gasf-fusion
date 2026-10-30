#!/usr/bin/env python
"""Train GASF-image and physical-feature fusion for 26-state recognition."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from nilm_pipeline.physics import FEATURE_NAMES
from train_from_manifest import ConservativeCNN


class HybridDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        features: np.ndarray,
        label_to_index: dict[str, int],
        image_size: int,
        channel_indices: tuple[int, ...],
    ):
        self.paths = frame["image_path"].tolist()
        self.labels = [label_to_index[label] for label in frame["label"]]
        self.features = features.astype(np.float32)
        self.image_size = image_size
        self.channel_indices = channel_indices

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        image = Image.open(self.paths[index]).convert("RGB")
        image = image.resize(
            (self.image_size, self.image_size),
            Image.Resampling.BILINEAR,
        )
        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = (tensor - mean) / std
        mask = torch.zeros(3, 1, 1)
        mask[list(self.channel_indices)] = 1.0
        return (
            tensor * mask,
            torch.from_numpy(self.features[index]),
            self.labels[index],
        )


class HybridCNN(nn.Module):
    def __init__(self, num_classes: int, feature_count: int):
        super().__init__()
        self.image_encoder = ConservativeCNN(num_classes).features
        self.physical_encoder = nn.Sequential(
            nn.Linear(feature_count, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(32, 32),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(160, 96),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(96, num_classes),
        )

    def forward(self, images: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        image_features = self.image_encoder(images).flatten(1)
        physical_features = self.physical_encoder(features)
        return self.classifier(
            torch.cat([image_features, physical_features], dim=1)
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--physical-features", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split-column", default="split")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--channels", default="vip")
    parser.add_argument("--max-per-label", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def limit_per_label(frame: pd.DataFrame, maximum: int | None, seed: int):
    if maximum is None:
        return frame
    return pd.concat(
        [
            part.sample(
                n=min(maximum, len(part)),
                random_state=seed,
            )
            for _, part in frame.groupby("label", sort=True)
        ],
        ignore_index=True,
    )


def evaluate(model, loader, device):
    model.eval()
    predictions = []
    targets = []
    with torch.no_grad():
        for images, features, labels in loader:
            logits = model(images.to(device), features.to(device))
            predictions.extend(logits.argmax(dim=1).cpu().numpy().tolist())
            targets.extend(labels.numpy().tolist())
    return np.asarray(predictions), np.asarray(targets)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    frame = pd.read_csv(args.manifest)
    physical = pd.read_csv(args.physical_features)
    physical = physical[["image_path", *FEATURE_NAMES]]
    if physical["image_path"].duplicated().any():
        raise ValueError("Duplicate image paths in physical feature table")
    frame = frame.merge(
        physical,
        on="image_path",
        how="left",
        validate="one_to_one",
    )
    if frame[FEATURE_NAMES].isna().any().any():
        raise ValueError("Missing physical features for some images")

    class_names = sorted(frame["label"].unique())
    label_to_index = {label: index for index, label in enumerate(class_names)}
    parts = {
        split: limit_per_label(
            frame[frame[args.split_column] == split].reset_index(drop=True),
            args.max_per_label,
            args.seed,
        )
        for split in ("train", "val", "test")
    }
    scaler = StandardScaler().fit(parts["train"][FEATURE_NAMES])
    feature_arrays = {
        split: scaler.transform(part[FEATURE_NAMES]).astype(np.float32)
        for split, part in parts.items()
    }

    channel_lookup = {"v": 0, "i": 1, "p": 2}
    selected_channels = tuple(
        channel_lookup[channel]
        for channel in args.channels.lower()
        if channel in channel_lookup
    )
    if not selected_channels or len(set(args.channels.lower())) != len(args.channels):
        raise ValueError("channels must be a non-empty unique subset of v, i, p")
    loaders = {
        split: DataLoader(
            HybridDataset(
                part,
                feature_arrays[split],
                label_to_index,
                args.image_size,
                selected_channels,
            ),
            batch_size=args.batch_size,
            shuffle=(split == "train"),
            num_workers=args.num_workers,
        )
        for split, part in parts.items()
    }
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
    model = HybridCNN(len(class_names), len(FEATURE_NAMES)).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.7, patience=3
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_path = args.output_dir / "best_model.pth"
    best_val = -1.0
    stale = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total = 0
        for images, features, labels in loaders["train"]:
            images = images.to(device)
            features = features.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images, features), labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(labels)
            total += len(labels)
        val_pred, val_true = evaluate(model, loaders["val"], device)
        val_accuracy = float(accuracy_score(val_true, val_pred))
        val_f1 = float(f1_score(val_true, val_pred, average="macro"))
        scheduler.step(val_accuracy)
        history.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / max(total, 1),
                "val_accuracy": val_accuracy,
                "val_macro_f1": val_f1,
            }
        )
        print(
            f"epoch={epoch:03d} loss={total_loss/max(total,1):.4f} "
            f"val_acc={val_accuracy:.4f} val_f1={val_f1:.4f}",
            flush=True,
        )
        if val_accuracy > best_val:
            best_val = val_accuracy
            stale = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": class_names,
                    "feature_names": FEATURE_NAMES,
                    "feature_mean": scaler.mean_.tolist(),
                    "feature_scale": scaler.scale_.tolist(),
                },
                best_path,
            )
        else:
            stale += 1
            if stale >= args.patience:
                print("Early stopping")
                break
    checkpoint = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_pred, test_true = evaluate(model, loaders["test"], device)
    metrics = {
        "best_val_accuracy": best_val,
        "test_accuracy": float(accuracy_score(test_true, test_pred)),
        "test_macro_f1": float(f1_score(test_true, test_pred, average="macro")),
        "class_names": class_names,
        "physical_features": FEATURE_NAMES,
        "split_sizes": {split: len(part) for split, part in parts.items()},
        "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "model": "hybrid_cnn_physics",
        "channels": args.channels,
        "image_size": args.image_size,
        "seed": args.seed,
        "history": history,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame(
        confusion_matrix(test_true, test_pred),
        index=class_names,
        columns=class_names,
    ).to_csv(args.output_dir / "confusion_matrix.csv")
    (args.output_dir / "classification_report.txt").write_text(
        classification_report(
            test_true,
            test_pred,
            target_names=class_names,
            zero_division=0,
        ),
        encoding="utf-8",
    )
    print(json.dumps({k: v for k, v in metrics.items() if k != "history"}, indent=2))


if __name__ == "__main__":
    main()
