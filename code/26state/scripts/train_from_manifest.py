#!/usr/bin/env python
"""Train and evaluate the compact CNN using a grouped manifest."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models


class ManifestImageDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        label_to_index: dict[str, int],
        image_size: int,
        channel_indices: tuple[int, ...],
        image_cache: Path | None = None,
    ):
        self.paths = frame["image_path"].tolist()
        self.labels = [label_to_index[label] for label in frame["label"]]
        self.image_size = image_size
        self.channel_indices = channel_indices
        self.image_cache = (
            np.load(image_cache, mmap_mode="r") if image_cache is not None else None
        )
        self.cache_indices = (
            frame["_cache_index"].to_numpy()
            if image_cache is not None
            else None
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        if self.image_cache is not None:
            array = (
                np.asarray(self.image_cache[self.cache_indices[index]], dtype=np.float32)
                / 255.0
            )
        else:
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
        return tensor * mask, self.labels[index]


class ConservativeCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.3),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Dropout2d(0.4),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(64, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(inputs))


def build_model(name: str, num_classes: int) -> nn.Module:
    if name == "simple":
        return ConservativeCNN(num_classes)
    if name == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(
            model.classifier[1].in_features,
            num_classes,
        )
        return model
    raise ValueError(f"Unknown model: {name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split-column", default="split")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--image-cache", type=Path, default=None)
    parser.add_argument(
        "--channels",
        default="vip",
        help="Image channels to retain: any subset of v, i, p (e.g. v, vi, vip).",
    )
    parser.add_argument(
        "--model",
        choices=("simple", "resnet18", "efficientnet_b0"),
        default="simple",
    )
    parser.add_argument(
        "--max-per-label",
        type=int,
        default=None,
        help="Development screening limit per label and split.",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def limit_per_label(
    frame: pd.DataFrame,
    maximum: int | None,
    seed: int,
) -> pd.DataFrame:
    if maximum is None:
        return frame
    parts = []
    for _, label_frame in frame.groupby("label", sort=True):
        count = min(maximum, len(label_frame))
        parts.append(label_frame.sample(n=count, random_state=seed))
    return pd.concat(parts, ignore_index=True)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate(model, loader, device, class_names):
    model.eval()
    predictions = []
    targets = []
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(device))
            predictions.extend(logits.argmax(dim=1).cpu().numpy().tolist())
            targets.extend(labels.numpy().tolist())
    return np.asarray(predictions), np.asarray(targets), class_names


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    frame = pd.read_csv(args.manifest)
    frame["_cache_index"] = np.arange(len(frame))
    required = {"image_path", "label", args.split_column}
    if not required.issubset(frame.columns):
        raise ValueError(f"Manifest must contain {sorted(required)}")
    unknown = set(frame[args.split_column]) - {"train", "val", "test"}
    if unknown:
        raise ValueError(f"Unknown split values: {unknown}")
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
    if any(part.empty for part in parts.values()):
        raise ValueError("Every split must contain at least one image")
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
            ManifestImageDataset(
                part,
                label_to_index,
                args.image_size,
                selected_channels,
                args.image_cache,
            ),
            batch_size=args.batch_size,
            shuffle=(split == "train"),
            num_workers=0 if args.image_cache is not None else args.num_workers,
        )
        for split, part in parts.items()
    }
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    model = build_model(args.model, len(class_names)).to(device)
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
        for images, labels in loaders["train"]:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(labels)
            total += len(labels)
        val_pred, val_true, _ = evaluate(model, loaders["val"], device, class_names)
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
    test_pred, test_true, _ = evaluate(model, loaders["test"], device, class_names)
    metrics = {
        "best_val_accuracy": best_val,
        "test_accuracy": float(accuracy_score(test_true, test_pred)),
        "test_macro_f1": float(f1_score(test_true, test_pred, average="macro")),
        "class_names": class_names,
        "split_sizes": {split: len(part) for split, part in parts.items()},
        "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "model": args.model,
        "channels": args.channels,
        "image_size": args.image_size,
        "seed": args.seed,
        "history": history,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    pd.DataFrame(
        confusion_matrix(test_true, test_pred),
        index=class_names,
        columns=class_names,
    ).to_csv(args.output_dir / "confusion_matrix.csv")
    report = classification_report(
        test_true,
        test_pred,
        target_names=class_names,
        zero_division=0,
    )
    (args.output_dir / "classification_report.txt").write_text(
        report, encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in metrics.items() if key != "history"}, indent=2))


if __name__ == "__main__":
    main()
