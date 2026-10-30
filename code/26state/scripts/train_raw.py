#!/usr/bin/env python
"""Train raw 1D waveform or V-I trajectory baselines."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset


class RawDataset(Dataset):
    def __init__(
        self,
        indices: np.ndarray,
        cache_path: Path,
        labels: np.ndarray,
        input_mode: str,
        image_size: int,
    ):
        self.indices = indices
        self.cache = np.load(cache_path, mmap_mode="r")
        self.labels = labels
        self.input_mode = input_mode
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int):
        row = np.array(
            self.cache[self.indices[position]],
            dtype=np.float32,
            copy=True,
        )
        if self.input_mode == "raw1d":
            tensor = torch.from_numpy(row)
        else:
            voltage = normalize(row[0])
            current = normalize(row[1])
            histogram, _, _ = np.histogram2d(
                voltage,
                current,
                bins=self.image_size,
                range=[[-1, 1], [-1, 1]],
            )
            image = np.log1p(histogram)
            maximum = image.max()
            if maximum > 0:
                image /= maximum
            image = np.repeat(image[None, :, :], 3, axis=0)
            tensor = torch.from_numpy(image.astype(np.float32))
        return tensor, int(self.labels[position])


def normalize(values: np.ndarray) -> np.ndarray:
    minimum = float(values.min())
    maximum = float(values.max())
    if maximum == minimum:
        return np.zeros_like(values)
    return 2.0 * (values - minimum) / (maximum - minimum) - 1.0


class Raw1DCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(3, 16, 7, padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(inputs))


class VI2DCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(inputs))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--input-mode", choices=("raw1d", "vi"), required=True)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--max-per-label", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def limit_indices(frame: pd.DataFrame, maximum: int | None, seed: int):
    if maximum is None:
        return frame
    return pd.concat(
        [
            part.sample(
                n=min(maximum, len(part)),
                random_state=seed,
            )
            for _, part in frame.groupby("label", sort=True)
        ]
    )


def run_epoch(model, loader, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    loss_function = nn.CrossEntropyLoss(label_smoothing=0.05)
    predictions = []
    targets = []
    total_loss = 0.0
    total = 0
    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        with torch.set_grad_enabled(training):
            logits = model(inputs)
            loss = loss_function(logits, labels)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        total_loss += float(loss.item()) * len(labels)
        total += len(labels)
        predictions.extend(logits.argmax(dim=1).detach().cpu().numpy().tolist())
        targets.extend(labels.detach().cpu().numpy().tolist())
    return (
        total_loss / max(total, 1),
        np.asarray(predictions),
        np.asarray(targets),
    )


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    frame = pd.read_csv(args.manifest)
    class_names = sorted(frame["label"].unique())
    label_to_index = {label: index for index, label in enumerate(class_names)}
    frame = frame.reset_index(drop=True)
    frame["label_index"] = frame["label"].map(label_to_index)
    parts = {
        split: limit_indices(
            frame[frame["split"] == split],
            args.max_per_label,
            args.seed,
        )
        for split in ("train", "val", "test")
    }
    loaders = {
        split: DataLoader(
            RawDataset(
                part.index.to_numpy(),
                args.cache,
                part["label_index"].to_numpy(),
                args.input_mode,
                args.image_size,
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
    model = (
        Raw1DCNN(len(class_names))
        if args.input_mode == "raw1d"
        else VI2DCNN(len(class_names))
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    best_val = -1.0
    history = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_path = args.output_dir / "best_model.pth"
    for epoch in range(1, args.epochs + 1):
        loss, _, _ = run_epoch(model, loaders["train"], device, optimizer)
        _, val_pred, val_true = run_epoch(model, loaders["val"], device)
        val_accuracy = float(accuracy_score(val_true, val_pred))
        val_f1 = float(f1_score(val_true, val_pred, average="macro"))
        history.append(
            {
                "epoch": epoch,
                "train_loss": loss,
                "val_accuracy": val_accuracy,
                "val_macro_f1": val_f1,
            }
        )
        print(
            f"epoch={epoch:03d} loss={loss:.4f} "
            f"val_acc={val_accuracy:.4f} val_f1={val_f1:.4f}",
            flush=True,
        )
        if val_accuracy > best_val:
            best_val = val_accuracy
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": class_names,
                },
                best_path,
            )
    checkpoint = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    _, test_pred, test_true = run_epoch(model, loaders["test"], device)
    metrics = {
        "best_val_accuracy": best_val,
        "test_accuracy": float(accuracy_score(test_true, test_pred)),
        "test_macro_f1": float(f1_score(test_true, test_pred, average="macro")),
        "class_names": class_names,
        "input_mode": args.input_mode,
        "split_sizes": {split: len(part) for split, part in parts.items()},
        "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
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
