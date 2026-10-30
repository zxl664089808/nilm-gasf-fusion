#!/usr/bin/env python
"""Shared CNN with separate appliance-type and ON/OFF heads."""

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

from train_from_manifest import ConservativeCNN


class MultiTaskDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        type_to_index: dict[str, int],
        state_to_index: dict[str, int],
        label_to_index: dict[str, int],
        image_size: int,
        channel_indices: tuple[int, ...],
    ):
        self.paths = frame["image_path"].tolist()
        self.types = [type_to_index[value] for value in frame["appliance"]]
        self.states = [state_to_index[value] for value in frame["state"]]
        self.labels = [label_to_index[value] for value in frame["label"]]
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
            self.types[index],
            self.states[index],
            self.labels[index],
        )


class MultiTaskCNN(nn.Module):
    def __init__(self, type_count: int, state_count: int):
        super().__init__()
        self.encoder = ConservativeCNN(type_count).features
        self.type_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, type_count),
        )
        self.state_head = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(32, state_count),
        )

    def forward(self, images: torch.Tensor):
        features = self.encoder(images).flatten(1)
        return self.type_head(features), self.state_head(features)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
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


def evaluate(model, loader, device, pair_to_joint_index):
    model.eval()
    type_predictions = []
    type_targets = []
    state_predictions = []
    state_targets = []
    joint_predictions = []
    joint_targets = []
    with torch.no_grad():
        for images, types, states, labels in loader:
            type_logits, state_logits = model(images.to(device))
            type_pred = type_logits.argmax(dim=1).cpu()
            state_pred = state_logits.argmax(dim=1).cpu()
            type_predictions.extend(type_pred.numpy().tolist())
            state_predictions.extend(state_pred.numpy().tolist())
            type_targets.extend(types.numpy().tolist())
            state_targets.extend(states.numpy().tolist())
            joint_predictions.extend(
                [
                    pair_to_joint_index[(int(type_value), int(state_value))]
                    for type_value, state_value in zip(type_pred, state_pred)
                ]
            )
            joint_targets.extend(labels.numpy().tolist())
    return {
        "type_pred": np.asarray(type_predictions),
        "type_true": np.asarray(type_targets),
        "state_pred": np.asarray(state_predictions),
        "state_true": np.asarray(state_targets),
        "joint_pred": np.asarray(joint_predictions),
        "joint_true": np.asarray(joint_targets),
    }


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    frame = pd.read_csv(args.manifest)
    required = {"image_path", "label", "appliance", "state", args.split_column}
    if not required.issubset(frame.columns):
        raise ValueError(f"Manifest must contain {sorted(required)}")
    type_names = sorted(frame["appliance"].unique())
    state_names = sorted(frame["state"].unique())
    joint_names = sorted(frame["label"].unique())
    if state_names != ["OFF", "ON"]:
        raise ValueError(f"Expected OFF/ON states, found {state_names}")
    type_to_index = {value: index for index, value in enumerate(type_names)}
    state_to_index = {value: index for index, value in enumerate(state_names)}
    label_to_index = {value: index for index, value in enumerate(joint_names)}
    pair_to_joint_index = {
        (type_index, state_index): label_to_index[
            f"{type_name}_{state_name}"
        ]
        for type_index, type_name in enumerate(type_names)
        for state_index, state_name in enumerate(state_names)
    }
    parts = {
        split: limit_per_label(
            frame[frame[args.split_column] == split].reset_index(drop=True),
            args.max_per_label,
            args.seed,
        )
        for split in ("train", "val", "test")
    }
    channel_lookup = {"v": 0, "i": 1, "p": 2}
    selected_channels = tuple(
        channel_lookup[channel]
        for channel in args.channels.lower()
        if channel in channel_lookup
    )
    loaders = {
        split: DataLoader(
            MultiTaskDataset(
                part,
                type_to_index,
                state_to_index,
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
    model = MultiTaskCNN(len(type_names), len(state_names)).to(device)
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
        for images, types, states, _ in loaders["train"]:
            images = images.to(device)
            types = types.to(device)
            states = states.to(device)
            optimizer.zero_grad(set_to_none=True)
            type_logits, state_logits = model(images)
            loss = criterion(type_logits, types) + criterion(state_logits, states)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(types)
            total += len(types)
        val = evaluate(
            model,
            loaders["val"],
            device,
            pair_to_joint_index,
        )
        val_accuracy = float(
            accuracy_score(val["joint_true"], val["joint_pred"])
        )
        val_f1 = float(
            f1_score(val["joint_true"], val["joint_pred"], average="macro")
        )
        scheduler.step(val_accuracy)
        history.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / max(total, 1),
                "val_accuracy": val_accuracy,
                "val_macro_f1": val_f1,
                "val_type_accuracy": float(
                    accuracy_score(val["type_true"], val["type_pred"])
                ),
                "val_state_accuracy": float(
                    accuracy_score(val["state_true"], val["state_pred"])
                ),
            }
        )
        print(
            f"epoch={epoch:03d} loss={total_loss/max(total,1):.4f} "
            f"val_joint={val_accuracy:.4f} val_f1={val_f1:.4f}",
            flush=True,
        )
        if val_accuracy > best_val:
            best_val = val_accuracy
            stale = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "type_names": type_names,
                    "state_names": state_names,
                    "joint_names": joint_names,
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
    test = evaluate(
        model,
        loaders["test"],
        device,
        pair_to_joint_index,
    )
    metrics = {
        "best_val_accuracy": best_val,
        "test_accuracy": float(
            accuracy_score(test["joint_true"], test["joint_pred"])
        ),
        "test_macro_f1": float(
            f1_score(test["joint_true"], test["joint_pred"], average="macro")
        ),
        "test_type_accuracy": float(
            accuracy_score(test["type_true"], test["type_pred"])
        ),
        "test_state_accuracy": float(
            accuracy_score(test["state_true"], test["state_pred"])
        ),
        "class_names": joint_names,
        "type_names": type_names,
        "state_names": state_names,
        "split_sizes": {split: len(part) for split, part in parts.items()},
        "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "model": "multitask_cnn",
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
        confusion_matrix(test["joint_true"], test["joint_pred"]),
        index=joint_names,
        columns=joint_names,
    ).to_csv(args.output_dir / "confusion_matrix.csv")
    (args.output_dir / "classification_report.txt").write_text(
        classification_report(
            test["joint_true"],
            test["joint_pred"],
            target_names=joint_names,
            zero_division=0,
        ),
        encoding="utf-8",
    )
    print(json.dumps({k: v for k, v in metrics.items() if k != "history"}, indent=2))


if __name__ == "__main__":
    main()
