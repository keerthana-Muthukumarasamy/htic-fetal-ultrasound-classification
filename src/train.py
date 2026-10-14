
"""Baseline training pipeline for fetal ultrasound plane classification."""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Dataset
from torchvision.models import (
    MobileNet_V3_Large_Weights,
    mobilenet_v3_large,
)


CLASS_NAMES = [
    "Other",
    "Fetal brain",
    "Fetal thorax",
    "Maternal cervix",
    "Fetal femur",
    "Fetal abdomen",
]

IMAGE_SIZE = 224
SEED = 42


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class FetalPlaneDataset(Dataset):
    """Load images and labels from the dataset metadata CSV."""

    def __init__(self, dataframe, images_dir, transform):
        self.dataframe = dataframe.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.class_to_idx = {
            name: idx for idx, name in enumerate(CLASS_NAMES)
        }

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        image_path = self.images_dir / str(row["Image_name"])

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image = self.transform(image)

        label = self.class_to_idx[row["Plane"]]
        return image, label


def load_metadata(csv_path):
    """Read the semicolon-delimited metadata and validate columns."""
    df = pd.read_csv(csv_path, sep=";")
    df.columns = df.columns.str.strip()

    required = {"Image_name", "Patient_num", "Plane", "Train"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required CSV columns: {sorted(missing)}")

    df["Plane"] = df["Plane"].astype(str).str.strip()
    df["Image_name"] = df["Image_name"].astype(str).str.strip()
    df["Patient_num"] = df["Patient_num"].astype(str).str.strip()
    df["Train"] = df["Train"].astype(str).str.strip()

    unknown = sorted(set(df["Plane"]) - set(CLASS_NAMES))
    if unknown:
        raise ValueError(f"Unexpected class labels: {unknown}")

    return df


def create_model(num_classes, device):
    weights = MobileNet_V3_Large_Weights.DEFAULT
    model = mobilenet_v3_large(weights=weights)

    # Replace the final classification layer for our six classes.
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)

    model = model.to(device)
    return model, weights.transforms()


def evaluate(model, loader, device, criterion):
    model.eval()
    all_labels, all_predictions = [], []
    total_loss, total_samples = 0.0, 0

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            total_loss += loss.item() * labels.size(0)
            total_samples += labels.size(0)
            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(logits.argmax(dim=1).cpu().tolist())

    return {
        "loss": total_loss / max(total_samples, 1),
        "accuracy": accuracy_score(all_labels, all_predictions),
        "macro_f1": f1_score(
            all_labels,
            all_predictions,
            labels=list(range(len(CLASS_NAMES))),
            average="macro",
            zero_division=0,
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--output", default="artifacts/baseline")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_metadata(args.csv)

    # Preserve the dataset's existing patient-disjoint groups:
    # Train=1 is the development pool; Train=0 is held out for testing.
    development = df[df["Train"] == "1"].copy()
    test_df = df[df["Train"] == "0"].copy()

    if development.empty or test_df.empty:
        raise ValueError("Expected non-empty Train=1 and Train=0 groups.")

    overlap = set(development["Patient_num"]) & set(test_df["Patient_num"])
    if overlap:
        raise ValueError(
            f"Patient leakage detected: {len(overlap)} patients overlap."
        )

    # Split the development pool into train and validation by patient.
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=0.15, random_state=SEED
    )
    train_idx, val_idx = next(
        splitter.split(
            development,
            groups=development["Patient_num"],
        )
    )
    train_df = development.iloc[train_idx].copy()
    val_df = development.iloc[val_idx].copy()

    train_patients = set(train_df["Patient_num"])
    val_patients = set(val_df["Patient_num"])
    test_patients = set(test_df["Patient_num"])

    assert not train_patients & val_patients
    assert not train_patients & test_patients
    assert not val_patients & test_patients

    weights = MobileNet_V3_Large_Weights.DEFAULT
    transform = weights.transforms()

    train_ds = FetalPlaneDataset(train_df, args.images, transform)
    val_ds = FetalPlaneDataset(val_df, args.images, transform)
    test_ds = FetalPlaneDataset(test_df, args.images, transform)

    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(train_ds, shuffle=True, **loader_options)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_options)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_options)

    model, _ = create_model(len(CLASS_NAMES), device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_val_f1 = -1.0
    history = []

    for epoch in range(args.epochs):
        model.train()
        running_loss, seen = 0.0, 0

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            seen += labels.size(0)

        val_metrics = evaluate(model, val_loader, device, criterion)
        train_loss = running_loss / max(seen, 1)

        record = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
        }
        history.append(record)
        print(json.dumps(record))

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": CLASS_NAMES,
                    "image_size": IMAGE_SIZE,
                    "seed": SEED,
                },
                output_dir / "best_model.pth",
            )

    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Evaluate the selected checkpoint on the held-out test set only once.
    checkpoint = torch.load(
        output_dir / "best_model.pth",
        map_location=device,
        weights_only=True,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate(model, test_loader, device, criterion)

    with open(output_dir / "test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)

    print("Final held-out test metrics:")
    print(json.dumps(test_metrics, indent=2))


if __name__ == "__main__":
    main()
