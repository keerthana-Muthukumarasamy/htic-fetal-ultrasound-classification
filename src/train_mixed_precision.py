
"""AMP FP16 training experiment; validation only, no held-out test evaluation."""

import argparse
import json
import random
import time
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset
from torchvision.models import MobileNet_V3_Large_Weights, mobilenet_v3_large

CLASS_NAMES = [
    "Other", "Fetal brain", "Fetal thorax",
    "Maternal cervix", "Fetal femur", "Fetal abdomen",
]
IMAGE_SIZE = 224
SEED = 42


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class FetalPlaneDataset(Dataset):
    def __init__(self, dataframe, images_dir, transform):
        self.dataframe = dataframe.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.class_to_idx = {name: i for i, name in enumerate(CLASS_NAMES)}

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        name = str(row["Image_name"]).strip()
        path = self.images_dir / name
        if not path.exists() and not Path(name).suffix:
            path = path.with_suffix(".png")
        if not path.is_file():
            raise FileNotFoundError(f"Image not found: {path}")
        with Image.open(path) as image:
            image = self.transform(image.convert("RGB"))
        return image, self.class_to_idx[row["Plane"]]


def load_metadata(csv_path):
    df = pd.read_csv(csv_path, sep=";")
    df.columns = df.columns.str.strip()
    required = {"Image_name", "Patient_num", "Plane", "Train"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    for col in ["Plane", "Image_name", "Patient_num", "Train"]:
        df[col] = df[col].astype(str).str.strip()
    unknown = sorted(set(df["Plane"]) - set(CLASS_NAMES))
    if unknown:
        raise ValueError(f"Unexpected labels: {unknown}")
    return df


def evaluate(model, loader, device, criterion):
    model.eval()
    labels_all, predictions_all = [], []
    total_loss, total_samples = 0.0, 0

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            # Keep validation in FP32 for a consistent evaluation procedure.
            with torch.autocast(device_type="cuda", enabled=False):
                logits = model(images)
                loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)
            total_samples += labels.size(0)
            labels_all.extend(labels.cpu().tolist())
            predictions_all.extend(logits.argmax(dim=1).cpu().tolist())

    return {
        "loss": total_loss / max(total_samples, 1),
        "accuracy": accuracy_score(labels_all, predictions_all),
        "macro_f1": f1_score(
            labels_all, predictions_all,
            labels=list(range(len(CLASS_NAMES))),
            average="macro", zero_division=0,
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("This FP16 AMP experiment requires a CUDA GPU.")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_metadata(args.csv)
    development = df[df["Train"] == "1"].copy()
    held_out = df[df["Train"] == "0"].copy()

    if development.empty or held_out.empty:
        raise ValueError("Expected non-empty development and held-out groups.")
    if set(development["Patient_num"]) & set(held_out["Patient_num"]):
        raise ValueError("Patient leakage between development and held-out data.")

    splitter = StratifiedGroupKFold(
        n_splits=5, shuffle=True, random_state=SEED
    )
    train_idx, val_idx = next(splitter.split(
        development,
        y=development["Plane"],
        groups=development["Patient_num"],
    ))
    train_df = development.iloc[train_idx].copy()
    val_df = development.iloc[val_idx].copy()

    train_patients = set(train_df["Patient_num"])
    val_patients = set(val_df["Patient_num"])
    held_out_patients = set(held_out["Patient_num"])
    assert not train_patients & val_patients
    assert not train_patients & held_out_patients
    assert not val_patients & held_out_patients

    weights = MobileNet_V3_Large_Weights.DEFAULT
    transform = weights.transforms()
    train_ds = FetalPlaneDataset(train_df, args.images, transform)
    val_ds = FetalPlaneDataset(val_df, args.images, transform)

    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": True,
    }
    train_loader = DataLoader(train_ds, shuffle=True, **loader_options)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_options)

    model = mobilenet_v3_large(weights=weights)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 6)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda")

    history = []
    best_val_f1 = -1.0
    total_start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)

    print(json.dumps({
        "device": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "precision": "FP16 AMP training; FP32 validation",
        "train_images": len(train_ds),
        "validation_images": len(val_ds),
        "held_out_test_evaluated": False,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "cpu_threads": torch.get_num_threads(),
        "train_patients": len(train_patients),
        "validation_patients": len(val_patients),
    }))

    for epoch in range(args.epochs):
        epoch_start = time.perf_counter()
        model.train()
        running_loss, seen = 0.0, 0

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(images)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * labels.size(0)
            seen += labels.size(0)

        val_metrics = evaluate(model, val_loader, device, criterion)
        torch.cuda.synchronize()
        epoch_seconds = time.perf_counter() - epoch_start

        record = {
            "epoch": epoch + 1,
            "train_loss": running_loss / max(seen, 1),
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "epoch_seconds": epoch_seconds,
        }
        history.append(record)
        print(json.dumps(record))

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            torch.save({
                "model_state_dict": model.state_dict(),
                "class_names": CLASS_NAMES,
                "image_size": IMAGE_SIZE,
                "seed": SEED,
                "experiment": "amp_fp16",
            }, output_dir / "best_model.pth")

    torch.cuda.synchronize()
    total_seconds = time.perf_counter() - total_start

    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    metadata = {
        "experiment": "AMP FP16 training",
        "precision": "FP16 autocast with GradScaler; FP32 validation",
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "platform": platform.platform(),
        "seed": SEED,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "optimizer": "AdamW",
        "training_images": len(train_ds),
        "validation_images": len(val_ds),
        "held_out_test_evaluated": False,
        "total_training_seconds": total_seconds,
        "peak_gpu_memory_mb": round(
            torch.cuda.max_memory_allocated(device) / (1024 ** 2), 2
        ),
        "best_validation_macro_f1": best_val_f1,
        "best_epoch": max(history, key=lambda x: x["val_macro_f1"])["epoch"],
    }
    with open(output_dir / "experiment_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("AMP experiment completed. Held-out test set was not evaluated.")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
