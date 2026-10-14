"""
Evaluate the selected FP16 ONNX model exactly once on the existing patient-isolated
held-out test partition.

Run from the project root:
    python src/evaluate_fp16_heldout_test.py

Expected project layout:
    data/FETAL_PLANES_DB_data.csv
    data/Images/
    artifacts/onnx/mobilenetv3_fetal_fp16.onnx

This script does not train or tune a model. It uses CSV Train=0 as the held-out
test partition and the same torchvision MobileNetV3-Large pretrained transform.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import onnxruntime as ort
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torchvision.models import MobileNet_V3_Large_Weights

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "data" / "FETAL_PLANES_DB_data.csv"
IMAGE_DIR = ROOT / "data" / "Images"
MODEL_PATH = ROOT / "artifacts" / "onnx" / "mobilenetv3_fetal_fp16.onnx"
OUTPUT_PATH = ROOT / "reports" / "fp16_heldout_test_metrics.json"

CLASS_NAMES = [
    "Other",
    "Fetal brain",
    "Fetal thorax",
    "Maternal cervix",
    "Fetal femur",
    "Fetal abdomen",
]
LABEL_TO_INDEX = {name: i for i, name in enumerate(CLASS_NAMES)}


def resolve_image(image_name: object) -> Path:
    name = str(image_name).strip()
    path = IMAGE_DIR / name
    if path.suffix == "":
        path = path.with_suffix(".png")
    if not path.exists():
        # Do not silently substitute similarly named files.
        raise FileNotFoundError(f"Image referenced by CSV was not found: {path}")
    return path


def main() -> int:
    for required in (CSV_PATH, IMAGE_DIR, MODEL_PATH):
        if not required.exists():
            print(f"ERROR: Required path not found: {required}", file=sys.stderr)
            return 2

    df = pd.read_csv(CSV_PATH, sep=";")
    required_columns = {"Image_name", "Patient_num", "Plane", "Train"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"CSV is missing required columns {sorted(missing_columns)}. "
            f"Found columns: {list(df.columns)}"
        )

    # Normalize the original CSV Train field without changing the split.
    train_values = df["Train"].astype(str).str.strip()
    test_df = df.loc[train_values == "0"].copy()
    if len(test_df) != 5271:
        raise ValueError(
            f"Expected 5,271 held-out rows where Train=0, found {len(test_df)}. "
            "Stop and verify the CSV/split instead of evaluating a different set."
        )
    unique_test_patients = int(test_df["Patient_num"].nunique())
    if unique_test_patients != 896:
        raise ValueError(
            f"Expected 896 held-out patients, found {unique_test_patients}. "
            "Stop and verify the split."
        )

    unknown_labels = sorted(set(test_df["Plane"].astype(str)) - set(CLASS_NAMES))
    if unknown_labels:
        raise ValueError(f"Unexpected class labels in held-out rows: {unknown_labels}")

    # Match the preprocessing used for MobileNetV3-Large validation.
    transform = MobileNet_V3_Large_Weights.DEFAULT.transforms()

    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    # Use the project’s selected laptop CPU setting.
    session_options.intra_op_num_threads = 8
    session_options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(MODEL_PATH),
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )
    input_meta = session.get_inputs()[0]
    input_name = input_meta.name

    y_true: list[int] = []
    y_pred: list[int] = []
    missing = []
    start_all = time.perf_counter()

    for row_number, row in enumerate(test_df.itertuples(index=False), start=1):
        row_dict = row._asdict()
        image_name = row_dict["Image_name"]
        label = str(row_dict["Plane"])
        image_path = resolve_image(image_name)

        # Convert grayscale/RGBA sources to RGB, then apply official pretrained transforms.
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            tensor = transform(image).unsqueeze(0)

        # ONNX model keeps float32 external I/O even though internal graph uses FP16.
        input_array = tensor.cpu().numpy().astype(np.float32, copy=False)
        logits = session.run(None, {input_name: input_array})[0]
        if logits.shape != (1, 6):
            raise ValueError(f"Unexpected model output shape {logits.shape}; expected (1, 6)")

        pred_idx = int(np.argmax(logits[0]))
        y_true.append(LABEL_TO_INDEX[label])
        y_pred.append(pred_idx)

        if row_number % 500 == 0:
            print(f"Evaluated {row_number:,} / {len(test_df):,} held-out images")

    elapsed = time.perf_counter() - start_all
    accuracy = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", labels=list(range(6)), zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=list(range(6)))
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(6)),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    result = {
        "model": str(MODEL_PATH.relative_to(ROOT)).replace("\\", "/"),
        "model_precision": "FP16 internal ONNX graph; float32 external input/output",
        "runtime": {
            "onnxruntime_version": ort.__version__,
            "execution_provider": session.get_providers(),
            "intra_op_num_threads": 8,
            "inter_op_num_threads": 1,
        },
        "evaluation": {
            "split_definition": "Original CSV rows where Train == 0",
            "test_images": int(len(test_df)),
            "unique_test_patients": unique_test_patients,
            "patient_isolated_from_development_pool": True,
            "preprocessing": "MobileNet_V3_Large_Weights.DEFAULT.transforms(); image converted to RGB",
            "class_order": CLASS_NAMES,
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "confusion_matrix_rows_true_columns_predicted": cm.tolist(),
            "classification_report": report,
            "elapsed_seconds_including_image_loading_preprocessing_and_inference": elapsed,
            "mean_elapsed_ms_per_image_including_loading_preprocessing_and_inference": elapsed * 1000 / len(test_df),
        },
        "notes": [
            "This is one final held-out test evaluation; do not tune the model using these test results.",
            "End-to-end elapsed time is a sequential evaluation timing, not a controlled latency benchmark.",
            "Compare these FP16 test metrics only against the FP32 baseline metrics computed on the same held-out test partition.",
        ],
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("\n=== FP16 HELD-OUT TEST RESULTS ===")
    print(f"Test images: {len(test_df):,}")
    print(f"Unique patients: {unique_test_patients}")
    print(f"Accuracy: {accuracy * 100:.4f}%")
    print(f"Macro-F1: {macro_f1 * 100:.4f}%")
    print(f"Elapsed total: {elapsed:.2f} s")
    print(f"Saved: {OUTPUT_PATH}")
    print("\nPer-class metrics:")
    for name in CLASS_NAMES:
        values = report[name]
        print(
            f"  {name:16s} precision={values['precision']:.4f} "
            f"recall={values['recall']:.4f} f1={values['f1-score']:.4f} "
            f"support={int(values['support'])}"
        )
    print("\nConfusion matrix (rows=true, columns=predicted):")
    print(cm)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
