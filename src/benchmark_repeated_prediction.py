
import json
import statistics
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image
from torchvision.models import MobileNet_V3_Large_Weights

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "artifacts" / "onnx" / "mobilenetv3_fetal_fp32.onnx"
IMAGE_PATH = ROOT / "data" / "Images" / "Patient00001_Plane1_13_of_15.png"
OUTPUT_PATH = ROOT / "benchmarks" / "repeated_prediction_8threads.json"

ITERATIONS = 100
WARMUP_ITERATIONS = 20
THREADS = 8

if not MODEL_PATH.is_file():
    raise FileNotFoundError(MODEL_PATH)
if not IMAGE_PATH.is_file():
    raise FileNotFoundError(IMAGE_PATH)

# Initialize preprocessing once, outside the timed loop.
transform = MobileNet_V3_Large_Weights.DEFAULT.transforms()

options = ort.SessionOptions()
options.intra_op_num_threads = THREADS
options.inter_op_num_threads = 1
options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

session = ort.InferenceSession(
    str(MODEL_PATH),
    sess_options=options,
    providers=["CPUExecutionProvider"],
)
input_name = session.get_inputs()[0].name


def load_and_preprocess():
    with Image.open(IMAGE_PATH) as image:
        image = image.convert("RGB")
        tensor = transform(image)
    return tensor.unsqueeze(0).numpy().astype(np.float32)


def infer(array):
    return session.run(None, {input_name: array})[0]


def postprocess(logits):
    values = logits[0]
    shifted = values - np.max(values)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum()
    return int(np.argmax(probabilities))


# Warm up preprocessing and inference.
for _ in range(WARMUP_ITERATIONS):
    infer(load_and_preprocess())

load_ms = []
preprocess_ms = []
inference_ms = []
postprocess_ms = []
end_to_end_ms = []

for _ in range(ITERATIONS):
    total_start = time.perf_counter()

    start = time.perf_counter()
    with Image.open(IMAGE_PATH) as image:
        image = image.convert("RGB")
    load_ms.append((time.perf_counter() - start) * 1000)

    start = time.perf_counter()
    tensor = transform(image)
    array = tensor.unsqueeze(0).numpy().astype(np.float32)
    preprocess_ms.append((time.perf_counter() - start) * 1000)

    start = time.perf_counter()
    logits = infer(array)
    inference_ms.append((time.perf_counter() - start) * 1000)

    start = time.perf_counter()
    predicted_index = postprocess(logits)
    postprocess_ms.append((time.perf_counter() - start) * 1000)

    end_to_end_ms.append((time.perf_counter() - total_start) * 1000)


def summarize(values):
    return {
        "mean_ms": round(statistics.mean(values), 3),
        "median_ms": round(statistics.median(values), 3),
        "p95_ms": round(float(np.percentile(values, 95)), 3),
    }


report = {
    "experiment": "Repeated warmed single-image prediction",
    "provider": "CPUExecutionProvider",
    "threads": THREADS,
    "warmup_iterations": WARMUP_ITERATIONS,
    "measurement_iterations": ITERATIONS,
    "image": IMAGE_PATH.name,
    "image_loading": summarize(load_ms),
    "preprocessing": summarize(preprocess_ms),
    "inference": summarize(inference_ms),
    "postprocessing": summarize(postprocess_ms),
    "end_to_end": summarize(end_to_end_ms),
    "predicted_class_index": predicted_index,
    "note": (
        "Includes image loading, preprocessing, inference and postprocessing; "
        "excludes process startup and ONNX session construction."
    ),
}

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

print(json.dumps(report, indent=2))
print(f"\nSaved results to: {OUTPUT_PATH}")
