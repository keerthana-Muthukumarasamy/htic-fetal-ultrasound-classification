
import json
import platform
import statistics
import time
from pathlib import Path
import os

import numpy as np
import onnxruntime as ort
from PIL import Image
from torchvision.models import MobileNet_V3_Large_Weights

CPU_THREADS = int(os.environ.get("ORT_CPU_THREADS", "8"))
ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "artifacts" / "onnx" / "mobilenetv3_fetal_fp32.onnx"
IMAGE_DIR = ROOT / "data" / "Images"
OUTPUT_PATH = (
    ROOT / "benchmarks" /
    f"pipeline_profile_controlled_{CPU_THREADS}threads.json"
)

NUM_IMAGES = 50
WARMUP_ROUNDS = 2
MEASUREMENT_ROUNDS = 5

if not MODEL_PATH.exists():
    raise FileNotFoundError(MODEL_PATH)
if not IMAGE_DIR.is_dir():
    raise FileNotFoundError(IMAGE_DIR)

image_paths = sorted(IMAGE_DIR.glob("*.png"))[:NUM_IMAGES]
if len(image_paths) < NUM_IMAGES:
    raise RuntimeError(f"Expected {NUM_IMAGES} PNG images; found {len(image_paths)}")

preprocess = MobileNet_V3_Large_Weights.DEFAULT.transforms()

options = ort.SessionOptions()
options.intra_op_num_threads = CPU_THREADS
options.inter_op_num_threads = 1
options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL


session = ort.InferenceSession(
    str(MODEL_PATH),
    sess_options=options,
    providers=["CPUExecutionProvider"],
)
input_name = session.get_inputs()[0].name

def load_image(path):
    with Image.open(path) as image:
        return image.convert("RGB")

def preprocess_image(image):
    tensor = preprocess(image).unsqueeze(0)
    return np.asarray(tensor.numpy(), dtype=np.float32)

def infer(array):
    return session.run(None, {input_name: array})[0]

# Warm up the model and preprocessing implementation.
for _ in range(WARMUP_ROUNDS):
    for path in image_paths[:5]:
        image = load_image(path)
        array = preprocess_image(image)
        infer(array)

load_ms = []
preprocess_ms = []
inference_ms = []
end_to_end_ms = []

# Repeat the same image set to reduce dependence on any one image.
for _ in range(MEASUREMENT_ROUNDS):
    for path in image_paths:
        total_start = time.perf_counter()

        start = time.perf_counter()
        image = load_image(path)
        load_ms.append((time.perf_counter() - start) * 1000)

        start = time.perf_counter()
        array = preprocess_image(image)
        preprocess_ms.append((time.perf_counter() - start) * 1000)

        start = time.perf_counter()
        logits = infer(array)
        inference_ms.append((time.perf_counter() - start) * 1000)

        end_to_end_ms.append((time.perf_counter() - total_start) * 1000)

def summarize(values):
    return {
        "mean_ms": round(statistics.mean(values), 3),
        "median_ms": round(statistics.median(values), 3),
        "p95_ms": round(float(np.percentile(values, 95)), 3),
    }

report = {
    "experiment": "Controlled laptop pipeline benchmark",
    "platform": platform.platform(),
    "processor": platform.processor(),
    "onnxruntime_version": ort.__version__,
    "provider": "CPUExecutionProvider",
    "cpu_threads": CPU_THREADS,
    "graph_optimization": "ORT_ENABLE_ALL",
    "unique_images": len(image_paths),
    "measurement_rounds": MEASUREMENT_ROUNDS,
    "warmup_images": WARMUP_ROUNDS * 5,
    "image_loading": summarize(load_ms),
    "preprocessing": summarize(preprocess_ms),
    "model_inference": summarize(inference_ms),
    "end_to_end": summarize(end_to_end_ms),
}

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

print(json.dumps(report, indent=2))
print(f"\nSaved results to: {OUTPUT_PATH}")
