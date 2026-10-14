
import json
import os
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import psutil
from PIL import Image
from torchvision.models import MobileNet_V3_Large_Weights

CPU_THREADS = int(os.environ.get("ORT_CPU_THREADS", "8"))
ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "artifacts" / "experiments"/ "onnx_fp16" / "mobilenetv3_fetal_fp16.onnx"
IMAGE_DIR = ROOT / "data" / "Images"
OUTPUT_PATH = ROOT / "benchmarks" / (
    f"pipeline_profile_extended_fp16_{CPU_THREADS}threads.json"
)

NUM_IMAGES = 50
WARMUP_IMAGES = 10
MEASUREMENT_ROUNDS = 5

if not MODEL_PATH.exists():
    raise FileNotFoundError(MODEL_PATH)
if not IMAGE_DIR.is_dir():
    raise FileNotFoundError(IMAGE_DIR)

image_paths = sorted(IMAGE_DIR.glob("*.png"))[:NUM_IMAGES]
if len(image_paths) < NUM_IMAGES:
    raise RuntimeError(f"Expected {NUM_IMAGES} images; found {len(image_paths)}")

weights = MobileNet_V3_Large_Weights.DEFAULT
preprocess = weights.transforms()
class_names = [
    "Other",
    "Fetal brain",
    "Fetal thorax",
    "Maternal cervix",
    "Fetal femur",
    "Fetal abdomen",
]

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

process = psutil.Process(os.getpid())
rss_before_mb = process.memory_info().rss / (1024 ** 2)
peak_rss_mb = rss_before_mb


def load_image(path):
    with Image.open(path) as image:
        return image.convert("RGB")


def preprocess_image(image):
    tensor = preprocess(image).unsqueeze(0)
    return np.asarray(tensor.numpy(), dtype=np.float32)


def infer(array):
    return session.run(None, {input_name: array})[0]


def postprocess(logits):
    scores = logits[0]
    predicted_index = int(np.argmax(scores))
    shifted = scores - np.max(scores)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum()
    confidence = float(probabilities[predicted_index])
    return {
        "class_index": predicted_index,
        "class_name": class_names[predicted_index],
        "confidence": confidence,
    }


def update_peak_memory():
    global peak_rss_mb
    current_mb = process.memory_info().rss / (1024 ** 2)
    peak_rss_mb = max(peak_rss_mb, current_mb)


def summarize(values):
    return {
        "mean_ms": round(statistics.mean(values), 3),
        "median_ms": round(statistics.median(values), 3),
        "p95_ms": round(float(np.percentile(values, 95)), 3),
    }


# Warm up all stages.
for path in image_paths[:min(WARMUP_IMAGES, len(image_paths))]:
    image = load_image(path)
    array = preprocess_image(image)
    logits = infer(array)
    postprocess(logits)

update_peak_memory()

load_ms = []
preprocess_ms = []
inference_ms = []
postprocess_ms = []
end_to_end_ms = []

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

        start = time.perf_counter()
        prediction = postprocess(logits)
        postprocess_ms.append((time.perf_counter() - start) * 1000)

        end_to_end_ms.append((time.perf_counter() - total_start) * 1000)
        update_peak_memory()

report = {
    "experiment": "Extended pipeline profiling",
    "platform": platform.platform(),
    "processor": platform.processor(),
    "onnxruntime_version": ort.__version__,
    "provider": "CPUExecutionProvider",
    "cpu_threads": CPU_THREADS,
    "graph_optimization": "ORT_ENABLE_ALL",
    "unique_images": len(image_paths),
    "measurement_rounds": MEASUREMENT_ROUNDS,
    "warmup_images": min(WARMUP_IMAGES, len(image_paths)),
    "image_loading": summarize(load_ms),
    "preprocessing": summarize(preprocess_ms),
    "model_inference": summarize(inference_ms),
    "postprocessing": summarize(postprocess_ms),
    "end_to_end": summarize(end_to_end_ms),
    "process_rss_before_benchmark_mb": round(rss_before_mb, 2),
    "process_peak_rss_observed_mb": round(peak_rss_mb, 2),
    "note": (
        "RSS is sampled during the benchmark, not a guaranteed OS-level "
        "absolute peak. It includes Python, libraries, model runtime, and data."
    ),
    "example_prediction": prediction,
}

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

print(json.dumps(report, indent=2))
print(f"\nSaved results to: {OUTPUT_PATH}")


