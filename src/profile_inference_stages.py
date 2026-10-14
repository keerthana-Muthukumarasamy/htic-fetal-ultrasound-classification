
import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image
from torchvision import transforms
from torchvision.models import MobileNet_V3_Large_Weights

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "artifacts" / "onnx" / "mobilenetv3_fetal_fp32.onnx"
IMAGE_DIR = ROOT / "data" / "Images"
OUTPUT_PATH = ROOT / "benchmarks" / "pipeline_profile.json"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
if not IMAGE_DIR.is_dir():
    raise FileNotFoundError(f"Image directory not found: {IMAGE_DIR}")

# Choose a real PNG image from the dataset.
image_paths = sorted(IMAGE_DIR.glob("*.png"))
if not image_paths:
    raise FileNotFoundError(f"No PNG images found in {IMAGE_DIR}")

image_path = image_paths[0]
print(f"Profiling image: {image_path.name}")

# Use the exact preprocessing associated with pretrained model weights.
preprocess = MobileNet_V3_Large_Weights.DEFAULT.transforms()

# Use the best thread setting from our laptop benchmark.
options = ort.SessionOptions()
options.intra_op_num_threads = 8
options.inter_op_num_threads = 1
options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

session = ort.InferenceSession(
    str(MODEL_PATH),
    sess_options=options,
    providers=["CPUExecutionProvider"],
)
input_name = session.get_inputs()[0].name

def load_image():
    with Image.open(image_path) as image:
        return image.convert("RGB")

def preprocess_image(image):
    # Returns a normalized CHW tensor; add the batch dimension.
    tensor = preprocess(image)
    return tensor.unsqueeze(0).numpy().astype(np.float32)

def infer(input_array):
    return session.run(None, {input_name: input_array})[0]

# Warm-up each stage before collecting measurements.
for _ in range(10):
    image = load_image()
    array = preprocess_image(image)
    infer(array)

load_ms = []
preprocess_ms = []
inference_ms = []
end_to_end_ms = []

# Run stages sequentially for the same image.
for _ in range(100):
    start_total = time.perf_counter()

    start = time.perf_counter()
    image = load_image()
    load_ms.append((time.perf_counter() - start) * 1000)

    start = time.perf_counter()
    array = preprocess_image(image)
    preprocess_ms.append((time.perf_counter() - start) * 1000)

    start = time.perf_counter()
    logits = infer(array)
    inference_ms.append((time.perf_counter() - start) * 1000)

    end_to_end_ms.append((time.perf_counter() - start_total) * 1000)

def summarize(values):
    return {
        "mean_ms": round(statistics.mean(values), 3),
        "median_ms": round(statistics.median(values), 3),
        "p95_ms": round(float(np.percentile(values, 95)), 3),
    }

report = {
    "experiment": "Laptop end-to-end pipeline profile",
    "platform": platform.platform(),
    "processor": platform.processor(),
    "onnxruntime_version": ort.__version__,
    "provider": "CPUExecutionProvider",
    "cpu_threads": 8,
    "image": image_path.name,
    "input_shape": list(array.shape),
    "warmup_iterations": 10,
    "measured_iterations": 100,
    "image_loading": summarize(load_ms),
    "preprocessing": summarize(preprocess_ms),
    "model_inference": summarize(inference_ms),
    "end_to_end": summarize(end_to_end_ms),
}

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with OUTPUT_PATH.open("w", encoding="utf-8") as file:
    json.dump(report, file, indent=2)

print("\nPipeline profile:")
for stage in ["image_loading", "preprocessing", "model_inference", "end_to_end"]:
    print(f"{stage}: {report[stage]}")

print(f"\nResults saved to: {OUTPUT_PATH}")
print("Model output shape:", logits.shape)
