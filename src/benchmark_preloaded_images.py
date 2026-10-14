
import statistics
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image
from torchvision.models import MobileNet_V3_Large_Weights

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "artifacts" / "onnx" / "mobilenetv3_fetal_fp32.onnx"
IMAGE_DIR = ROOT / "data" / "Images"

NUM_IMAGES = 50
ROUNDS = 5

paths = sorted(IMAGE_DIR.glob("*.png"))[:NUM_IMAGES]
if len(paths) != NUM_IMAGES:
    raise RuntimeError(f"Expected {NUM_IMAGES} images; found {len(paths)}")

# Load images before timing the preloaded mode.
images = []
for path in paths:
    with Image.open(path) as im:
        images.append(im.convert("RGB"))

preprocess = MobileNet_V3_Large_Weights.DEFAULT.transforms()

options = ort.SessionOptions()
options.intra_op_num_threads = 8
options.inter_op_num_threads = 1
options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

session = ort.InferenceSession(
    str(MODEL_PATH),
    sess_options=options,
    providers=["CPUExecutionProvider"],
)
input_name = session.get_inputs()[0].name

def prepare(image):
    return np.asarray(
        preprocess(image).unsqueeze(0).numpy(),
        dtype=np.float32,
    )

def infer(array):
    return session.run(None, {input_name: array})[0]

# Warm up.
for image in images[:5]:
    infer(prepare(image))

preprocess_ms = []
inference_ms = []
preloaded_total_ms = []


for _ in range(ROUNDS):
    for image in images:
        total_start = time.perf_counter()

        start = time.perf_counter()
        array = prepare(image)
        preprocess_ms.append((time.perf_counter() - start) * 1000)

        start = time.perf_counter()
        infer(array)
        inference_ms.append((time.perf_counter() - start) * 1000)

        preloaded_total_ms.append(
            (time.perf_counter() - total_start) * 1000
        )


def report(values):
    return {
        "mean_ms": round(statistics.mean(values), 3),
        "median_ms": round(statistics.median(values), 3),
        "p95_ms": round(float(np.percentile(values, 95)), 3),
    }

print("Images preloaded into RAM:", len(images))
print("Preprocessing:", report(preprocess_ms))
print("Inference:", report(inference_ms))
print("Preloaded end-to-end:", report(preloaded_total_ms))
