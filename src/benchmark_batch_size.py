
import json
import statistics
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image
from torchvision import transforms
from torchvision.models import MobileNet_V3_Large_Weights

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "artifacts" / "onnx" / "mobilenetv3_fetal_fp32_dynamic_batch.onnx"
IMAGE_DIR = ROOT / "data" / "Images"
OUTPUT = ROOT / "benchmarks" / "batch_benchmark.json"

THREADS = 8
NUM_IMAGES = 128
WARMUP = 10
REPEATS = 30
BATCH_SIZES = [1, 4, 8, 16]

paths = sorted(IMAGE_DIR.glob("*.png"))[:NUM_IMAGES]
if len(paths) < NUM_IMAGES:
    raise RuntimeError(f"Expected {NUM_IMAGES} images, found {len(paths)}")

preprocess = MobileNet_V3_Large_Weights.DEFAULT.transforms()

# Load and preprocess the same images once.
# This isolates model batching from disk and preprocessing costs.
t0 = time.perf_counter()
arrays = []

for path in paths:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        arrays.append(preprocess(rgb).numpy())

images = np.stack(arrays).astype(np.float32)
print(f"Prepared {len(images)} images: {images.shape}")

options = ort.SessionOptions()
options.intra_op_num_threads = THREADS
options.inter_op_num_threads = 1
options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

session = ort.InferenceSession(
    str(MODEL),
    sess_options=options,
    providers=["CPUExecutionProvider"],
)
input_name = session.get_inputs()[0].name

results = []

for batch_size in BATCH_SIZES:
    # Warm-up
    for _ in range(WARMUP):
        session.run(None, {input_name: images[:batch_size]})

    elapsed_per_image = []
    elapsed_per_batch = []

    for _ in range(REPEATS):
        start = time.perf_counter()

        for start_idx in range(0, len(images), batch_size):
            batch = images[start_idx:start_idx + batch_size]
            tick = time.perf_counter()
            outputs = session.run(None, {input_name: batch})
            elapsed_per_batch.append(
                (time.perf_counter() - tick) * 1000
            )

        total_seconds = time.perf_counter() - start
        elapsed_per_image.append(
            total_seconds * 1000 / len(images)
        )

    median_ms_per_image = statistics.median(elapsed_per_image)
    throughput = 1000 / median_ms_per_image

    result = {
        "batch_size": batch_size,
        "threads": THREADS,
        "images_per_repeat": len(images),
        "repeats": REPEATS,
        "median_ms_per_image": round(median_ms_per_image, 3),
        "throughput_images_per_second": round(throughput, 2),
        "median_batch_call_ms": round(
            statistics.median(elapsed_per_batch), 3
        ),
    }
    results.append(result)

    print(result)

report = {
    "experiment": "ONNX Runtime batch inference benchmark",
    "model": str(MODEL),
    "provider": "CPUExecutionProvider",
    "threads": THREADS,
    "input_shape_per_image": [3, 224, 224],
    "preprocessing_included_in_timing": False,
    "results": results,
}

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
print(f"\nSaved results to: {OUTPUT}")
