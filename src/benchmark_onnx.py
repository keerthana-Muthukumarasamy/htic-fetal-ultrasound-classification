
import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

# Project paths
ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "artifacts" / "onnx" / "mobilenetv3_fetal_fp32.onnx"
OUTPUT_PATH = ROOT / "benchmarks" / "laptop_thread_benchmark.json"

if not MODEL_PATH.is_file():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# Use identical input for every configuration.
# This is a model-only benchmark, not the full image pipeline.
rng = np.random.default_rng(42)
input_data = rng.random((1, 3, 224, 224), dtype=np.float32)

results = []

for thread_count in [1, 2, 4, 8]:
    options = ort.SessionOptions()
    options.intra_op_num_threads = thread_count
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

    session = ort.InferenceSession(
        str(MODEL_PATH),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )

    input_name = session.get_inputs()[0].name

    # Warm up before measuring.
    for _ in range(30):
        session.run(None, {input_name: input_data})

    # Measure 10 batches of 20 single-image inferences.
    # Each individual inference is timed.
    latencies_ms = []

    for _ in range(10):
        batch_latencies = []

        for _ in range(20):
            start = time.perf_counter()
            session.run(None, {input_name: input_data})
            batch_latencies.append(
                (time.perf_counter() - start) * 1000
            )

        latencies_ms.extend(batch_latencies)

    ordered = sorted(latencies_ms)
    p95 = float(np.percentile(ordered, 95))
    median = statistics.median(ordered)
    mean = statistics.mean(ordered)

    result = {
        "threads": thread_count,
        "mean_latency_ms": round(mean, 3),
        "median_latency_ms": round(median, 3),
        "p95_latency_ms": round(p95, 3),
        "throughput_images_per_second": round(1000 / mean, 2),
        "samples": len(latencies_ms),
    }

    results.append(result)

    print(
        f"Threads={thread_count} | "
        f"Median={median:.2f} ms | "
        f"P95={p95:.2f} ms | "
        f"Mean={mean:.2f} ms | "
        f"Throughput={1000 / mean:.2f} images/s"
    )

report = {
    "experiment": "Laptop ONNX Runtime CPU thread comparison",
    "model": str(MODEL_PATH),
    "input_shape": [1, 3, 224, 224],
    "platform": platform.platform(),
    "processor": platform.processor(),
    "onnxruntime_version": ort.__version__,
    "provider": "CPUExecutionProvider",
    "warmup_iterations_per_configuration": 30,
    "measured_iterations_per_configuration": 200,
    "results": results,
}

with OUTPUT_PATH.open("w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

print(f"\nResults saved to: {OUTPUT_PATH}")
