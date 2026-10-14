
import json
import time
import platform
from pathlib import Path

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]

MODELS = {
    "FP32": ROOT / "artifacts" / "onnx" / "mobilenetv3_fetal_fp32.onnx",
    "FP16": ROOT / "artifacts" / "onnx" / "mobilenetv3_fetal_fp16.onnx",
}

THREADS = 8
WARMUP = 30
ITERATIONS = 200
SEED = 42

OUTPUT = ROOT / "benchmarks" / "fp32_vs_fp16_laptop.json"


def benchmark_model(name, model_path, input_data):
    if not model_path.exists():
        raise FileNotFoundError(f"{name} model not found: {model_path}")

    options = ort.SessionOptions()
    options.intra_op_num_threads = THREADS
    options.inter_op_num_threads = 1
    options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )

    session = ort.InferenceSession(
        str(model_path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    print(f"\n{name}")
    print("Model:", model_path)
    print("Size:", round(model_path.stat().st_size / (1024 ** 2), 2), "MiB")
    print("Input:", session.get_inputs()[0].shape)

    # Warm-up: exclude these runs from timing.
    for _ in range(WARMUP):
        session.run([output_name], {input_name: input_data})

    latencies_ms = []

    for _ in range(ITERATIONS):
        start = time.perf_counter()
        output = session.run([output_name], {input_name: input_data})
        latencies_ms.append((time.perf_counter() - start) * 1000)

    logits = output[0]

    if not np.isfinite(logits).all():
        raise ValueError(f"{name} produced non-finite output values.")

    latencies_ms = np.asarray(latencies_ms)

    result = {
        "model_path": str(model_path.relative_to(ROOT)),
        "model_size_mib": round(
            model_path.stat().st_size / (1024 ** 2), 4
        ),
        "provider": "CPUExecutionProvider",
        "cpu_threads": THREADS,
        "warmup_iterations": WARMUP,
        "measured_iterations": ITERATIONS,
        "median_latency_ms": round(float(np.median(latencies_ms)), 4),
        "p95_latency_ms": round(float(np.percentile(latencies_ms, 95)), 4),
        "mean_latency_ms": round(float(np.mean(latencies_ms)), 4),
        "throughput_images_per_second": round(
            1000 / float(np.mean(latencies_ms)), 2
        ),
        "output_shape": list(logits.shape),
    }

    print(json.dumps(result, indent=2))
    return result


def main():
    print("ONNX Runtime:", ort.__version__)
    print("CPU:", platform.processor() or platform.machine())
    print("Threads:", THREADS)
    print("Warm-up:", WARMUP, "| Iterations:", ITERATIONS)

    # Use exactly the same deterministic input for both models.
    rng = np.random.default_rng(SEED)
    input_data = rng.random(
        (1, 3, 224, 224), dtype=np.float32
    )

    results = {}

    # Run each model with identical settings and input.
    for name, path in MODELS.items():
        results[name] = benchmark_model(name, path, input_data)

    fp32 = results["FP32"]
    fp16 = results["FP16"]

    comparison = {
        "fp32_to_fp16_size_reduction_percent": round(
            (1 - fp16["model_size_mib"] / fp32["model_size_mib"]) * 100,
            2,
        ),
        "fp16_median_latency_change_percent": round(
            (fp16["median_latency_ms"] / fp32["median_latency_ms"] - 1)
            * 100,
            2,
        ),
        "fp16_mean_latency_change_percent": round(
            (fp16["mean_latency_ms"] / fp32["mean_latency_ms"] - 1)
            * 100,
            2,
        ),
        "fp16_throughput_change_percent": round(
            (fp16["throughput_images_per_second"]
             / fp32["throughput_images_per_second"] - 1) * 100,
            2,
        ),
    }

    report = {
        "experiment": "FP32 vs FP16 ONNX laptop CPU benchmark",
        "hardware": platform.processor() or platform.machine(),
        "onnxruntime_version": ort.__version__,
        "input_shape": [1, 3, 224, 224],
        "input_note": "Same deterministic synthetic input for both models",
        "results": results,
        "comparison": comparison,
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\nCOMPARISON")
    print(json.dumps(comparison, indent=2))
    print("\nSaved report:", OUTPUT)


if __name__ == "__main__":
    main()
