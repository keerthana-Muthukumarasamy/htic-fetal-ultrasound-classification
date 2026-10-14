# FP32 CPU Inference Benchmark

## Environment and methodology

- Platform: Linux-6.6.122+-x86_64-with-glibc2.39
- PyTorch version: 2.11.0+cu130
- ONNX Runtime version: 1.31.0
- Execution provider: CPUExecutionProvider
- CPU threads: 2
- Input shape: [1, 3, 224, 224]
- Warm-up iterations per runtime: 20
- Timed iterations per runtime: 100
- Model precision: FP32
- Benchmark scope: model-only; image decoding and preprocessing excluded
- ONNX model size: 16.049 MB

## Results

| Metric | PyTorch CPU | ONNX Runtime CPU |
|---|---:|---:|
| Mean latency (ms/image) | 31.811 | 7.572 |
| Median latency (ms/image) | 30.425 | 7.296 |
| P95 latency (ms/image) | 37.813 | 8.864 |
| Throughput (images/s) | 31.44 | 132.06 |

Mean-latency speedup: **4.201x**.

## Interpretation and limitations

ONNX Runtime CPU showed lower model-only latency than PyTorch CPU in this
Colab environment. The result is specific to this runtime, machine,
thread configuration, input shape, and measurement procedure.

This is not a Windows laptop benchmark and excludes image loading and
preprocessing. Classification accuracy equivalence on the full held-out
test set has not been established by this timing experiment.

## Next validation

Repeat benchmarking on the target laptop and measure end-to-end latency.
Evaluate any optimized model on the same isolated test set before making
accuracy or deployment claims.
