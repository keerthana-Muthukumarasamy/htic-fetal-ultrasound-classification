# Dynamic INT8 Quantization Findings

## Measured results

| Metric | FP32 ONNX | Dynamic INT8 ONNX |
|---|---:|---:|
| Model size (MB) | 16.049 | 4.222 |
| Mean latency (ms) | 32.764 | 160.806 |
| Median latency (ms) | 34.353 | 131.420 |
| P95 latency (ms) | 58.477 | 319.458 |
| Throughput (images/s) | 30.52 | 6.22 |

Model size reduction: **73.69%**.
Dynamic INT8 mean latency was **4.91x** FP32 latency.

## Methodology

- ONNX Runtime CPUExecutionProvider
- Two intra-op CPU threads
- Input shape: 1 x 3 x 224 x 224
- 20 warm-up iterations and 100 timed iterations per model
- Random FP32 input; model-only inference
- Image decoding and preprocessing excluded
- Environment: Linux-6.6.122+-x86_64-with-glibc2.39

## Interpretation

Dynamic INT8 substantially reduced file size but performed worse on latency
and throughput in this benchmark. Therefore, this candidate is not selected
for latency-sensitive deployment based on these measurements.

A likely explanation is quantization overhead on supported operators without
sufficient acceleration of the model's convolution-heavy computation. This
remains a hypothesis until operator-level profiling is performed.

Classification quality on real ultrasound images has not been established
by this timing benchmark. Results are environment-specific and should be
rechecked on the target laptop.
