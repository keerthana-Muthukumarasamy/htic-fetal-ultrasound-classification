# Quantization Experiment: Validation and Decision

## Validation results

| Metric | FP32 ONNX | Dynamic INT8 | Static INT8 |
|---|---:|---:|---:|
| Accuracy | 94.0934% | 72.1154% | 51.5797% |
| Macro-F1 | 0.9323 | 0.6517 | 0.2775 |
| Agreement with FP32 | Reference | 72.3214% | 50.2060% |

## Model sizes

- FP32 ONNX: 16.049 MB
- Dynamic INT8 ONNX: 4.222 MB
- Static INT8 ONNX: 4.496 MB

## Decision

FP32 ONNX is the current preferred deployment candidate based on
validation classification quality and the earlier CPU benchmarks.

Neither quantized candidate is selected for deployment in its current form:
both show substantial validation quality degradation. Their smaller file
sizes do not compensate for this measured accuracy loss.

## Limitations and follow-up

- Validation results are not held-out test results.
- Quantization-induced accuracy loss has not yet been traced to a specific
  operator or calibration issue.
- CPU timings varied across benchmark runs; final deployment timing must
  be repeated under controlled conditions on the target laptop.
- The held-out test set remains reserved for final evaluation.
