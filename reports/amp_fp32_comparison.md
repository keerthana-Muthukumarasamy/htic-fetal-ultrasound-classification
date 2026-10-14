# FP32 Baseline vs AMP FP16 Training

## Best validation results

| Metric | FP32 baseline | AMP FP16 |
|---|---:|---:|
| Best epoch | 5 | 5 |
| Validation accuracy | 94.0934% | 93.9560% |
| Validation macro-F1 | 0.932310 | 0.931684 |
| Total training time | Not recorded | 430.99 seconds |
| Peak GPU memory | Not recorded | 786.20 MB |

## Interpretation

AMP FP16 training completed five epochs on a Tesla T4 GPU, with FP32
validation. Its best validation macro-F1 was 0.931684,
compared with 0.932310 for the FP32 baseline.

The held-out test set was not evaluated during the AMP experiment.
A training speedup cannot be claimed because a comparable FP32 training
duration was not recorded. GPU memory reduction also cannot be calculated
without a baseline measurement.

## Reproducibility

- Optimizer: AdamW
- Learning rate: 0.0001
- Batch size: 32
- Epochs: 5
- Seed: 42
- AMP: FP16 autocast with GradScaler
- Validation: FP32
