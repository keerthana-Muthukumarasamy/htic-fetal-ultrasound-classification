# Project Structure and File Guide

**Project:** HTIC Fetal Ultrasound Classification and Model Optimization

This document explains what each project folder contains and what the
main files are used for. It documents the current structure; it does not
require moving or renaming files.

## Project root

  -----------------------------------------------------------------------
  Item                                Purpose
  ----------------------------------- -----------------------------------
  `data/`                             Local dataset files.

  `src/`                              Python scripts for dataset checks,
                                      training, prediction, benchmarking,
                                      and profiling.

  `artifacts/`                        Trained model checkpoints, exported
                                      ONNX models, and optimization
                                      experiment outputs.

  `benchmarks/`                       JSON files containing measured
                                      model and pipeline performance
                                      results.

  `reports/`                          Dataset audits, validation results,
                                      optimization findings, and
                                      verification reports.

  `configs/`                          Reserved for configuration files;
                                      it was empty in the last
                                      inspection.

  `docs/`                             Intended location for submission
                                      documents such as the technical
                                      report and deployment proposal.

  `file_inventory.csv`                Earlier inventory of project files.

  `project_inventory_windows.csv`     Windows project file inventory.

  `.gitignore`                        Git exclusion rules, if created
                                      during setup.

  `README.md`                         Project overview and usage
                                      instructions; not present in the
                                      last root-file inspection.

  `requirements.txt`                  Python dependency list; not present
                                      in the last root-file inspection.

  `Dockerfile`                        Container build instructions; not
                                      present in the last root-file
                                      inspection.
  -----------------------------------------------------------------------

## `data/`

  Item        Purpose
  ----------- ---------------------------------------------
  `Images/`   Ultrasound image files used by the project.

Keep the dataset locally unless its license and size permit
redistribution. Document how to obtain it in the README.

## `src/` --- Python source code

  -----------------------------------------------------------------------
  File                                Purpose
  ----------------------------------- -----------------------------------
  `audit_dataset.py`                  Audits the dataset and records
                                      basic dataset information.

  `benchmark_batch_size.py`           Measures model performance with
                                      different batch sizes.

  `benchmark_fp32_fp16.py`            Compares FP32 and FP16 ONNX model
                                      performance on the laptop.

  `benchmark_onnx.py`                 Benchmarks ONNX inference
                                      performance.

  `benchmark_preloaded_images.py`     Benchmarks inference using images
                                      that are already loaded.

  `predict_onnx.py`                   Runs prediction on an image using
                                      an ONNX model.

  `profile_inference_stages.py`       Measures individual
                                      inference-related stages.

  `profile_pipeline.py`               Profiles image loading,
                                      preprocessing, and inference.

  `train.py`                          Trains the baseline fetal
                                      ultrasound image classifier.

  `train_mixed_precision.py`          Trains the classifier using
                                      automatic mixed precision (AMP).

  `validate_data_splits.py`           Checks dataset split validity and
                                      patient separation.
  -----------------------------------------------------------------------

## `artifacts/` --- Models and experiment outputs

### `artifacts/baseline/`

  File                      Purpose
  ------------------------- ------------------------------------
  `best_model.pth`          Saved baseline PyTorch checkpoint.
  `training_history.json`   Baseline training history.
  `test_metrics.json`       Baseline held-out test metrics.

### `artifacts/onnx/`

  ---------------------------------------------------------------------------------
  File                                          Purpose
  --------------------------------------------- -----------------------------------
  `mobilenetv3_fetal_fp32.onnx`                 FP32 ONNX export of the classifier.

  `mobilenetv3_fetal_fp32_dynamic_batch.onnx`   FP32 ONNX export with a dynamic
                                                batch dimension.

  `mobilenetv3_fetal_fp16.onnx`                 FP16-converted ONNX model
                                                considered for optimized inference.
  ---------------------------------------------------------------------------------

### `artifacts/experiments/amp_fp16/`

  File                         Purpose
  ---------------------------- ----------------------------------------------
  `best_model.pth`             Checkpoint from the AMP training experiment.
  `experiment_metadata.json`   Metadata describing the AMP experiment.
  `training_history.json`      Training history for the AMP experiment.

### `artifacts/experiments/onnx_fp16/`

  -----------------------------------------------------------------------
  File                                Purpose
  ----------------------------------- -----------------------------------
  `mobilenetv3_fetal_fp16.onnx`       FP16 ONNX experiment output. This
                                      duplicates the FP16 model stored
                                      under `artifacts/onnx/`; retain
                                      both until provenance and
                                      references are checked.

  -----------------------------------------------------------------------

### `artifacts/experiments/onnx_dynamic_int8/`

  ---------------------------------------------------------------------------
  File                                    Purpose
  --------------------------------------- -----------------------------------
  `mobilenetv3_fetal_dynamic_int8.onnx`   Dynamic INT8 ONNX experiment model.

  `quantization_metadata.json`            Metadata for the dynamic INT8
                                          experiment.
  ---------------------------------------------------------------------------

### `artifacts/experiments/onnx_static_int8/`

  ---------------------------------------------------------------------------
  File                                    Purpose
  --------------------------------------- -----------------------------------
  `calibration_manifest.json`             Records calibration inputs used for
                                          static quantization.

  `mobilenetv3_fetal_preprocessed.onnx`   Preprocessed ONNX model produced
                                          during static quantization
                                          preparation.

  `mobilenetv3_fetal_static_int8.onnx`    Static INT8 ONNX experiment model.

  `quantization_metadata.json`            Metadata for the static INT8
                                          experiment.
  ---------------------------------------------------------------------------

**Interpretation note:** AMP training, FP16 ONNX conversion, and INT8
quantization are distinct experiments. The recorded INT8 validation
results showed substantial classification-quality degradation, so those
models should be treated as experimental alternatives rather than the
selected deployment candidate.

## `benchmarks/` --- Performance measurements

  ---------------------------------------------------------------------------------
  File                                          Purpose
  --------------------------------------------- -----------------------------------
  `batch_benchmark.json`                        Results comparing batch sizes.

  `fp32_vs_dynamic_int8_cpu.json`               CPU comparison involving FP32 and
                                                dynamic INT8 models.

  `fp32_vs_fp16_laptop.json`                    Laptop CPU comparison of FP32 and
                                                FP16 model inference.

  `fp32_vs_static_int8_cpu.json`                CPU comparison involving FP32 and
                                                static INT8 models.

  `laptop_thread_benchmark.json`                Results comparing CPU thread
                                                counts.

  `onnx_cpu_comparison.json`                    ONNX CPU benchmark results.

  `pipeline_profile.json`                       Earlier pipeline profiling results.

  `pipeline_profile_controlled.json`            Pipeline profiling results from a
                                                controlled run.

  `pipeline_profile_controlled_1threads.json`   Controlled pipeline results with
                                                one CPU thread.

  `pipeline_profile_controlled_4threads.json`   Controlled pipeline results with
                                                four CPU threads.

  `pipeline_profile_controlled_8threads.json`   Controlled pipeline results with
                                                eight CPU threads.
  ---------------------------------------------------------------------------------

Keep results from different machines, execution providers, input
conditions, and benchmark scripts distinct when reporting comparisons.

## `reports/` --- Validation and experiment documentation

  -----------------------------------------------------------------------------------
  File or folder                                  Purpose
  ----------------------------------------------- -----------------------------------
  `amp_fp32_comparison.json`                      Structured comparison between AMP
                                                  and baseline training.

  `amp_fp32_comparison.md`                        Readable AMP-versus-baseline
                                                  comparison.

  `dataset_audit.json`                            Dataset audit results.

  `dynamic_int8_findings.md`                      Findings from the dynamic INT8
                                                  experiment.

  `fp32_vs_fp16_validation.json`                  Validation comparison between FP32
                                                  and FP16 ONNX models.

  `onnx_cpu_benchmark.md`                         Readable summary of ONNX CPU
                                                  benchmarking.

  `quantization_decision.md`                      Rationale for the quantization
                                                  decision.

  `quantization_validation_results.json`          Quantization validation results.

  `split_validation.json`                         Dataset split validation results.

  `onnx_verification/verification_report.txt`     Human-readable ONNX verification
                                                  report.

  `onnx_verification/verification_results.json`   Structured ONNX verification
                                                  results.
  -----------------------------------------------------------------------------------

## Suggested reading order for a reviewer

1.  Read `README.md` for the project overview and usage instructions
    once it has been created.
2.  Review `reports/dataset_audit.json` and
    `reports/split_validation.json` for dataset checks and split
    validation.
3.  Review `artifacts/baseline/` for the baseline checkpoint and
    held-out test metrics.
4.  Review `reports/amp_fp32_comparison.md` for the training-time
    optimization comparison.
5.  Review `reports/fp32_vs_fp16_validation.json` and
    `benchmarks/fp32_vs_fp16_laptop.json` for FP16 validation and laptop
    performance evidence.
6.  Review the INT8 findings and quantization decision to understand why
    those alternatives were not selected.
7.  Review CPU-thread and pipeline benchmark files for runtime
    optimization measurements.

