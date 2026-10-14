# HTIC Fetal Ultrasound Classification & Model Optimization

**Assignment:** Deep Learning Model Optimization & Edge Deployment\
**Institution:** Healthcare Technology Innovation Centre (HTIC), IIT
Madras\
**Model:** MobileNetV3-Large\
**Optimized inference format:** FP16 ONNX\
**Status:** Laptop CPU inference and benchmarking completed; edge-device
deployment proposed for future validation.

------------------------------------------------------------------------

## 1. Project Overview

This project develops and evaluates a deep learning pipeline for
classifying static two-dimensional fetal ultrasound images into six
anatomical categories. It focuses on model training, patient-level
evaluation, mixed-precision training, ONNX model conversion,
quantization experiments, CPU inference optimization, and deployment
feasibility for resource-constrained devices.

The objective is to evaluate the trade-offs between classification
performance, model size, inference latency, and deployment practicality.

## 2. Demo Videos

Watch the recorded demonstrations of the inference workflow:

**
htic_fetal_ultrasound\docs\sample_inference_video
**

The demonstrations show the inference workflow on a laptop and through
Docker. Refer to the source code and benchmark reports for
implementation and measurement details.


## 3. Dataset and Classification Task

The project uses the FETAL_PLANES_DB dataset.

-   **Total dataset:** 12,400 images from 1,792 patients.
-   **Development pool:** 7,129 images from 896 patients.
-   **Held-out test set:** 5,271 images from a separate group of 896
    patients.
-   **Number of classes:** Six.

The classification categories are:

1.  Other
2.  Fetal brain
3.  Fetal thorax
4.  Maternal cervix
5.  Fetal femur
6.  Fetal abdomen

Patient-level separation was used to reduce the risk of data leakage
between development and test data.

## 4. Technical Approach

### Model Training

A pretrained MobileNetV3-Large image classifier was fine-tuned for the
six-class classification task. Training-time Automatic Mixed Precision
(AMP) was evaluated using PyTorch on an NVIDIA Tesla T4 GPU.

### Automatic Mixed Precision (AMP) Training

Automatic Mixed Precision (AMP) was evaluated during training using
PyTorch autocasting and `GradScaler` on an NVIDIA Tesla T4 GPU. AMP uses
lower-precision operations where appropriate and may help manage GPU
memory consumption and training efficiency.

  Metric                  AMP experiment
  ----------------------- -----------------
  GPU                     NVIDIA Tesla T4
  Training duration       430.99 seconds
  Training epochs         5
  Peak GPU memory usage   786.2 MB
  Validation accuracy     93.96%
  Validation macro-F1     93.17%

For comparison, the recorded FP32 baseline achieved 94.09% validation
accuracy and 93.23% validation macro-F1.

These results indicate similar validation performance between the
recorded AMP experiment and the FP32 baseline. A comparable FP32
training duration was not recorded, so no training-speed improvement is
claimed.

**Optimization distinction:** AMP was evaluated as a training-time
optimization, whereas FP16 ONNX was evaluated as an inference-time model
optimization. They serve different purposes in the model lifecycle.

### Model Optimization

The trained FP32 model was converted to FP16 ONNX for inference using
ONNX Runtime.

The following experiments were conducted:

-   Training-time Automatic Mixed Precision (AMP) evaluation using
    PyTorch autocast and `GradScaler`.
-   FP32 baseline evaluation.
-   FP16 ONNX conversion and evaluation.
-   Dynamic INT8 quantization.
-   Static INT8 quantization.
-   CPU thread configuration and inference benchmarking.
-   Dynamic batching experiments.

The tested INT8 configurations substantially reduced validation
performance. Therefore, FP16 ONNX was selected as the preferred
optimized model among the evaluated configurations.

## 5. Evaluation Results

### Held-Out Test Performance

The held-out test set contains 5,271 images from patients separated from
the development pool.

  Metric              FP32 PyTorch baseline   FP16 ONNX
  ----------------- ----------------------- -----------
  Test accuracy                    92.1647%    92.1836%
  Test macro-F1                    91.1170%    91.1595%
  Model file size                  16.05 MB     8.05 MB

The FP16 ONNX model is approximately **49.81% smaller** than the FP32
model. The small differences in test accuracy and macro-F1 should be
interpreted as observed experimental results, not evidence that FP16
intrinsically improves classification quality.

### CPU Model-Only Inference Benchmark

The following model-only measurements were recorded on the laptop CPU
using ONNX Runtime.

  Metric                                   FP32 ONNX         FP16 ONNX
  -------------------------------- ----------------- -----------------
  Median model inference latency           4.6826 ms         4.4083 ms
  Mean model inference latency             5.0691 ms         4.4409 ms
  P95 model inference latency              7.3332 ms         4.9821 ms
  Throughput                         197.28 images/s   225.18 images/s

These are model-only benchmark measurements and exclude the full cost of
image loading and preprocessing.

### End-to-End Pipeline Benchmark

A separate pipeline profile measured image loading, preprocessing,
inference, and postprocessing.

  Metric                        FP32 pipeline   FP16 pipeline
  --------------------------- --------------- ---------------
  Median end-to-end latency         16.586 ms       17.026 ms
  P95 end-to-end latency            21.944 ms       21.361 ms
  Sampled peak process RSS          340.65 MB       345.56 MB

In this profile, FP16 did not improve median end-to-end latency. These
results demonstrate why model-only latency and full-pipeline latency
must be reported separately.

### INT8 Quantization Experiments

  Configuration     Validation accuracy   Validation macro-F1
  --------------- --------------------- ---------------------
  Dynamic INT8                   72.12%                65.17%
  Static INT8                    51.58%                27.75%

The tested INT8 configurations showed substantial accuracy degradation.
They were not selected as the final inference model.

## 6. Repository Structure

``` text
htic-fetal-ultrasound-classification/
├── artifacts/       # Trained models and optimized model artifacts
├── benchmarks/      # Benchmark measurements
├── docs/            # Technical and deployment documentation
│   └── sample_inference_video/  # Recorded demonstrations
├── reports/         # Experiment reports and summaries
├── src/             # Training, prediction, evaluation and profiling scripts
├── Dockerfile
├── requirements.txt
├── requirements-docker.txt
├── PROJECT_STRUCTURE_GUIDE.md
└── README.md
```

The dataset is not included in the repository. Obtain it separately and
place the required sample image under `data/Images/` before following
the commands below.

## 7. How to Run

### Option A --- Local Python

**Prerequisites:** Python, the project files, the FP16 ONNX model, and a
sample image.

Open PowerShell in the project directory:

``` powershell
cd E:\htic_fetal_ultrasound
```

Create and activate a virtual environment:

``` powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the dependencies:

``` powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run inference using the optimized FP16 ONNX model:

``` powershell
python src/predict_onnx.py data/Images/Patient00001_Plane1_1_of_15.png --model artifacts/experiments/onnx_fp16/mobilenetv3_fetal_fp16.onnx --threads 8
```

This command assumes the sample image and model exist at the specified
paths.

### Option B --- Docker

**Prerequisites:** Docker Desktop installed and running, plus the files
required by the Docker build.

Build the image from the project root:

``` powershell
docker build -t htic-fetal-ultrasound:latest .
```

Run inference using the bundled sample, if it is included in the image:

``` powershell
docker run --rm htic-fetal-ultrasound:latest /app/sample.png
```

Alternatively, mount the local image directory into the container:

``` powershell
docker run --rm -v "${PWD}\data\Images:/images" htic-fetal-ultrasound:latest /images/Patient00001_Plane1_1_of_15.png
```

The Docker commands depend on the image path, entrypoint, and
sample-file configuration in the committed Dockerfile. The mounted-image
command requires the sample image to exist locally.

## 8. Reports and Supporting Artifacts

from pathlib import Path

section = """## ⭐ Benchmark Evidence & Experimental Results

**All benchmark results, optimization experiments, and supporting evidence are available for technical review.**

📄 **[READ THE CONSOLIDATED BENCHMARK REPORT](HTIC_All_Benchmarks_and_Experiments_Consolidated_V2_FINAL.docx)**

This report consolidates the experiments conducted throughout the project, including model optimization, FP16 ONNX conversion, INT8 quantization, CPU inference benchmarking, and performance comparisons.

### Explore the Supporting Evidence

- 📊 [Detailed Benchmark Results](benchmarks/)
- 🧪 [Experiment Reports](reports/)
- 🧠 [Trained and Optimized Models](artifacts/)
- 💻 [Training, Inference & Benchmarking Code](src/)
- 🎥 [Laptop & Docker Demonstration Videos](docs/sample_inference_video/)
- 📑 [Technical and Deployment Documentation](docs/)

### Key Results

| Metric | Result |
|---|---:|
| FP16 ONNX model size | 8.05 MB |
| Model-size reduction vs. FP32 | 49.81% |
| Held-out test accuracy | 92.1836% |
| Held-out test macro-F1 | 91.1595% |
| Model-only median CPU inference latency | 4.4083 ms |
| End-to-end median pipeline latency | 17.026 ms |

> **Measurement note:** Model-only latency and end-to-end pipeline latency are separate measurements. FP16 reduced model size and modestly improved model-only latency, but did not improve median end-to-end latency in the recorded pipeline benchmark. All performance figures refer to the tested laptop environment; Jetson and tablet performance has not yet been measured.
"""
## 9. Edge Deployment Proposal
### Proposed Edge Deployment Device

| Component | Selection / Explanation |
|---|---|
| **Target Device** | NVIDIA Jetson Orin Nano 8 GB |
| **Deployment Objective** | Run fetal ultrasound image classification locally on an edge device. |
| **Model Architecture** | MobileNetV3-Large |
| **Optimized Model Format** | FP16 ONNX |
| **Inference Pipeline** | Ultrasound Image → Preprocessing → MobileNetV3-Large FP16 ONNX → Predicted Fetal Plane Class |
| **Device Selection Rationale** | Provides a compact NVIDIA computing platform for edge AI experimentation, with GPU acceleration capabilities and a potential path to optimized inference using compatible ONNX Runtime or TensorRT configurations. |
| **Expected Benefits** | Local inference, reduced dependence on cloud connectivity, and potential suitability for resource-constrained deployment environments. |
| **Current Validation** | Model accuracy, model size, and CPU inference performance have been evaluated on a laptop. |
| **Hardware Validation Status** | Jetson Orin Nano deployment and performance have not yet been experimentally validated. |
| **Next Steps** | Validate model compatibility, benchmark inference latency, measure memory and power consumption, assess thermal behavior, and verify classification consistency on the actual device. |

> **Note:** The NVIDIA Jetson Orin Nano 8 GB is the proposed target platform, not a device used for the reported benchmarks. Laptop measurements should not be interpreted as Jetson performance results.

## 10. Scope and Limitations

-   The demonstrated task is static 2D ultrasound image classification.
-   Laptop CPU inference and the reported benchmark configurations were
    evaluated experimentally.
-   FP16 ONNX reduced model file size but did not reduce median
    end-to-end latency in the recorded pipeline profile.
-   The tested INT8 configurations showed substantial
    validation-performance degradation.
-   Jetson and tablet deployment remain proposed extensions requiring
    on-device validation.
-   This is an engineering and research demonstration, not a clinically
    validated diagnostic system.

------------------------------------------------------------------------

**Reproducibility:** Refer to the committed source code, model
artifacts, benchmark files, and supporting reports when interpreting the
reported results. Actual performance depends on hardware, software
versions, thread settings, and measurement methodology.
