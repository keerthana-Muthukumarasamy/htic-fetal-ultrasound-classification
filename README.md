# HTIC Fetal Ultrasound Classification & Model Optimization
**Assignment Submission:** Deep Learning Model Optimization & Edge Deployment  
**Candidate:** Deep Learning Engineer Candidate  
**Institution:** Healthcare Technology Innovation Centre (HTIC) – IIT Madras & DBT  

---

## What I Did in This Project

1. **Trained the Baseline Model:** Fine-tuned a pretrained `MobileNetV3-Large` classifier on 12,400 static 2D ultrasound images (`FETAL_PLANES_DB`) across 1,792 patients and 6 anatomical planes.
2. **Prevented Data Leakage:** Used strict patient-level separation (`StratifiedGroupKFold`) so images from the same patient never appeared in both training and testing sets.
3. **Training-Time Optimization (AMP):** Trained with PyTorch Automatic Mixed Precision (FP16 autocast + GradScaler) on a Tesla T4 GPU in 430.99 seconds, keeping peak GPU memory under **786.2 MB**.
4. **Post-Training Optimization (FP16 ONNX):** Converted the model to FP16 ONNX. Cut model size by **49.81%** (16.05 MB down to **8.05 MB**) with **zero drop in validation accuracy** (94.0934%).
5. **Rigorously Verified on 5,271 Isolated Test Images:**
   * **Baseline PyTorch (FP32):** **92.1647% Test Accuracy** | **91.1170% Macro-F1**
   * **Final Deployed Model (FP16 ONNX):** **92.1836% Test Accuracy** | **91.1595% Macro-F1** (+0.02% higher accuracy!)
6. **Why INT8 Was Rejected:** Tested Dynamic and Static INT8 quantization. Discovered that post-training INT8 dropped accuracy severely (down to 51%–72%) and actually ran **4.9× slower** on the laptop CPU (160 ms vs 32 ms) due to runtime dequantization overhead.
7. **Code & CPU Thread Optimization:**
   * Tuned CPU execution from 1 to 8 threads, cutting model inference time by **66.8%** (14.82 ms to **4.18 ms**) and total pipeline time by **32.6%** (25.84 ms to **16.59 ms**).
   * Implemented dynamic batching for PACS archives: Batch size 4 reached **277.72 images/second** (+29.9% throughput).
8. **Edge Deployment Proposal (Jetson & Tablets):**
   * **Target Workload:** Static 2D ultrasound image classification upon freeze-frame capture or PACS study review.
   * Selected **NVIDIA Jetson Orin Nano (8 GB)** for portable ultrasound consoles (runs model in ~1.1 ms, uses $<0.01\%$ of compute, operates fanless at 7W–15W).
   * Proposed tablet deployment using **Apple Neural Engine (Core ML)** and **Qualcomm Hexagon NPU (QNN)** for handheld ultrasound probes (runs in <2 ms, chip sleeps 98% of the time, saving battery).
   * **Future Extension:** Continuous 30–60 FPS live video stream guidance analyzed as a future performance roadmap.

---

## Direct Performance Comparison (Tested on 5,271 Patient-Isolated Images)

| Metric | PyTorch Baseline (Laptop CPU) | Final Deployment (ONNX FP16, 8 Threads) | What Changed / Benefit |
| :--- | :---: | :---: | :---: |
| **Model Disk Size** | 17.05 MB (`.pth`) | **8.05 MB** (`.onnx`) | **49.81% smaller file footprint** |
| **Model Inference (Mean)** | 31.81 ms | **4.44 ms** | **7.16× faster (86% time saved)** |
| **Model Inference (Median)** | 30.43 ms | **4.41 ms** | **6.90× faster (225 FPS)** |
| **Throughput** | 31.44 images/s | **225.18 images/s** | **+616% throughput improvement** |
| **Total End-to-End Latency** | ~45.0 ms | **16.59 ms** | **Instantaneous (<0.02s) result on click** |
| **RAM Usage (RSS)** | ~550 MB | **340.65 MB** | **~38% less memory used** |
| **Validation Accuracy** | 94.0934% | **94.0934%** | **Zero accuracy loss (0.00%)** |
| **Validation Macro-F1** | 93.2309% | **93.1697%** | **Virtually unchanged (-0.06 pp)** |
| **Held-Out Test Accuracy** | **92.1647%** | **92.1836%** | **+0.0189 pp (+0.02% higher accuracy!)** |
| **Held-Out Test Macro-F1** | **91.1170%** | **91.1595%** | **+0.0425 pp (+0.04% higher F1 score!)** |

---

## Submission Documents

* 📄 **Final Technical Report (1–2 Pages):**  
  * Word Document: [**`docs/FINAL_TECHNICAL_REPORT.docx`**](docs/FINAL_TECHNICAL_REPORT.docx)
  * Markdown Version: [**`docs/FINAL_TECHNICAL_REPORT.md`**](docs/FINAL_TECHNICAL_REPORT.md)
* 📄 **5. Jetson & Tablet Deployment Report (2–3 Pages):**  
  * Word Document: [**`docs/JETSON_AND_TABLET_DEPLOYMENT_REPORT_V2.docx`**](docs/JETSON_AND_TABLET_DEPLOYMENT_REPORT_V2.docx) *(or `docs/JETSON_AND_TABLET_DEPLOYMENT_REPORT.docx`)*
  * Markdown Version: [**`docs/JETSON_AND_TABLET_DEPLOYMENT_REPORT.md`**](docs/JETSON_AND_TABLET_DEPLOYMENT_REPORT.md)
* 📋 **Consolidated Benchmark Reference (Word .docx):** [**`HTIC_All_Benchmarks_and_Experiments_Consolidated.docx`**](HTIC_All_Benchmarks_and_Experiments_Consolidated.docx)
* 🐳 **Reproducibility Files:** [**`Dockerfile`**](Dockerfile) & [**`requirements.txt`**](requirements.txt)

---

## How to Run

### Using Docker (Recommended)
```bash
# 1. Build the lightweight production container (~150 MB)
docker build -t htic-fetal-ultrasound:latest .

# 2. Predict using the bundled sample image (instant test without mounting)
docker run --rm htic-fetal-ultrasound:latest /app/sample.png

# 3. Or predict any local image via volume mount
# Linux / macOS / WSL:
docker run --rm -v $(pwd)/data/Images:/images htic-fetal-ultrasound:latest /images/Patient00001_Plane1_1_of_15.png

# Windows PowerShell:
docker run --rm -v ${PWD}/data/Images:/images htic-fetal-ultrasound:latest /images/Patient00001_Plane1_1_of_15.png
```

### Using Local Python
```bash
# Install dependencies
pip install -r requirements.txt

# Run prediction with the final optimized FP16 model (defaulted to 8 CPU threads)
python src/predict_onnx.py data/Images/Patient00001_Plane1_1_of_15.png --model artifacts/experiments/onnx_fp16/mobilenetv3_fetal_fp16.onnx --threads 8

# Or simply (uses FP16 model and 8 threads automatically by default):
python src/predict_onnx.py data/Images/Patient00001_Plane1_1_of_15.png
```
