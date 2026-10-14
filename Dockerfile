# ==============================================================================
# HTIC Fetal Ultrasound Classification & Model Optimization
# Production Inference Container: Lightweight Multi-Threaded ONNX Runtime CPU
# ==============================================================================

FROM python:3.10-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=8
# Note: ORT_CPU_THREADS is NOT a recognized ONNX Runtime environment variable.
# Thread count for ORT is set via SessionOptions.intra_op_num_threads in predict_onnx.py.
# OMP_NUM_THREADS=8 controls the underlying OpenMP thread pool used by CPU kernels.

WORKDIR /app

# Install minimal OS runtime libraries (OpenMP runtime libgomp1 for multi-threaded CPU inference)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install lightweight production inference dependencies
COPY requirements-docker.txt /app/requirements.txt
RUN pip install --no-cache-dir --retries 5 --timeout 60 -r /app/requirements.txt

# Copy application source code and deployment models
COPY src/ /app/src/
COPY artifacts/experiments/onnx_fp16/ /app/artifacts/experiments/onnx_fp16/
COPY artifacts/onnx/ /app/artifacts/onnx/
COPY data/Images/Patient00001_Plane1_1_of_15.png /app/sample.png

# Create non-root user for security best practices
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Set default execution to the deployed FP16 ONNX model with 8 CPU threads
ENTRYPOINT ["python", "/app/src/predict_onnx.py", "--model", "/app/artifacts/experiments/onnx_fp16/mobilenetv3_fetal_fp16.onnx", "--threads", "8"]

# Default argument displays help or can run the bundled sample image
CMD ["--help"]

