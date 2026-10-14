
import argparse
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image


CLASS_NAMES = [
    "Other",
    "Fetal brain",
    "Fetal thorax",
    "Maternal cervix",
    "Fetal femur",
    "Fetal abdomen",
]

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Default to the optimized deployment model: FP16 ONNX
_candidate_fp16 = PROJECT_ROOT / "artifacts" / "experiments" / "onnx_fp16" / "mobilenetv3_fetal_fp16.onnx"
_fallback_fp16 = PROJECT_ROOT / "artifacts" / "onnx" / "mobilenetv3_fetal_fp16.onnx"
_fallback_fp32 = PROJECT_ROOT / "artifacts" / "onnx" / "mobilenetv3_fetal_fp32.onnx"

if _candidate_fp16.is_file():
    DEFAULT_MODEL = _candidate_fp16
elif _fallback_fp16.is_file():
    DEFAULT_MODEL = _fallback_fp16
else:
    DEFAULT_MODEL = _fallback_fp32


def preprocess_image(image_path):
    """
    Preprocess image for MobileNetV3-Large.
    Matches standard ImageNet transforms:
    Resize shortest edge to 232 (bilinear) -> CenterCrop 224x224 -> Normalize (mean/std).
    Works standalone with Pillow and NumPy without requiring PyTorch in production.
    """
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        w, h = image.size

        # Resize shortest edge to 232 maintaining aspect ratio
        if w < h:
            new_w = 232
            new_h = int(round(h * 232 / w))
        else:
            new_h = 232
            new_w = int(round(w * 232 / h))

        image = image.resize((new_w, new_h), Image.Resampling.BILINEAR)

        # Center crop to 224x224
        left = (new_w - 224) // 2
        top = (new_h - 224) // 2
        image = image.crop((left, top, left + 224, top + 224))

        # Convert to float32 [0, 1] and normalize
        arr = np.array(image, dtype=np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        arr = (arr - mean) / std

        # Transpose HWC -> CHW and add batch dimension -> (1, 3, 224, 224)
        arr = np.transpose(arr, (2, 0, 1))
        return np.expand_dims(arr, 0).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(
        description="Predict a fetal ultrasound plane using an ONNX model."
    )
    parser.add_argument("image", type=Path, help="Path to an ultrasound image")
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help="Path to the ONNX model",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=8,
        help="ONNX Runtime intra-op CPU threads (default: 8)",
    )
    args = parser.parse_args()

    if not args.image.is_file():
        parser.error(f"Image not found: {args.image}")
    if not args.model.is_file():
        parser.error(f"Model not found: {args.model}")
    if args.threads < 1:
        parser.error("--threads must be at least 1")

    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads
    options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )

    session = ort.InferenceSession(
        str(args.model),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )

    input_name = session.get_inputs()[0].name

    start = time.perf_counter()
    input_tensor = preprocess_image(args.image)
    preprocessing_ms = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    logits = session.run(None, {input_name: input_tensor})[0][0]
    inference_ms = (time.perf_counter() - start) * 1000

    # Convert logits to probabilities for a confidence estimate.
    shifted = logits - np.max(logits)
    probabilities = np.exp(shifted) / np.sum(np.exp(shifted))
    predicted_index = int(np.argmax(probabilities))

    print(f"Image: {args.image}")
    print(f"Predicted class: {CLASS_NAMES[predicted_index]}")
    print(f"Confidence: {probabilities[predicted_index] * 100:.2f}%")
    print(f"Preprocessing: {preprocessing_ms:.3f} ms")
    print(f"Model inference: {inference_ms:.3f} ms")
    print(f"CPU threads: {args.threads}")
    print("Note: confidence is a model probability, not a guarantee of correctness.")


if __name__ == "__main__":
    main()
