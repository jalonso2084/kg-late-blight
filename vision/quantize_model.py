import os
import onnx
import pandas as pd
from PIL import Image
from onnxruntime.quantization import quantize_static, QuantType, CalibrationDataReader
import numpy as np

# --- CONFIGURATION ---
ONNX_MODEL_PATH = "vision/mosvit_blite.onnx"
QUANTIZED_MODEL_PATH = "vision/mosvit_blite_int8.onnx"
CALIBRATION_DATA_CSV = "data/labels.csv"
NUM_CALIBRATION_IMAGES = 100 # Number of images to use for calibration
IMAGE_SIZE = 224

# --- HELPER CLASS FOR STATIC QUANTIZATION ---
class VisionDataReader(CalibrationDataReader):
    def __init__(self, data_folder, model_path):
        self.image_paths = pd.read_csv(data_folder)['image_path'].tolist()
        self.model_path = model_path
        self.data_count = len(self.image_paths)
        self.enumerated_data = None

    def get_next(self):
        if self.enumerated_data is None:
            calibration_paths = self.image_paths[:NUM_CALIBRATION_IMAGES]
            self.enumerated_data = iter(self._preprocess_images(calibration_paths))

        return next(self.enumerated_data, None)

    def _preprocess_images(self, image_paths):
        session = onnx.load(self.model_path)
        input_name = session.graph.input[0].name

        for image_path in image_paths:
            try:
                img = Image.open(image_path).convert('RGB').resize((IMAGE_SIZE, IMAGE_SIZE))
                img_np = np.array(img, dtype=np.float32) / 255.0
                img_np = (img_np - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
                img_np = np.transpose(img_np, (2, 0, 1))
                img_np = np.expand_dims(img_np, axis=0)

                # --- THIS IS THE FIX: Ensure final type is float32 ---
                yield {input_name: img_np.astype(np.float32)}

            except Exception as e:
                print(f"Skipping problematic image {image_path}: {e}")
                continue

# --- MAIN FUNCTION ---
def quantize_model_static():
    print("Starting static quantization. This may take a few moments...")

    calibration_data_reader = VisionDataReader(CALIBRATION_DATA_CSV, ONNX_MODEL_PATH)

    quantize_static(
        model_input=ONNX_MODEL_PATH,
        model_output=QUANTIZED_MODEL_PATH,
        calibration_data_reader=calibration_data_reader,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QUInt8
    )

    print(f"\nModel successfully quantized using static method.")
    print(f"Quantized INT8 model saved to: {QUANTIZED_MODEL_PATH}")

if __name__ == "__main__":
    quantize_model_static()