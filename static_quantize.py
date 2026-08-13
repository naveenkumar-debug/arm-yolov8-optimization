import os
import cv2
import numpy as np

from onnxruntime.quantization import (
    quantize_static,
    CalibrationDataReader,
    QuantType,
    QuantFormat,
)

MODEL = "yolov8n_512.onnx"
OUTPUT = "yolov8n_512_static_int8.onnx"
IMAGE_DIR = "validation"

IMAGE_FILES = [
    "frame_100.jpg",
    "frame_500.jpg",
    "frame_1000.jpg",
    "frame_1500.jpg",
]


def letterbox(image, size=512):
    h, w = image.shape[:2]

    scale = min(size / w, size / h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    resized = cv2.resize(image, (new_w, new_h))

    canvas = np.full((size, size, 3), 114, dtype=np.uint8)

    x = (size - new_w) // 2
    y = (size - new_h) // 2

    canvas[y:y + new_h, x:x + new_w] = resized

    return canvas


class YOLOCalibrationDataReader(CalibrationDataReader):

    def __init__(self):
        self.data = []

        for filename in IMAGE_FILES:
            path = os.path.join(IMAGE_DIR, filename)

            image = cv2.imread(path)

            if image is None:
                print("WARNING: Could not read", path)
                continue

            image = letterbox(image, 512)

            # BGR -> RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # uint8 -> float32
            image = image.astype(np.float32) / 255.0

            # HWC -> CHW
            image = np.transpose(image, (2, 0, 1))

            # Add batch dimension
            image = np.expand_dims(image, axis=0)

            self.data.append({"images": image})

        self.index = 0

        print("Calibration images:", len(self.data))

    def get_next(self):
        if self.index >= len(self.data):
            return None

        item = self.data[self.index]
        self.index += 1
        return item


print("========================================")
print("STATIC INT8 QUANTIZATION")
print("========================================")
print("Model:", MODEL)
print("Output:", OUTPUT)
print("Calibration:", IMAGE_DIR)
print()

reader = YOLOCalibrationDataReader()

quantize_static(
    model_input=MODEL,
    model_output=OUTPUT,
    calibration_data_reader=reader,
    quant_format=QuantFormat.QDQ,
    activation_type=QuantType.QUInt8,
    weight_type=QuantType.QInt8,
    per_channel=True,
)

print()
print("========================================")
print("STATIC INT8 COMPLETE")
print("========================================")
print("Saved:", OUTPUT)
