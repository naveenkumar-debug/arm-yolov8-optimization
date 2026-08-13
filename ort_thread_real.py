import onnxruntime as ort
import cv2
import numpy as np
import time

MODEL = "yolov8n_512_static_int8.onnx"
IMAGE = "validation/frame_100.jpg"
IMG_SIZE = 512


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


# Load real image
image = cv2.imread(IMAGE)

if image is None:
    print("ERROR: Could not load", IMAGE)
    exit()

# Same general preprocessing used for YOLO
image = letterbox(image, IMG_SIZE)

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = image.astype(np.float32) / 255.0

image = np.transpose(image, (2, 0, 1))
image = np.expand_dims(image, axis=0)

print("Input shape:", image.shape)
print()


for threads in [1, 2, 3, 4]:

    options = ort.SessionOptions()

    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1

    options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )

    session = ort.InferenceSession(
        MODEL,
        sess_options=options,
        providers=["CPUExecutionProvider"]
    )

    input_name = session.get_inputs()[0].name

    # Warm-up
    for _ in range(10):
        session.run(None, {input_name: image})

    times = []

    # Benchmark
    for _ in range(50):

        start = time.perf_counter()

        session.run(None, {input_name: image})

        end = time.perf_counter()

        times.append((end - start) * 1000)

    avg_ms = np.mean(times)
    fps = 1000.0 / avg_ms

    print(
        f"Threads: {threads} | "
        f"Latency: {avg_ms:.2f} ms | "
        f"FPS: {fps:.2f}"
    )
