import onnxruntime as ort
import numpy as np
import time

MODEL = "yolov8n_512.onnx"
INPUT_NAME = "images"

x = np.random.rand(1, 3, 512, 512).astype(np.float32)

for threads in [1, 2, 3, 4]:

    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    session = ort.InferenceSession(
        MODEL,
        sess_options=options,
        providers=["CPUExecutionProvider"]
    )

    # Warm-up
    for _ in range(10):
        session.run(None, {INPUT_NAME: x})

    times = []

    for _ in range(50):
        start = time.perf_counter()
        session.run(None, {INPUT_NAME: x})
        end = time.perf_counter()
        times.append((end - start) * 1000)

    avg_ms = sum(times) / len(times)
    fps = 1000 / avg_ms

    print(
        f"Threads: {threads} | "
        f"Latency: {avg_ms:.2f} ms | "
        f"FPS: {fps:.2f}"
    )
