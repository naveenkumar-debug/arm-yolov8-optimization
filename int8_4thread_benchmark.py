import cv2
import time
import psutil
import os
import onnxruntime
from ultralytics import YOLO


# ============================================================
# FORCE ONNX RUNTIME TO USE 4 CPU THREADS
# ============================================================

_original_session = onnxruntime.InferenceSession


def _4thread_session(path, sess_options=None, providers=None, **kwargs):

    if sess_options is None:
        sess_options = onnxruntime.SessionOptions()

    sess_options.intra_op_num_threads = 4
    sess_options.inter_op_num_threads = 1
    sess_options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL

    print("ONNX Runtime configuration:")
    print("  Intra-op threads :", sess_options.intra_op_num_threads)
    print("  Inter-op threads :", sess_options.inter_op_num_threads)
    print("  Execution mode   : ORT_SEQUENTIAL")

    return _original_session(
        path,
        sess_options=sess_options,
        providers=providers,
        **kwargs
    )


onnxruntime.InferenceSession = _4thread_session
# ============================================================
# CONFIGURATION
# ============================================================

VIDEO = "people.mp4"
MODEL = "yolov8n_512_static_int8.onnx"

IMG_SIZE = 512
CONF = 0.25


# ============================================================
# LOAD MODEL
# ============================================================

print()
print("==============================================")
print("     YOLOv8n STATIC INT8 - 4 THREAD TEST")
print("==============================================")
print()

print("Loading model...")
model = YOLO(MODEL)

print("Model loaded successfully.")


# ============================================================
# OPEN VIDEO
# ============================================================

cap = cv2.VideoCapture(VIDEO)

if not cap.isOpened():
    print("ERROR: Cannot open video")
    exit()

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))


print()
print("Model  :", MODEL)
print("Video  :", VIDEO)
print("Frames :", total_frames)
print("Input  :", "512 x 512")
print()


# ============================================================
# WARM-UP
# ============================================================

print("Warming up model...")

for i in range(5):

    ret, frame = cap.read()

    if ret:

        model.predict(
            frame,
            imgsz=IMG_SIZE,
            conf=CONF,
            verbose=False
        )


# ============================================================
# RESTART VIDEO
# ============================================================

cap.release()

cap = cv2.VideoCapture(VIDEO)
frames = 0

total_inference = 0
total_end_to_end = 0

cpu_sum = 0
ram_sum = 0
temp_sum = 0


start_total = time.perf_counter()


# ============================================================
# MAIN BENCHMARK LOOP
# ============================================================

while True:

    loop_start = time.perf_counter()

    ret, frame = cap.read()

    if not ret:
        break


    # --------------------------------------------------------
    # INFERENCE TIMING
    # --------------------------------------------------------

    inference_start = time.perf_counter()

    results = model.predict(
        frame,
        imgsz=IMG_SIZE,
        conf=CONF,
        verbose=False
    )

    inference_end = time.perf_counter()

    inference_ms = (
        inference_end - inference_start
    ) * 1000


    # --------------------------------------------------------
    # END-TO-END TIMING
    # --------------------------------------------------------

    loop_end = time.perf_counter()

    end_to_end_ms = (
        loop_end - loop_start
    ) * 1000


    total_inference += inference_ms
    total_end_to_end += end_to_end_ms


    # --------------------------------------------------------
    # SYSTEM MEASUREMENTS
    # --------------------------------------------------------

    cpu = psutil.cpu_percent(interval=None)

    ram = psutil.virtual_memory().percent


    # Raspberry Pi temperature

    try:

        with open(
            "/sys/class/thermal/thermal_zone0/temp",
            "r"
        ) as f:

            temperature = int(f.read()) / 1000

    except:

        temperature = 0


    cpu_sum += cpu
    ram_sum += ram
    temp_sum += temperature


    frames += 1


    # --------------------------------------------------------
    # PROGRESS OUTPUT
    # --------------------------------------------------------

    if frames % 100 == 0:

        elapsed = (
            time.perf_counter()
            - start_total
        )

        fps = frames / elapsed


        print(
            f"Frame {frames}/{total_frames} | "
            f"FPS: {fps:.2f} | "
            f"Inference: {inference_ms:.1f} ms | "
            f"CPU: {cpu:.1f}% | "
            f"RAM: {ram:.1f}% | "
            f"Temp: {temperature:.1f} C"
        )


# ============================================================
# FINAL CALCULATIONS
# ============================================================

end_total = time.perf_counter()

total_time = end_total - start_total


average_fps = frames / total_time

average_inference = (
    total_inference / frames
)

average_end_to_end = (
    total_end_to_end / frames
)

average_cpu = cpu_sum / frames

average_ram = ram_sum / frames

average_temp = temp_sum / frames


# ============================================================
# MODEL SIZE
# ============================================================

model_size_mb = (
    os.path.getsize(MODEL)
    / (1024 * 1024)
)
cap.release()


# ============================================================
# FINAL RESULTS
# ============================================================

print()

print("==============================================")
print("       FINAL INT8 4-THREAD RESULTS")
print("==============================================")

print(
    f"Model size          : "
    f"{model_size_mb:.2f} MB"
)

print(
    f"Frames processed    : "
    f"{frames}"
)

print(
    f"Total time          : "
    f"{total_time:.2f} seconds"
)

print("----------------------------------------------")

print(
    f"Average FPS         : "
    f"{average_fps:.2f}"
)

print(
    f"Average inference   : "
    f"{average_inference:.2f} ms"
)

print(
    f"Average end-to-end  : "
    f"{average_end_to_end:.2f} ms"
)

print("----------------------------------------------")

print(
    f"Average CPU         : "
    f"{average_cpu:.2f} %"
)

print(
    f"Average RAM         : "
    f"{average_ram:.2f} %"
)

print(
    f"Average temperature : "
    f"{average_temp:.2f} C"
)

print("==============================================")
print("       INT8 4-THREAD TEST COMPLETE")
print("==============================================")
