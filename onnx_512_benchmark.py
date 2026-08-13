import cv2
import time
import psutil
from ultralytics import YOLO

VIDEO = "people.mp4"
MODEL = "yolov8n_512.onnx"
IMG_SIZE = 512
CONF = 0.25

print("Loading YOLOv8n...")
model = YOLO(MODEL)

cap = cv2.VideoCapture(VIDEO)

if not cap.isOpened():
    print("ERROR: Cannot open video")
    exit()

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print()
print("========================================")
print("       YOLOv8 BASELINE BENCHMARK")
print("========================================")
print("Model  :", MODEL)
print("Video  :", VIDEO)
print("Frames :", total_frames)
print("Input  :", "512 x 512")
print("========================================")
print()

# Warm-up
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

# Restart video
cap.release()
cap = cv2.VideoCapture(VIDEO)

frames = 0
total_inference = 0
total_end_to_end = 0

cpu_sum = 0
ram_sum = 0
temp_sum = 0

start_total = time.perf_counter()

while True:

    loop_start = time.perf_counter()

    ret, frame = cap.read()

    if not ret:
        break

    # YOLO inference timing
    inference_start = time.perf_counter()

    results = model.predict(
        frame,
        imgsz=IMG_SIZE,
        conf=CONF,
        verbose=False
    )

    inference_end = time.perf_counter()

    inference_ms = (inference_end - inference_start) * 1000

    # End-to-end timing
    loop_end = time.perf_counter()
    end_to_end_ms = (loop_end - loop_start) * 1000

    total_inference += inference_ms
    total_end_to_end += end_to_end_ms

    # System measurements
    cpu = psutil.cpu_percent(interval=0.01)
    ram = psutil.virtual_memory().percent

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

    # Print every 100 frames
    if frames % 100 == 0:

        elapsed = time.perf_counter() - start_total

        fps = frames / elapsed

        print(
            f"Frame {frames}/{total_frames} | "
            f"FPS: {fps:.2f} | "
            f"Inference: {inference_ms:.1f} ms | "
            f"CPU: {cpu:.1f}% | "
            f"RAM: {ram:.1f}% | "
            f"Temp: {temperature:.1f} C"
        )

end_total = time.perf_counter()

total_time = end_total - start_total

average_fps = frames / total_time
average_inference = total_inference / frames
average_end_to_end = total_end_to_end / frames

average_cpu = cpu_sum / frames
average_ram = ram_sum / frames
average_temp = temp_sum / frames

# Model size
import os

model_size_mb = os.path.getsize(MODEL) / (1024 * 1024)

cap.release()

print()
print("========================================")
print("          FINAL BASELINE RESULTS")
print("========================================")

print(f"Model size          : {model_size_mb:.2f} MB")
print(f"Frames processed    : {frames}")
print(f"Total time          : {total_time:.2f} seconds")

print("----------------------------------------")

print(f"Average FPS         : {average_fps:.2f}")
print(f"Average inference   : {average_inference:.2f} ms")
print(f"Average end-to-end  : {average_end_to_end:.2f} ms")

print("----------------------------------------")

print(f"Average CPU         : {average_cpu:.2f} %")
print(f"Average RAM         : {average_ram:.2f} %")
print(f"Average temperature : {average_temp:.2f} C")

print("========================================")
print("        BASELINE COMPLETE")
print("========================================")
