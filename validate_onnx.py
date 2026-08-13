import cv2
from ultralytics import YOLO
import os

MODEL = "yolov8n.onnx"
VIDEO = "people.mp4"

os.makedirs("validation", exist_ok=True)

model = YOLO(MODEL)
cap = cv2.VideoCapture(VIDEO)

if not cap.isOpened():
    print("ERROR: Cannot open video")
    exit()

total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print("Video frames:", total)
print("Starting validation...")

check_frames = [100, 500, 1000, 1500]

frame_no = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame_no += 1

    if frame_no in check_frames:

        results = model.predict(
            frame,
            imgsz=640,
            conf=0.25,
            verbose=False
        )

        annotated = results[0].plot()

        filename = f"validation/frame_{frame_no}.jpg"

        cv2.imwrite(filename, annotated)

        print()
        print("Frame:", frame_no)

        if results[0].boxes is None or len(results[0].boxes) == 0:
            print("No objects detected")
        else:
            for box in results[0].boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                name = model.names[class_id]

                print(name, "confidence:", round(confidence, 2))

    if frame_no >= max(check_frames):
        break

cap.release()

print()
print("================================")
print("VALIDATION COMPLETE")
print("================================")
print("Images saved in validation/")
