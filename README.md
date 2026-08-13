# arm-yolov8-optimization
YOLOv8 INT8 optimization and benchmarking on Arm architecture
# ARM YOLOv8 Optimization Benchmark

## Overview

This project benchmarks YOLOv8 on an ARM-based Raspberry Pi and evaluates
optimization using ONNX Runtime, static INT8 quantization, and multi-threaded
CPU inference.

## Hardware

- Raspberry Pi
- ARM CPU
- CPU inference

## Benchmark Results

| Metric | Baseline FP32 | Optimized INT8 |
|---|---:|---:|
| Model | YOLOv8n | YOLOv8n Static INT8 |
| Input | 640×640 | 512×512 |
| FPS | 3.69 | 16.64 |
| Inference Latency | 257.20 ms | 55.32 ms |
| End-to-End Latency | 260.69 ms | 59.75 ms |
| Model Size | 6.25 MB | 3.41 MB |
| Temperature | 48.95°C | 60.98°C |

## Optimization Techniques

- ONNX Runtime CPU inference
- Static INT8 quantization
- 4-thread execution
- Reduced input resolution
- ARM CPU benchmarking

## Performance Improvement

The optimized configuration achieved:

- **4.51× higher FPS**
- **78.5% lower inference latency**
- **45.4% smaller model**

> Note: The comparison uses different input resolutions
> (640×640 vs 512×512). Therefore, the measured improvement represents
> the combined effect of quantization, threading, and reduced input
> resolution.

## Objective

The objective is to demonstrate practical AI workload optimization
for ARM-based edge devices.
