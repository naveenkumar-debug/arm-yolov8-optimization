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

## Optimization Approach

The optimization was implemented as a sequence of steps:

1. **FP32 Baseline**  
   YOLOv8n was first benchmarked as the baseline configuration on the ARM-based Raspberry Pi.

2. **ONNX Runtime**  
   The YOLOv8 model was converted to ONNX and executed using ONNX Runtime for CPU inference.

3. **Static INT8 Quantization**  
   The model was converted to a static INT8 ONNX model to reduce computational and memory requirements.

4. **Input Resolution Optimization**  
   The optimized pipeline uses a 512×512 input resolution to reduce the computational workload.

5. **4-Thread CPU Execution**  
   ONNX Runtime was configured to use 4 intra-operation CPU threads and 1 inter-operation thread.

### Optimization Configuration

- Execution Provider: CPU
- Intra-op threads: 4
- Inter-op threads: 1
- Execution mode: ORT_SEQUENTIAL
- Input resolution: 512×512
- Model format: ONNX
- Quantization: Static INT8

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
