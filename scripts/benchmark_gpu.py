"""
Phase 3.3 — GPU latency benchmark.

Run after confirming GPU is active (3.1).

Usage:
    python scripts/benchmark_gpu.py

Targets:
    YOLO11s:    mean < 8ms  (T4),  mean < 6ms  (RTX 3060/3070)
    RTMPose-m:  mean < 3ms  (T4)
    Budget:     YOLO + RTMPose < 15ms total for 15-FPS real-time

If YOLO11s > 15ms or RTMPose > 5ms on GPU, the CUDA setup is broken.
"""
from __future__ import annotations

import sys
import time

import numpy as np

# ---------------------------------------------------------------------------
# 1. Environment check
# ---------------------------------------------------------------------------

print("=" * 60)
print("Netra AI — Phase 3 GPU benchmark")
print("=" * 60)

try:
    import torch
    cuda_ok = torch.cuda.is_available()
    print(f"CUDA available : {cuda_ok}")
    if cuda_ok:
        print(f"CUDA device    : {torch.cuda.get_device_name(0)}")
        print(f"CUDA version   : {torch.version.cuda}")
except ImportError:
    print("torch not installed — skipping torch check")
    cuda_ok = False

try:
    import onnxruntime as ort
    ort_providers = ort.get_available_providers()
    print(f"ORT providers  : {ort_providers}")
    ort_cuda = "CUDAExecutionProvider" in ort_providers
except ImportError:
    print("onnxruntime not installed")
    ort_cuda = False
    ort_providers = []

print()

if not cuda_ok:
    print("WARNING: CUDA not available — benchmarks will run on CPU (expect 5-10x slower)")
    print("Install onnxruntime-gpu and a CUDA-enabled torch build before deploying.")

# ---------------------------------------------------------------------------
# 2. YOLO11s benchmark
# ---------------------------------------------------------------------------

DETECTOR_PATH = "/models/yolo11s.pt"
WARMUP        = 10
RUNS          = 100
FRAME         = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
DEVICE        = "0" if cuda_ok else "cpu"

print("-" * 60)
print(f"YOLO11s benchmark  (path={DETECTOR_PATH}  device={DEVICE})")
print("-" * 60)

try:
    from ultralytics import YOLO
    model = YOLO(DETECTOR_PATH)

    try:
        yolo_device = str(next(model.model.parameters()).device)
        print(f"YOLO model device: {yolo_device}")
    except Exception:
        pass

    print(f"Warmup ({WARMUP} runs)…", end=" ", flush=True)
    for _ in range(WARMUP):
        model(FRAME, device=DEVICE, verbose=False)
    print("done")

    print(f"Benchmark ({RUNS} runs)…", end=" ", flush=True)
    yolo_times: list[float] = []
    for _ in range(RUNS):
        t0 = time.perf_counter()
        model(FRAME, device=DEVICE, verbose=False)
        yolo_times.append((time.perf_counter() - t0) * 1000)
    print("done")

    yolo_mean = float(np.mean(yolo_times))
    yolo_p95  = float(np.percentile(yolo_times, 95))
    yolo_min  = float(np.min(yolo_times))
    print(f"YOLO11s  mean={yolo_mean:.1f}ms  p95={yolo_p95:.1f}ms  min={yolo_min:.1f}ms")

    if yolo_mean > 15:
        print("FAIL: YOLO11s > 15ms — CUDA setup problem, do not continue to Phase 4")
        sys.exit(1)
    elif yolo_mean > 8:
        print("WARN: YOLO11s > 8ms target — acceptable on older GPU, watch total budget")
    else:
        print("PASS: YOLO11s within target")

except FileNotFoundError:
    print(f"SKIP: {DETECTOR_PATH} not found — copy model to /models/ first")
    yolo_mean = None

print()

# ---------------------------------------------------------------------------
# 3. RTMPose-m benchmark
# ---------------------------------------------------------------------------

RTMPOSE_PATH = "/models/rtmpose-m.onnx"
POSE_WARMUP  = 10
POSE_RUNS    = 200

print("-" * 60)
print(f"RTMPose-m benchmark  (path={RTMPOSE_PATH})")
print("-" * 60)

try:
    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if ort_cuda else ["CPUExecutionProvider"]
    )
    session   = ort.InferenceSession(RTMPOSE_PATH, providers=providers)
    active    = session.get_providers()[0]
    print(f"RTMPose provider: {active}")
    if active != "CUDAExecutionProvider":
        print("WARNING: RTMPose not using CUDA")

    input_name = session.get_inputs()[0].name
    crop = np.random.randn(1, 3, 256, 192).astype(np.float32)

    print(f"Warmup ({POSE_WARMUP} runs)…", end=" ", flush=True)
    for _ in range(POSE_WARMUP):
        session.run(None, {input_name: crop})
    print("done")

    print(f"Benchmark ({POSE_RUNS} runs)…", end=" ", flush=True)
    pose_times: list[float] = []
    for _ in range(POSE_RUNS):
        t0 = time.perf_counter()
        session.run(None, {input_name: crop})
        pose_times.append((time.perf_counter() - t0) * 1000)
    print("done")

    pose_mean = float(np.mean(pose_times))
    pose_p95  = float(np.percentile(pose_times, 95))
    pose_min  = float(np.min(pose_times))
    print(f"RTMPose-m  mean={pose_mean:.1f}ms  p95={pose_p95:.1f}ms  min={pose_min:.1f}ms")

    if pose_mean > 5:
        print("FAIL: RTMPose > 5ms — CUDA setup problem, do not continue to Phase 4")
        sys.exit(1)
    elif pose_mean > 3:
        print("WARN: RTMPose > 3ms target — acceptable on older GPU, watch total budget")
    else:
        print("PASS: RTMPose-m within target")

except FileNotFoundError:
    print(f"SKIP: {RTMPOSE_PATH} not found — copy model to /models/ first")
    pose_mean = None

print()

# ---------------------------------------------------------------------------
# 4. Total budget summary
# ---------------------------------------------------------------------------

print("=" * 60)
print("BUDGET SUMMARY (target: < 15ms total for 15 FPS)")
print("=" * 60)

if yolo_mean is not None and pose_mean is not None:
    total = yolo_mean + pose_mean
    # Assume 3 persons average per frame → pose runs 3 crops
    total_3p = yolo_mean + pose_mean * 3
    print(f"  YOLO11s:          {yolo_mean:6.1f} ms")
    print(f"  RTMPose (1 crop): {pose_mean:6.1f} ms")
    print(f"  RTMPose (3 crops):{pose_mean*3:6.1f} ms  (typical 3-person scene)")
    print(f"  Total (3 persons):{total_3p:6.1f} ms")
    if total_3p < 15:
        print("  STATUS: within 15ms budget — safe for 15 FPS realtime")
    else:
        print("  STATUS: over 15ms budget — reduce batch size or upgrade GPU")
else:
    print("  One or more models not found — run after placing models in /models/")

print()
