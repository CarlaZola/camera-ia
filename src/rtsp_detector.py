import cv2
import time
import torch
import psutil
from ultralytics import YOLO
from src.config import build_rtsp, YOLO_MODEL

# ======================
# RTSP (SEM SEGREDOS)
# ======================
RTSP_IA = build_rtsp("/cam/realmonitor?channel=1&subtype=1")
RTSP_VIEW = build_rtsp("/cam/realmonitor?channel=1&subtype=0")

# ======================
# GPU / MODELO
# ======================
print("CUDA disponível:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))

model = YOLO(YOLO_MODEL).to("cuda")

# ======================
# CAMERAS
# ======================
cap_ia = cv2.VideoCapture(RTSP_IA)
cap_view = cv2.VideoCapture(RTSP_VIEW)

if not cap_ia.isOpened() or not cap_view.isOpened():
    raise RuntimeError("❌ Não foi possível abrir RTSP")

w_ia = int(cap_ia.get(cv2.CAP_PROP_FRAME_WIDTH))
h_ia = int(cap_ia.get(cv2.CAP_PROP_FRAME_HEIGHT))
w_v = int(cap_view.get(cv2.CAP_PROP_FRAME_WIDTH))
h_v = int(cap_view.get(cv2.CAP_PROP_FRAME_HEIGHT))

sx = w_v / w_ia
sy = h_v / h_ia

prev_time = time.time()

# ======================
# REDE
# ======================
def total_bytes_recv():
    return sum(i.bytes_recv for i in psutil.net_io_counters(pernic=True).values())

net_prev = total_bytes_recv()
net_prev_time = time.time()

# ======================
# LOOP
# ======================
while True:
    start_latency = time.time()

    ret1, frame_ia = cap_ia.read()
    ret2, frame_view = cap_view.read()

    if not ret1 or not ret2:
        print("⚠️ Stream perdido")
        break

    latency_ms = (time.time() - start_latency) * 1000

    curr_time = time.time()
    fps = 1 / max(curr_time - prev_time, 1e-6)
    prev_time = curr_time

    results = model(frame_ia, device=0, verbose=False)[0]

    for box in results.boxes:
        cls = results.names[int(box.cls[0])]
        if cls in ["person", "car", "truck", "bus", "motorcycle"]:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1*sx), int(y1*sy), int(x2*sx), int(y2*sy)
            cv2.rectangle(frame_view, (x1,y1), (x2,y2), (0,255,0), 2)

    net_now = total_bytes_recv()
    net_now_time = time.time()
    mbps = ((net_now - net_prev) * 8) / max((net_now_time - net_prev_time) * 1024 * 1024, 1e-6)
    net_prev, net_prev_time = net_now, net_now_time

    gpu_mem = torch.cuda.memory_allocated() / 1024 / 1024
    gpu_total = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024

    overlay = [
        f"FPS IA: {fps:.1f}",
        f"Latência RTSP: {latency_ms:.0f} ms",
        f"Rede: {mbps:.2f} Mbps",
        f"GPU VRAM: {gpu_mem:.0f}/{gpu_total:.0f} MB",
        "Status: ONLINE",
    ]

    for i, text in enumerate(overlay):
        cv2.putText(frame_view, text, (20, 30 + i * 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2

