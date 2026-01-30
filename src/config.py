import os
from dotenv import load_dotenv

load_dotenv()

RTSP_URL = os.getenv("RTSP_URL")
DISCORD_WEBHOOK = os.getenv("DISCORD_WEBHOOK")


RTSP_USER = os.getenv("RTSP_USER")
RTSP_PASS = os.getenv("RTSP_PASS")
RTSP_HOST = os.getenv("RTSP_HOST")
RTSP_PORT = os.getenv("RTSP_PORT")

RTSP_IA_PATH = os.getenv("RTSP_IA_PATH")
RTSP_VIEW_PATH = os.getenv("RTSP_VIEW_PATH")

YOLO_MODEL = os.getenv("YOLO_MODEL", "yolov8n.pt")

RTSP_IA = os.getenv("RTSP_IA")
RTSP_VIEW = os.getenv("RTSP_VIEW")

WEB_HOST = os.getenv("WEB_HOST", "0.0.0.0")
WEB_PORT = int(os.getenv("WEB_PORT", 5000))
WEB_USER = os.getenv("WEB_USER", "admin")
WEB_PASSWORD = os.getenv("WEB_PASSWORD", "admin")

MODEL_PATH = os.getenv("MODEL_PATH", "yolov8n.pt")

IA_WIDTH = int(os.getenv("IA_WIDTH", 704))
IA_HEIGHT = int(os.getenv("IA_HEIGHT", 480))

VIDEO_WIDTH = int(os.getenv("VIDEO_WIDTH", 960))
VIDEO_HEIGHT = int(os.getenv("VIDEO_HEIGHT", 540))
MAX_VIDEO_DURATION = int(os.getenv("MAX_VIDEO_DURATION", 6))

ALERTA_COOLDOWN = int(os.getenv("ALERTA_COOLDOWN", 50))

CLASSES_ALERTA = ["person", "car", "motorcycle", "bus", "truck"]
MOVIMENTO_MIN_AREA = 8000
JANELA_EVENTO = 3


def build_rtsp(path: str) -> str:
    return f"rtsp://{RTSP_USER}:{RTSP_PASS}@{RTSP_HOST}:{RTSP_PORT}{path}"
