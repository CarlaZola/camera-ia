import cv2
import torch
from ultralytics import YOLO
from flask import Flask, Response, request
from functools import wraps

from src.config import (
    RTSP_IA,
    WEB_HOST,
    WEB_PORT,
    WEB_USER,
    WEB_PASSWORD,
    MODEL_PATH,
    CLASSES_ALERTA,
)

# ======================
# FLASK
# ======================
app = Flask(__name__)

# ======================
# AUTH BASIC
# ======================
def check_auth(username, password):
    return username == WEB_USER and password == WEB_PASSWORD

def authenticate():
    return Response(
        "Login necessário",
        401,
        {"WWW-Authenticate": 'Basic realm="Camera IA"'},
    )

def requires_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth = request.authorization
        if not auth or not check_auth(auth.username, auth.password):
            return authenticate()
        return f(*args, **kwargs)
    return decorated

# ======================
# GPU / MODELO
# ======================
print("CUDA disponível:", torch.cuda.is_available())
device = "cuda" if torch.cuda.is_available() else "cpu"

model = YOLO(MODEL_PATH)
model.to(device)

# ======================
# CAMERA
# ======================
cap = cv2.VideoCapture(RTSP_IA)
if not cap.isOpened():
    raise RuntimeError("❌ Não foi possível abrir RTSP")

# ======================
# STREAM MJPEG
# ======================
def gerar_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, device=0, verbose=False)[0]

        for box in results.boxes:
            cls = results.names[int(box.cls[0])]
            conf = float(box.conf[0])

            if cls in CLASSES_ALERTA:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"{cls} {conf:.2f}",
                    (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

        ok, buffer = cv2.imencode(".jpg", frame)
        if not ok:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + buffer.tobytes()
            + b"\r\n"
        )

# ======================
# ROTAS
# ======================
@app.route("/")
@requires_auth
def index():
    return """
    <html>
    <head><title>Câmera IA</title></head>
    <body style="background:black; text-align:center;">
        <h2 style="color:white;">Câmera com IA</h2>
        <img src="/video" style="width:90%; border:2px solid #0f0;">
    </body>
    </html>
    """

@app.route("/video")
@requires_auth
def video():
    return Response(
        gerar_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )

# ======================
# START
# ======================
if __name__ == "__main__":
    print(f"🌐 Acesse: http://SEU_IP:{WEB_PORT}")
    app.run(host=WEB_HOST, port=WEB_PORT, threaded=True)

