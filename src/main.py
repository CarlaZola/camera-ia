import cv2
import time
import os
import requests
import torch
from datetime import datetime
from ultralytics import YOLO

from src.config import (
    RTSP_URL,
    DISCORD_WEBHOOK,
    IA_WIDTH,
    IA_HEIGHT,
    VIDEO_WIDTH,
    VIDEO_HEIGHT,
    MAX_VIDEO_DURATION,
    ALERTA_COOLDOWN,
    CLASSES_ALERTA,
    MOVIMENTO_MIN_AREA,
    JANELA_EVENTO,
)

# ======================
# DETECÇÃO DE MOVIMENTO
# ======================
fgbg = cv2.createBackgroundSubtractorMOG2(
    history=300,
    varThreshold=50,
    detectShadows=False
)

# ======================
# CONTROLE DE EVENTOS
# ======================
gravando = False
inicio_gravacao = 0
ultimo_evento = 0
video_writer = None
nome_video = ""
ultimo_alerta = 0

# ======================
# FUNÇÕES DISCORD
# ======================


def enviar_snapshot_discord(frame, classe, confianca):
    global ultimo_alerta
    agora = time.time()

    if agora - ultimo_alerta < ALERTA_COOLDOWN:
        return

    ultimo_alerta = agora
    nome_img = f"alerta_{int(agora)}.jpg"
    cv2.imwrite(nome_img, frame)

    mensagem = (
        f"🚨 **Detecção IA**\n"
        f"Objeto: **{classe}**\n"
        f"Confiança: **{confianca:.2f}**\n"
        f"Hora: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}"
    )

    try:
        with open(nome_img, "rb") as img:
            r = requests.post(
                DISCORD_WEBHOOK,
                data={"content": mensagem},
                files={"file": img},
                timeout=10
            )
            print(f"📸 Snapshot enviado | status: {r.status_code}")
    finally:
        os.remove(nome_img)


def enviar_video_discord(nome_video, classe):
    mensagem = (
        f"🎥 **Vídeo do Evento**\n"
        f"Objeto: **{classe}**\n"
        f"Hora: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}"
    )

    try:
        with open(nome_video, "rb") as video:
            r = requests.post(
                DISCORD_WEBHOOK,
                data={"content": mensagem},
                files={"file": video},
                timeout=20
            )
            print(f"🎥 Vídeo enviado | status: {r.status_code}")
    finally:
        if os.path.exists(nome_video):
            os.remove(nome_video)
            print("🗑 Vídeo removido")


# ======================
# MODELO IA
# ======================
model = YOLO("models/yolov8n.pt")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f"Modelo carregado em {device}")

# ======================
# CAPTURA RTSP
# ======================
cap = cv2.VideoCapture(RTSP_URL)
if not cap.isOpened():
    raise RuntimeError("❌ Não foi possível abrir o RTSP")

print(f"IA: {IA_WIDTH}x{IA_HEIGHT} | VIDEO: {VIDEO_WIDTH}x{VIDEO_HEIGHT}")

# ======================
# LOOP PRINCIPAL
# ======================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # MOVIMENTO
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    mask = fgbg.apply(gray)
    mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)[1]
    contornos, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    movimento = any(cv2.contourArea(c) > MOVIMENTO_MIN_AREA for c in contornos)

    # IA
    deteccao = False
    classe_detectada = None
    confianca_detectada = 0

    if movimento:
        frame_ia = cv2.resize(frame, (IA_WIDTH, IA_HEIGHT))
        results = model(frame_ia, device=device, verbose=False)[0]

        for box in results.boxes:
            cls = results.names[int(box.cls[0])]
            conf = float(box.conf[0])

            if cls in CLASSES_ALERTA:
                deteccao = True
                classe_detectada = cls
                confianca_detectada = conf
                ultimo_evento = time.time()

                if not gravando:
                    gravando = True
                    inicio_gravacao = time.time()
                    nome_video = f"evento_{int(inicio_gravacao)}.mp4"
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    video_writer = cv2.VideoWriter(
                        nome_video,
                        fourcc,
                        20,
                        (VIDEO_WIDTH, VIDEO_HEIGHT)
                    )
                    print("🎥 Evento iniciado")

                break

    # GRAVA VÍDEO
    if gravando and video_writer:
        frame_video = cv2.resize(frame, (VIDEO_WIDTH, VIDEO_HEIGHT))
        video_writer.write(frame_video)

    # FINALIZA EVENTO
    if gravando:
        tempo = time.time() - inicio_gravacao
        tempo_sem_evento = time.time() - ultimo_evento

        if tempo >= MAX_VIDEO_DURATION or tempo_sem_evento >= JANELA_EVENTO:
            gravando = False
            video_writer.release()
            enviar_video_discord(nome_video, classe_detectada)
            print("📤 Evento finalizado")

    # VISUALIZAÇÃO
    cv2.imshow("IA Camera", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
