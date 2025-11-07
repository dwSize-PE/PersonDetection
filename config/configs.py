# config/configs.py

import os

# === Caminhos ===
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

VIDEO_PATH = os.path.join(BASE_DIR, "data", "video.mp4")

# === Stream / Exibição ===
WINDOW_NAME = "Video Stream"
TARGET_FPS = 30                    # apenas para manter exibição fluida
FULLSCREEN = False                 # pode ativar depois

# === Buffer de Frames ===
USE_LATEST_FRAME = True           # TRUE = sempre último frame (recomendado)
FRAME_GRAB_DELAY = 1              # ms (evitar uso excessivo de CPU)

# === Futuro: YOLO ===
YOLO_MODEL_PATH = os.path.join(BASE_DIR, "models", "yolov8n-face.onnx")
GPU_ID = -1                       # -1 força CPU, 0 tenta GPU
