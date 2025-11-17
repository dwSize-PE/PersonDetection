"""
Detector Thread — YOLO assíncrono com classificação de escala

Responsabilidades:
- Detectar pessoas via YOLOv11-pose (thread isolada, não bloqueia FPS)
- Validar detecções (cabeça OU corpo visível com confiança)
- Classificar escala (NEAR/MID/FAR/DESC) com histerese temporal
- Detectar toque nas bordas (sinaliza pad refletivo para cropper)
- Quantizar bbox_id (grid 8px) para estabilizar cache de escala
- Compartilhar detecções via locks thread-safe

Formato de saída: [x1, y1, x2, y2, conf, cls, keypoints, scale, had_pad]
"""

import threading
import time
import cv2
import queue
import numpy as np
from ultralytics import YOLO

# ============================================================
# FILAS DE COMUNICAÇÃO (Thread-Safe)
# ============================================================
# input_queue: (frame, frame_index)
# result_queue: (detections, frame_index)
input_queue = queue.Queue(maxsize=2)
result_queue = queue.Queue(maxsize=2)

latest_frame = None
latest_detections = []
lock_frame = threading.Lock()
lock_det = threading.Lock()

running = True

# ============================================================
# THRESHOLDS DE ESCALA (baseado em altura bbox / altura frame)
# ============================================================
SCALE_NEAR_MIN = 0.45
SCALE_MID_MIN = 0.20
SCALE_FAR_MIN = 0.10
SCALE_HYST = 0.005  # histerese para evitar flip rápido

# ============================================================
# THRESHOLDS BYTETRACK (det_high / det_low)
# ============================================================
DET_HIGH_THR = 0.55
DET_LOW_THR = 0.20

# ============================================================
# CACHE DE ESCALA (para histerese temporal)
# ============================================================
_scale_cache = {}  # bbox_id -> escala anterior

def submit_frame(frame, frame_index):
    """
    Envia frame para fila de processamento (não bloqueante se cheio).
    Substitui o antigo set_frame.
    """
    if not running:
        return

    # Tenta colocar na fila; se cheia, descarta o frame antigo para manter realtime
    try:
        input_queue.put_nowait((frame.copy(), frame_index))
    except queue.Full:
        # Opcional: Dropar frame antigo e colocar o novo, ou apenas ignorar o novo
        pass


def reset_detector_state():
    """
    Reseta estado global do detector entre loops de vídeo.
    DEVE ser chamado em stream.py quando novo loop começa (no start_stream).
    """
    global _scale_cache
    _scale_cache.clear()
    print("[DETECTOR_RESET] _scale_cache limpo")


def get_result():
    """
    Recupera resultado do detector (se disponível).
    
    Retorna
    -------
    tuple ou None
        (detections, frame_index)
    """
    try:
        return result_queue.get_nowait()
    except queue.Empty:
        return None

def person_is_valid(keypoints):
    """
    Retorna True se a pessoa for válida (tem corpo).
    Critério: cabeça OU corpo detectados com confiança.
    """
    head_ids = [0, 1, 2, 3, 4]
    body_ids = [5, 6, 11, 12]

    head_ok = any(keypoints[i][2] > 0.2 for i in head_ids)
    body_ok = any(keypoints[i][2] > 0.2 for i in body_ids)

    return head_ok or body_ok


def classify_scale(bbox_height, frame_height, prev_scale=None):
    """
    Classifica escala da detecção: NEAR / MID / FAR / DESC
    Aplica histerese se prev_scale fornecido.
    
    Parâmetros
    ----------
    bbox_height : float
        Altura da bbox
    frame_height : int
        Altura do frame
    prev_scale : str | None
        Escala anterior (para histerese)
    
    Retorna
    -------
    scale : str
        "NEAR", "MID", "FAR", "DESC"
    """
    ratio = bbox_height / frame_height
    
    # ============================================================
    # HISTERESE: se próximo de threshold e tinha escala anterior
    # ============================================================
    if prev_scale is not None:
        # NEAR ↔ MID boundary
        if prev_scale == "NEAR" and ratio > (SCALE_NEAR_MIN - SCALE_HYST):
            return "NEAR"
        if prev_scale == "MID" and ratio > (SCALE_NEAR_MIN + SCALE_HYST):
            return "NEAR"
        
        # MID ↔ FAR boundary
        if prev_scale == "MID" and ratio > (SCALE_MID_MIN - SCALE_HYST):
            return "MID"
        if prev_scale == "FAR" and ratio > (SCALE_MID_MIN + SCALE_HYST):
            return "MID"
        
        # FAR ↔ DESC boundary
        if prev_scale == "FAR" and ratio > (SCALE_FAR_MIN - SCALE_HYST):
            return "FAR"
        if prev_scale == "DESC" and ratio > (SCALE_FAR_MIN + SCALE_HYST):
            return "FAR"
    
    # ============================================================
    # CLASSIFICAÇÃO DIRETA (sem histerese ou primeira vez)
    # ============================================================
    if ratio > SCALE_NEAR_MIN:
        return "NEAR"
    elif ratio > SCALE_MID_MIN:
        return "MID"
    elif ratio > SCALE_FAR_MIN:
        return "FAR"
    else:
        return "DESC"


def apply_reflective_pad(frame, bbox):
    """
    Aplica pad refletivo se bbox toca bordas do frame.
    Retorna bbox corrigida e flag indicando se houve pad.
    
    Parâmetros
    ----------
    frame : np.ndarray
        Frame BGR
    bbox : (x1, y1, x2, y2)
        Bounding box original
    
    Retorna
    -------
    bbox_padded : (x1, y1, x2, y2)
        Bbox ajustada (mesmas coords se sem pad)
    had_pad : bool
        True se aplicou pad refletivo
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    
    # ============================================================
    # DETECTA TOQUE NAS BORDAS (margem 5px)
    # ============================================================
    touch_left = x1 < 5
    touch_top = y1 < 5
    touch_right = x2 > w - 5
    touch_bottom = y2 > h - 5
    
    if not (touch_left or touch_top or touch_right or touch_bottom):
        return bbox, False
    
    # ============================================================
    # PAD REFLETIVO (expand bbox dentro dos limites válidos)
    # ============================================================
    # Estratégia: não alterar bbox, apenas sinalizar que crop
    # deve usar cv2.BORDER_REFLECT ao extrair
    # (implementação real do pad será no cropper)
    
    return bbox, True


def _make_bbox_id(x1, y1, x2, y2):
    """
    Gera ID único para bbox (para cache de escala).
    Quantização em grid de 8px para estabilizar escala.
    """
    g = 8
    return f"{int(x1//g)*g}_{int(y1//g)*g}_{int(x2//g)*g}_{int(y2//g)*g}"


def detector_thread():
    global latest_frame, latest_detections, running, _scale_cache

    model = YOLO("models/yolov11n-pose.pt")
    
    frame_count = 0

    while running:
        try:
            # Bloqueia esperando frame (timeout permite checar running)
            frame, frame_index = input_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        frame_h, frame_w = frame.shape[:2]
        
        # ============================================================
        # YOLO DETECÇÃO
        # ============================================================
        t0_yolo = time.perf_counter()
        results = model(frame, verbose=False)
        t1_yolo = time.perf_counter()
        yolo_ms = (t1_yolo - t0_yolo) * 1000.0
        dets = []
        
        scales_count = {"NEAR": 0, "MID": 0, "FAR": 0, "DESC": 0}
        n_high = 0

        for r in results:
            if r.boxes is None or r.keypoints is None:
                continue
                
            for box, kp in zip(r.boxes, r.keypoints):
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls = int(box.cls[0])

                # ============================================================
                # FILTRO: só pessoa (cls=0)
                # ============================================================
                if cls != 0:
                    continue

                # ============================================================
                # KEYPOINTS EXTRACTION
                # ============================================================
                keypoints = []
                if kp.xy is not None and kp.conf is not None:
                    for (x, y), c in zip(kp.xy[0].tolist(), kp.conf[0].tolist()):
                        keypoints.append((x, y, c))
                else:
                    # fallback: 17 keypoints zerados
                    keypoints = [(0.0, 0.0, 0.0)] * 17

                # ============================================================
                # VALIDAÇÃO: pessoa real (não parcial demais)
                # ============================================================
                if not person_is_valid(keypoints):
                    continue

                # ============================================================
                # CLASSIFICAÇÃO DE ESCALA (com histerese)
                # ============================================================
                bbox_h = y2 - y1
                bbox_id = _make_bbox_id(x1, y1, x2, y2)
                prev_scale = _scale_cache.get(bbox_id)
                
                scale = classify_scale(bbox_h, frame_h, prev_scale)
                _scale_cache[bbox_id] = scale
                scales_count[scale] += 1

                # ============================================================
                # PAD REFLETIVO (se toca borda)
                # ============================================================
                bbox_original = (x1, y1, x2, y2)
                bbox_final, had_pad = apply_reflective_pad(frame, bbox_original)

                # ============================================================
                # CONTAGEM HIGH-CONFIDENCE (ByteTrack)
                # ============================================================
                if conf >= DET_HIGH_THR:
                    n_high += 1

                # ============================================================
                # FORMATO SAÍDA: [x1, y1, x2, y2, conf, cls, keypoints, scale, had_pad]
                # ============================================================
                dets.append([
                    bbox_final[0], bbox_final[1], bbox_final[2], bbox_final[3],
                    conf, cls, keypoints, scale, had_pad
                ])

        # ============================================================
        # ENVIA RESULTADO (Detecções + Frame Index Original)
        # ============================================================
        if running:
            try:
                result_queue.put((dets, frame_index), timeout=0.1)
            except queue.Full:
                pass

        # ============================================================
        # LOG: [DET] resumo do frame
        # ============================================================
        if frame_count % 30 == 0 or len(dets) > 0:  # log a cada 30 frames ou quando há detecções
            n_total = len(dets)
            print(f"[DET] f={frame_count} n={n_total} | high={n_high} | scales: N={scales_count['NEAR']} M={scales_count['MID']} F={scales_count['FAR']} D={scales_count['DESC']}")
            print(f"[YOLO_TIME] f={frame_count} time={yolo_ms:.1f}ms")
        
        frame_count += 1
        
        # ============================================================
        # LIMPEZA DE CACHE (evita crescimento infinito)
        # ============================================================
        if frame_count % 300 == 0:  # a cada 10s @ 30fps
            _scale_cache.clear()

        time.sleep(0.001)