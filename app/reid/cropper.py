"""
Cropper — Recorte ombro→pernas para Re-ID
Evita cabeça para melhorar discriminação entre pessoas.
"""

from __future__ import annotations
import numpy as np
from typing import List, Tuple, Optional

# Índices COCO keypoints (YOLOv11-pose)
KP_L_SHOULDER = 5
KP_R_SHOULDER = 6
KP_L_HIP = 11
KP_R_HIP = 12
KP_L_KNEE = 13
KP_R_KNEE = 14
KP_L_ANKLE = 15
KP_R_ANKLE = 16

# Threshold mínimo de confiança para aceitar keypoint
KP_THR = 0.20


def _valid(kp: Tuple[float, float, float]) -> bool:
    """Verifica se keypoint tem confiança suficiente"""
    return kp[2] is not None and kp[2] >= KP_THR


def crop_body(frame: np.ndarray,
              bbox: Tuple[float, float, float, float],
              keypoints: List[Tuple[float, float, float]]) -> Optional[np.ndarray]:
    """
    Recorta corpo (ombro→pernas) baseado em keypoints, removendo cabeça.
    
    Estratégia comercial:
    - Remove cabeça completamente (melhora discriminação)
    - Usa ombros como teto do crop
    - Estende até tornozelos quando disponível
    - Fallback robusto para keypoints ausentes
    - Expande lateralmente para capturar silhueta completa

    Parâmetros
    ----------
    frame : np.ndarray
        Frame BGR.
    bbox : (x1, y1, x2, y2)
        Bounding box da pessoa (YOLO).
    keypoints : lista de (x, y, conf)
        17 keypoints COCO format.

    Retorna
    -------
    crop : np.ndarray | None
        Recorte BGR (ombro→pernas) ou None se inválido.
    """

    h_img, w_img = frame.shape[:2]
    x1, y1, x2, y2 = bbox

    # -------------------------------
    # 1) Keypoints relevantes
    # -------------------------------
    kps = keypoints

    # Ombros (teto do crop)
    shoulders = []
    for idx in (KP_L_SHOULDER, KP_R_SHOULDER):
        if idx < len(kps) and _valid(kps[idx]):
            shoulders.append((kps[idx][0], kps[idx][1]))
    
    # Quadris (meio do corpo)
    hips = []
    for idx in (KP_L_HIP, KP_R_HIP):
        if idx < len(kps) and _valid(kps[idx]):
            hips.append((kps[idx][0], kps[idx][1]))

    # Joelhos/tornozelos (chão do crop)
    legs = []
    for idx in (KP_L_KNEE, KP_R_KNEE, KP_L_ANKLE, KP_R_ANKLE):
        if idx < len(kps) and _valid(kps[idx]):
            legs.append((kps[idx][0], kps[idx][1]))

    # -------------------------------
    # 2) Limites verticais (y_min → y_max)
    # -------------------------------

    # Teto: ombros (remove cabeça completamente)
    if shoulders:
        y_top = min(y for _, y in shoulders)
    else:
        # Fallback: 25% abaixo do topo do bbox (estimativa de ombros)
        y_top = y1 + 0.25 * (y2 - y1)

    # Chão: tornozelos/joelhos
    if legs:
        y_bottom = max(y for _, y in legs)
    elif hips:
        # Fallback: estima pernas a partir dos quadris
        y_bottom = max(y for _, y in hips) + 0.4 * (y2 - y1)
    else:
        # Fallback: bbox completo
        y_bottom = y2

    # -------------------------------
    # 3) Limites horizontais
    # -------------------------------

    xs = []
    if shoulders:
        xs.extend([x for x, _ in shoulders])
    if hips:
        xs.extend([x for x, _ in hips])
    if legs:
        xs.extend([x for x, _ in legs])

    if xs:
        x_left = min(xs)
        x_right = max(xs)
    else:
        x_left, x_right = x1, x2

    # Expande lateralmente (captura silhueta completa)
    margin_x = 0.15 * (x_right - x_left)
    x_left -= margin_x
    x_right += margin_x

    # -------------------------------
    # 4) Sanitiza limites (dentro da imagem)
    # -------------------------------
    x_left = max(0, int(x_left))
    x_right = min(w_img, int(x_right))
    y_top = max(0, int(y_top))
    y_bottom = min(h_img, int(y_bottom))

    # -------------------------------
    # 5) Valida tamanho mínimo
    # -------------------------------
    if x_right - x_left < 20 or y_bottom - y_top < 40:
        return None

    # -------------------------------
    # 6) Extrai recorte
    # -------------------------------
    crop = frame[y_top:y_bottom, x_left:x_right]
    if crop.size == 0:
        return None

    return crop