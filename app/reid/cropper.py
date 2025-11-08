"""
Cropper — Recorte ombro→pernas para Re-ID

Evita cabeça para melhorar discriminação entre pessoas.
Aplica pad refletivo quando bbox toca bordas do frame.

Estratégia comercial:
- Remove cabeça completamente (melhora discriminação)
- Usa ombros como teto do crop
- Estende até tornozelos quando disponível
- Fallback robusto para keypoints ausentes
- Expande lateralmente para capturar silhueta completa
- Pad refletivo em bordas (evita crops cortados)
"""

from __future__ import annotations
import numpy as np
import cv2
from typing import List, Tuple, Optional

# ============================================================
# ÍNDICES COCO KEYPOINTS (YOLOv11-pose)
# ============================================================
KP_L_SHOULDER = 5
KP_R_SHOULDER = 6
KP_L_HIP = 11
KP_R_HIP = 12
KP_L_KNEE = 13
KP_R_KNEE = 14
KP_L_ANKLE = 15
KP_R_ANKLE = 16

# ============================================================
# THRESHOLD MÍNIMO DE CONFIANÇA PARA ACEITAR KEYPOINT
# ============================================================
KP_THR = 0.20

# ============================================================
# PAD REFLETIVO
# ============================================================
PAD_BORDER_MARGIN = 5       # pixels - margem para detectar toque na borda
PAD_REFLECTIVE_SIZE = 20    # pixels - tamanho do pad refletivo


def _valid(kp: Tuple[float, float, float]) -> bool:
    """Verifica se keypoint tem confiança suficiente"""
    return kp[2] is not None and kp[2] >= KP_THR


def crop_body(frame: np.ndarray,
              bbox: Tuple[float, float, float, float],
              keypoints: List[Tuple[float, float, float]],
              had_pad: bool = False) -> Optional[np.ndarray]:
    """
    Recorta corpo (ombro→pernas) baseado em keypoints, removendo cabeça.
    Aplica pad refletivo se bbox toca borda.
    
    Estratégia comercial:
    - Remove cabeça completamente (melhora discriminação)
    - Usa ombros como teto do crop
    - Estende até tornozelos quando disponível
    - Fallback robusto para keypoints ausentes
    - Expande lateralmente para capturar silhueta completa
    - Pad refletivo evita crops cortados nas bordas

    Parâmetros
    ----------
    frame : np.ndarray
        Frame BGR.
    bbox : (x1, y1, x2, y2)
        Bounding box da pessoa (YOLO).
    keypoints : lista de (x, y, conf)
        17 keypoints COCO format.
    had_pad : bool
        Flag do detector indicando toque na borda

    Retorna
    -------
    crop : np.ndarray | None
        Recorte BGR (ombro→pernas) ou None se inválido.
    """

    h_img, w_img = frame.shape[:2]
    x1, y1, x2, y2 = bbox

    # ============================================================
    # 1) PAD REFLETIVO — evita cortes quando toca bordas
    # ============================================================
    frame_padded = frame
    offset_x = 0
    offset_y = 0

    if had_pad or _bbox_touches_border(bbox, (h_img, w_img)):
        frame_padded = cv2.copyMakeBorder(
            frame,
            PAD_REFLECTIVE_SIZE, PAD_REFLECTIVE_SIZE,
            PAD_REFLECTIVE_SIZE, PAD_REFLECTIVE_SIZE,
            cv2.BORDER_REFLECT
        )
        offset_x = PAD_REFLECTIVE_SIZE
        offset_y = PAD_REFLECTIVE_SIZE
        h_img = frame_padded.shape[0]
        w_img = frame_padded.shape[1]
        x1 += offset_x
        x2 += offset_x
        y1 += offset_y
        y2 += offset_y

    # ============================================================
    # 2) EXTRAÇÃO SEGURA DOS KEYPOINTS RELEVANTES
    # ============================================================
    kps = keypoints
    def _get_xy(idx):
        """Retorna coordenadas ajustadas de um keypoint válido"""
        if idx < len(kps) and _valid(kps[idx]):
            return (kps[idx][0] + offset_x, kps[idx][1] + offset_y)
        return None

    # Grupos anatômicos
    shoulders = [p for i in (KP_L_SHOULDER, KP_R_SHOULDER) if (p := _get_xy(i))]
    hips      = [p for i in (KP_L_HIP, KP_R_HIP) if (p := _get_xy(i))]
    knees     = [p for i in (KP_L_KNEE, KP_R_KNEE) if (p := _get_xy(i))]
    ankles    = [p for i in (KP_L_ANKLE, KP_R_ANKLE) if (p := _get_xy(i))]

    # ============================================================
    # 3) DETECÇÃO DE PRESENÇA CORPORAL
    # ============================================================
    has_shoulders  = len(shoulders) >= 1
    has_lower_body = len(hips) >= 1 or len(knees) >= 1 or len(ankles) >= 1

    # Sem região corporal visível (só cabeça/ombros) → aguardar
    if not has_lower_body:
        return None

    # ============================================================
    # 4) TETO DO CROP — ombros (remove cabeça completamente)
    # ============================================================
    if has_shoulders:
        y_top = min(y for _, y in shoulders)
    elif hips:
        # fallback: usa quadris se ombros ausentes
        y_top = min(y for _, y in hips) - 0.4 * (y2 - y1)
    else:
        y_top = y1 + 0.25 * (y2 - y1)

    # ============================================================
    # 5) CHÃO DO CROP — tornozelos, joelhos ou quadris
    # ============================================================
    if ankles:
        y_bottom = max(y for _, y in ankles)
    elif knees:
        y_bottom = max(y for _, y in knees) + 0.15 * (y2 - y1)
    elif hips:
        y_bottom = max(y for _, y in hips) + 0.40 * (y2 - y1)
    else:
        y_bottom = y2

    # ============================================================
    # 6) LIMITES HORIZONTAIS — união de ombros, quadris e pernas
    # ============================================================
    xs = []
    for group in (shoulders, hips, knees, ankles):
        xs.extend([x for x, _ in group])
    if xs:
        x_left, x_right = min(xs), max(xs)
    else:
        x_left, x_right = x1, x2

    # ============================================================
    # 7) EXPANSÃO LATERAL (±15%) — captura silhueta completa
    # ============================================================
    margin_x = 0.15 * (x_right - x_left)
    x_left -= margin_x
    x_right += margin_x

    # ============================================================
    # 8) CLAMP — mantém dentro dos limites válidos
    # ============================================================
    x_left   = max(0, int(x_left))
    x_right  = min(w_img, int(x_right))
    y_top    = max(0, int(y_top))
    y_bottom = min(h_img, int(y_bottom))

    # ============================================================
    # 9) VALIDAÇÃO DE TAMANHO MÍNIMO
    # ============================================================
    if x_right - x_left < 20 or y_bottom - y_top < 40:
        return None

    # ============================================================
    # 10) EXTRAÇÃO FINAL DO CROP
    # ============================================================
    crop = frame_padded[y_top:y_bottom, x_left:x_right]
    if crop.size == 0:
        return None

    # ============================================================
    # 11) VALIDA COBERTURA CORPORAL EFETIVA
    # ============================================================
    # Garante que o crop abrange ao menos 60% da bbox (corpo visível)
    height_crop = y_bottom - y_top
    height_bbox = y2 - y1
    if height_crop < 0.6 * height_bbox:
        return None

    return crop


def _bbox_touches_border(bbox: Tuple[float, float, float, float],
                         frame_shape: Tuple[int, int]) -> bool:
    """
    Detecta se bbox toca bordas do frame.
    
    Parâmetros
    ----------
    bbox : (x1, y1, x2, y2)
        Bounding box
    frame_shape : (height, width)
        Dimensões do frame
    
    Retorna
    -------
    touches : bool
        True se toca alguma borda
    """
    h, w = frame_shape
    x1, y1, x2, y2 = bbox
    
    touch_left = x1 < PAD_BORDER_MARGIN
    touch_top = y1 < PAD_BORDER_MARGIN
    touch_right = x2 > w - PAD_BORDER_MARGIN
    touch_bottom = y2 > h - PAD_BORDER_MARGIN
    
    return touch_left or touch_top or touch_right or touch_bottom


def compute_aspect_ratio(crop: np.ndarray) -> float:
    """
    Calcula aspect ratio do crop (w/h).
    
    Usado para metadata do buffer.
    
    Retorna
    -------
    aspect_ratio : float
    """
    if crop is None or crop.size == 0:
        return 0.0
    
    h, w = crop.shape[:2]
    return w / max(h, 1)


def compute_hsv_mean(crop: np.ndarray) -> np.ndarray:
    """
    Calcula média HSV do crop.
    
    Usado para metadata do buffer e validação de diversidade.
    
    Retorna
    -------
    hsv_mean : np.ndarray (3,)
        [H_mean, S_mean, V_mean]
    """
    if crop is None or crop.size == 0:
        return np.zeros(3, dtype=np.float32)
    
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hsv_mean = hsv.mean(axis=(0, 1))
    
    return hsv_mean.astype(np.float32)


def validate_crop_quality(crop: np.ndarray,
                          bbox: Tuple[float, float, float, float]) -> Tuple[bool, float]:
    """
    Valida qualidade do crop (cobertura em relação à bbox esperada).
    
    Parâmetros
    ----------
    crop : np.ndarray
        Crop extraído
    bbox : (x1, y1, x2, y2)
        Bounding box original
    
    Retorna
    -------
    valid : bool
        True se crop tem cobertura suficiente
    coverage : float
        Razão entre área do crop e área da bbox (0.0-1.0)
    """
    if crop is None or crop.size == 0:
        return False, 0.0
    
    crop_h, crop_w = crop.shape[:2]
    crop_area = crop_h * crop_w
    
    x1, y1, x2, y2 = bbox
    bbox_area = (x2 - x1) * (y2 - y1)
    
    if bbox_area <= 0:
        return False, 0.0
    
    coverage = crop_area / bbox_area
    coverage = min(coverage, 1.0)
    
    # Threshold: 95% de cobertura (90% se NEAR)
    # (escala não disponível aqui, usar 95% como padrão)
    valid = coverage >= 0.95
    
    return valid, coverage