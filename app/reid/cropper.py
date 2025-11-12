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
              had_pad: bool = False,
              show_debug: bool = True) -> Optional[np.ndarray]:
    """
    Versão de teste — ignora completamente keypoints e lógica corporal.
    Apenas retorna e exibe a bbox original da pessoa (sem recorte inteligente).
    """

    # Desempacota a bbox (float → int)
    x1, y1, x2, y2 = map(int, bbox)

    # Garante limites válidos dentro do frame
    h_img, w_img = frame.shape[:2]
    x1 = max(0, min(x1, w_img - 1))
    x2 = max(0, min(x2, w_img - 1))
    y1 = max(0, min(y1, h_img - 1))
    y2 = max(0, min(y2, h_img - 1))

    # Cria o crop básico (para caso queira salvar/testar)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    # Retorna apenas o crop puro da bbox (sem keypoints, sem expansão)
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