"""
Embedder Thread — Pipeline assíncrono de Re-ID com gate de qualidade

Responsabilidades:
- Consumir tracks ativos do ByteTracker (ACTIVE + PENDING)
- Gate de qualidade multi-critério (conf, cobertura, blur, escala)
- Crop ombro→pernas via keypoints
- OSNet embedding extraction
- Buffer temporal com stride e metadata
- Detectar REFIND (volta de TEMP_LOST) e skip Re-ID durante reanexação
- Verificar consistência pós-REFIND (detecta IDs trocados em multi-alvo denso)
- Propagar bbox/density/frame_height para Anti-Teleport e Lock adaptativo

Roda em thread separada para não bloquear FPS do vídeo.
"""

from __future__ import annotations
import threading
import time
import cv2
import numpy as np
from typing import Dict, Set, Optional, Any, List
from collections import deque

import torch
import torch.nn.functional as F

from .cropper import crop_body
from app.osnet.osnet_model import OsNetEmbedder
from .gallery import Gallery
from .reidentifier import ReIdentifier

# ============================================================
# GATE DE QUALIDADE - THRESHOLDS
# ============================================================
GATE_CONF_MIN = 0.50        # confiança YOLO mínima
GATE_COVERAGE_MIN = 0.55    # cobertura do crop (0.90 se NEAR)
GATE_BLUR_MIN = 20.0        # Laplacian variance mínima
GATE_SCALE_REJECT = "DESC"  # rejeita escala DESCARTÁVEL

# ============================================================
# BLUR Z-SCORE ROLLING WINDOW
# ============================================================
BLUR_WINDOW_SIZE = 30       # amostras para calcular μ e σ


class ReIDEmbedderThread:
    """
    Thread assíncrona de Re-ID com gate de qualidade.
    
    Consome tracks ativos + detecções (bbox + keypoints + metadata) do pipeline principal.
    Não bloqueia FPS — processa de forma assíncrona.
    """

    def __init__(self,
                 gallery: Gallery,
                 osnet: OsNetEmbedder,
                 reid: ReIdentifier,
                 lock_frame: threading.Lock,
                 shared_frame: Dict[str, Any],
                 lock_tracks: threading.Lock,
                 shared_tracks: Dict[str, Any],
                 sleep_ms: int = 5):
        """
        Parâmetros
        ----------
        gallery : Gallery
            Galeria de embeddings
        osnet : OsNetEmbedder
            Modelo OSNet para embeddings
        reid : ReIdentifier
            Motor de Re-ID
        lock_frame : threading.Lock
            Lock para acessar frame compartilhado
        shared_frame : dict
            {'frame': np.ndarray, 'frame_index': int}
        lock_tracks : threading.Lock
            Lock para acessar tracks compartilhados
        shared_tracks : dict
            {'tracks': [...], 'temp_lost': [...], 'density': float, 'frame_height': int}
        sleep_ms : int
            Tempo de sleep entre iterações (ms)
        """
        self.gallery = gallery
        self.osnet = osnet
        self.reid = reid

        self.lock_frame = lock_frame
        self.lock_tracks = lock_tracks
        self.shared_frame = shared_frame
        self.shared_tracks = shared_tracks

        self.running = False
        self.thread = None
        self.sleep_ms = sleep_ms

        # ============================================================
        # CONTROLE DE REFIND (volta de TEMP_LOST)
        # ============================================================
        self._tracks_temp_lost_last_frame: Set[int] = set()
        
        # ============================================================
        # BLUR ROLLING WINDOW (para z-score)
        # ============================================================
        self._blur_window = deque(maxlen=BLUR_WINDOW_SIZE)

    # =========================================================================
    # CONTROLE DA THREAD
    # =========================================================================

    def start(self):
        """Inicia thread assíncrona"""
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        print("[ReID] Thread iniciada.")

    def stop(self):
        """Para thread assíncrona"""
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)
            self.thread = None
        print("[ReID] Thread finalizada.")

    # =========================================================================
    # LOOP PRINCIPAL
    # =========================================================================

    def _loop(self):
        """Loop principal da thread — processa tracks assincronamente"""
        while self.running:
            time.sleep(self.sleep_ms / 1000.0)

            # ============================================================
            # 1) CAPTURA FRAME COMPARTILHADO
            # ============================================================
            with self.lock_frame:
                frame: Optional[np.ndarray] = self.shared_frame.get('frame')
                frame_index: int = self.shared_frame.get('frame_index', 0)

            if frame is None:
                continue

            frame_h, frame_w = frame.shape[:2]

            # ============================================================
            # 2) CAPTURA TRACKS + TEMP_LOST COMPARTILHADOS
            # ============================================================
            with self.lock_tracks:
                tracks: List[dict] = self.shared_tracks.get('tracks', [])
                temp_lost_tracks: List[dict] = self.shared_tracks.get('temp_lost', [])
                density: float = self.shared_tracks.get('density', 0.0)
                frame_height: int = self.shared_tracks.get('frame_height', 1080)

            if not tracks:
                # Sem tracks ativos — atualiza cache e continua
                self._tracks_temp_lost_last_frame = set()
                continue

            # ============================================================
            # 3) DETECTA REFIND (voltou de TEMP_LOST)
            # ============================================================
            current_temp_lost = {t['track_id'] for t in temp_lost_tracks}
            current_active = {t['track_id'] for t in tracks}
            
            # IDs que voltaram de TEMP_LOST → REFIND (ByteTracker reatou)
            refind_ids = self._tracks_temp_lost_last_frame & current_active

            # ============================================================
            # 4) PROCESSA CADA TRACK ATIVO
            # ============================================================
            for track in tracks:
                track_id = track.get('track_id')
                bbox = track.get('bbox')
                keypoints = track.get('keypoints')
                scale = track.get('scale', 'UNKNOWN')
                conf = track.get('score', 0.0)
                had_pad = track.get('had_pad', False)

                if track_id is None or bbox is None or keypoints is None:
                    continue

                # ============================================================
                # GATE REFIND: skip Re-ID se voltou de TEMP_LOST
                # ============================================================
                if track_id in refind_ids:
                    # ============================================================
                    # VERIFICAÇÃO DE CONSISTÊNCIA (multi-alvo denso)
                    # ============================================================
                    # ByteTracker pode reatar IDs trocados durante oclusão mútua
                    # Detecta via similaridade: se sim < 0.40, houve troca
                    if self.reid.is_promoted(track_id):
                        pid = self.reid.get_global_id(track_id)
                        
                        # Coleta embedding atual para comparar
                        crop = crop_body(frame, bbox, keypoints, had_pad=had_pad)
                        if crop is not None:
                            emb = self.osnet.extract_one(crop)
                            identity = self.reid.bank.get(pid)
                            
                            if emb is not None and identity is not None:
                                sim = float(torch.matmul(
                                    F.normalize(emb, p=2, dim=0).view(1, -1),
                                    identity.embedding.view(1, -1).T
                                ).item())
                                
                                # INCONSISTÊNCIA GRAVE (troca de ID detectada)
                                if sim < 0.40:
                                    print(f"[EMB_REFIND] f={frame_index} tid=T{track_id} pid=P{pid:02d} sim={sim:.2f} INCONSISTENT → force_reid")
                                    # Não skip — permite Re-ID para corrigir
                                else:
                                    # Consistente — skip on_new_track (refind legítimo)
                                    print(f"[EMB_REFIND] f={frame_index} tid=T{track_id} pid=P{pid:02d} sim={sim:.2f} skip=YES (refind)")
                                    continue
                    else:
                        # Não promovido ainda — skip on_new_track (refind)
                        print(f"[EMB_REFIND] f={frame_index} tid=T{track_id} skip=YES (refind)")
                        continue

                # ============================================================
                # GATE DE QUALIDADE (pré-embedding)
                # ============================================================
                gate_pass, gate_reason = self._quality_gate(
                    conf=conf,
                    scale=scale,
                    bbox=bbox,
                    frame_shape=(frame_h, frame_w)
                )

                if not gate_pass:
                    # Log: gate rejeitou
                    print(f"[EMB_GATE] f={frame_index} tid=T{track_id} pass=NO reason={gate_reason}")
                    continue

                # ============================================================
                # CROP OMBRO→PERNAS
                # ============================================================
                crop = crop_body(frame, bbox, keypoints, had_pad=had_pad)
                if crop is None:
                    print(f"[EMB_GATE] f={frame_index} tid=T{track_id} pass=NO reason=crop_failed")
                    continue

                # ============================================================
                # VALIDAÇÃO: cobertura do crop
                # ============================================================
                coverage = self._compute_coverage(bbox, (frame_h, frame_w))
                coverage_min = 0.90 if scale == "NEAR" else GATE_COVERAGE_MIN
                
                if coverage < coverage_min:
                    print(f"[EMB_GATE] f={frame_index} tid=T{track_id} pass=NO reason=low_coverage coverage={coverage:.2f}")
                    continue

                # ============================================================
                # VALIDAÇÃO: blur (Laplacian)
                # ============================================================
                blur_score = self._compute_blur(crop)
                
                if blur_score < GATE_BLUR_MIN:
                    print(f"[EMB_GATE] f={frame_index} tid=T{track_id} pass=NO reason=low_blur blur={blur_score:.1f}")
                    continue

                # ============================================================
                # BLUR Z-SCORE (qualidade relativa ao frame)
                # ============================================================
                blur_z = self._compute_blur_z(blur_score)

                # ============================================================
                # METADATA: aspect ratio e HSV
                # ============================================================
                aspect_ratio = crop.shape[1] / max(crop.shape[0], 1)
                hsv_mean = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).mean(axis=(0, 1))

                # ============================================================
                # LOG: gate passou
                # ============================================================
                print(f"[EMB_GATE] f={frame_index} tid=T{track_id} pass=YES conf={conf:.2f} blur={blur_score:.1f} blur_z={blur_z:+.1f} scale={scale}")

                # ============================================================
                # OSNET EMBEDDING EXTRACTION
                # ============================================================
                emb = self.osnet.extract_one(crop)
                if emb is None:
                    continue

                # ============================================================
                # BUFFER: adiciona com stride temporal
                # ============================================================
                added = self.gallery.add(
                    track_id=track_id,
                    embedding=emb,
                    scale=scale,
                    blur_z=blur_z,
                    aspect_ratio=aspect_ratio,
                    hsv_mean=hsv_mean,
                    frame_index=frame_index
                )

                if added:
                    size = self.gallery.count(track_id)
                    print(f"[BUF_ADD] f={frame_index} tid=T{track_id} size={size}/10 scale={scale} blur_z={blur_z:+.1f} stride_ok=YES")
                else:
                    # Rejeitado por stride
                    pass

                # ============================================================
                # ESTRATÉGIA RE-ID:
                # - Se track já promovido: apenas acumula embeddings
                # - Se track novo: tenta Re-ID após min_samples
                # ============================================================
                if self.reid.is_promoted(track_id):
                    # Track já tem id_global — apenas atualiza Gallery
                    self.reid.on_track_active(track_id, emb, frame_index)
                else:
                    # Track novo — tenta Re-ID (propaga bbox/density/frame_height)
                    id_global = self.reid.on_new_track(
                        track_id=track_id,
                        embedding=emb,
                        frame_index=frame_index,
                        bbox=bbox,
                        density=density,
                        frame_height=frame_height
                    )
                    if id_global is not None:
                        # Re-ID bem-sucedido (log já feito pelo ReIdentifier)
                        pass

            # ============================================================
            # ATUALIZA CACHE TEMP_LOST PARA PRÓXIMO FRAME
            # ============================================================
            self._tracks_temp_lost_last_frame = current_temp_lost

    # =========================================================================
    # GATE DE QUALIDADE
    # =========================================================================

    def _quality_gate(self, 
                      conf: float,
                      scale: str,
                      bbox: tuple,
                      frame_shape: tuple) -> tuple:
        """
        Gate de qualidade multi-critério.
        
        Critérios:
        1. conf_yolo ≥ 0.50
        2. scale ≠ DESC
        3. bbox dentro do frame (não cortada demais)
        
        Retorna
        -------
        pass : bool
            True se passou no gate
        reason : str
            Motivo de rejeição (se fail)
        """
        # ============================================================
        # CRITÉRIO 1: confiança YOLO
        # ============================================================
        if conf < GATE_CONF_MIN:
            return False, f"low_conf_{conf:.2f}"
        
        # ============================================================
        # CRITÉRIO 2: escala válida
        # ============================================================
        if scale == GATE_SCALE_REJECT:
            return False, "scale_DESC"
        
        # ============================================================
        # CRITÉRIO 3: bbox não cortada demais (implícito na cobertura)
        # (validado após crop)
        # ============================================================
        
        return True, "ok"

    # embedder.py
    def _compute_coverage(self, bbox: tuple, frame_shape: tuple) -> float:
        """
        Cobertura = fração da ALTURA da bbox que está dentro do frame (0..1).
        Não usa área do crop, pois o crop ombro→pernas é sub-região da bbox.
        """
        H, W = frame_shape[:2]
        x1, y1, x2, y2 = bbox

        # Altura original da bbox
        bbox_h = max(1e-6, y2 - y1)

        # Altura visível (bbox clampada ao frame)
        vy1 = max(0.0, min(float(y1), float(H)))
        vy2 = max(0.0, min(float(y2), float(H)))
        visible_h = max(0.0, vy2 - vy1)

        cov = visible_h / bbox_h
        # clamp extra de segurança
        return 0.0 if bbox_h <= 0 else max(0.0, min(1.0, cov))

    def _compute_blur(self, crop: np.ndarray) -> float:
        """
        Calcula blur score usando Laplacian variance.
        
        Retorna
        -------
        blur_score : float
            Variância do Laplacian (maior = mais nítido)
        """
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()
        return float(variance)

    def _compute_blur_z(self, blur_score: float) -> float:
        """
        Calcula z-score do blur (qualidade relativa ao frame).
        
        Usa rolling window de 30 amostras para μ e σ.
        
        Retorna
        -------
        blur_z : float
            Z-score (positivo = acima da média, negativo = abaixo)
        """
        # ============================================================
        # ADICIONA AO ROLLING WINDOW
        # ============================================================
        self._blur_window.append(blur_score)
        
        # ============================================================
        # CALCULA Z-SCORE
        # ============================================================
        if len(self._blur_window) < 2:
            return 0.0  # insuficiente para calcular
        
        blur_mean = np.mean(self._blur_window)
        blur_std = np.std(self._blur_window)
        
        if blur_std < 1e-6:
            return 0.0  # sem variação
        
        blur_z = (blur_score - blur_mean) / blur_std
        return float(blur_z)

    # =========================================================================
    # QUERIES (para uso externo)
    # =========================================================================

    def get_global_id(self, track_id: int) -> Optional[int]:
        """Retorna id_global de um track (se promovido)"""
        return self.reid.get_global_id(track_id)

    def get_color(self, track_id: int) -> tuple:
        """Retorna cor BGR para um track"""
        return self.reid.get_color(track_id)

    def is_promoted(self, track_id: int) -> bool:
        """Verifica se track já tem id_global"""
        return self.reid.is_promoted(track_id)