"""
Embedder Thread — Pipeline assíncrono de Re-ID

Orquestra o fluxo completo:
1. Consome tracks ativos do ByteTracker
2. Crop ombro→pernas via keypoints
3. OSNet embedding extraction
4. ReIdentifier para gerenciar identidades
5. Detecta tracks perdidos e consolida no banco

Roda em thread separada para não bloquear FPS do vídeo.
"""

from __future__ import annotations
import threading
import time
from typing import Dict, Set, Optional, Any, List

import numpy as np
import torch

from .cropper import crop_body
from app.osnet.osnet_model import OsNetEmbedder
from .gallery import Gallery
from .reidentifier import ReIdentifier


class ReIDEmbedderThread:
    """
    Thread assíncrona de Re-ID.
    
    Consome tracks ativos + detecções (bbox + keypoints) do pipeline principal.
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
            {'tracks': [{'track_id', 'bbox', 'keypoints'}, ...]}
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

        # Controle de tracks perdidos
        self._previous_track_ids: Set[int] = set()

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

            # 1) Captura frame compartilhado
            with self.lock_frame:
                frame: Optional[np.ndarray] = self.shared_frame.get('frame')
                frame_index: int = self.shared_frame.get('frame_index', 0)

            if frame is None:
                continue

            # 2) Captura tracks compartilhados
            with self.lock_tracks:
                tracks: List[dict] = self.shared_tracks.get('tracks', [])

            if not tracks:
                # Sem tracks ativos — detecta se todos foram perdidos
                self._detect_lost_tracks(set(), frame_index)
                continue

            # 3) Detecta tracks perdidos (comparação com frame anterior)
            current_track_ids = {t['track_id'] for t in tracks}
            self._detect_lost_tracks(current_track_ids, frame_index)

            # 4) Processa cada track ativo
            for track in tracks:
                track_id = track.get('track_id')
                bbox = track.get('bbox')
                keypoints = track.get('keypoints')

                if track_id is None or bbox is None or keypoints is None:
                    continue

                # Crop ombro→pernas
                crop = crop_body(frame, bbox, keypoints)
                if crop is None:
                    continue

                # OSNet embedding
                emb = self.osnet.extract_one(crop)
                if emb is None:
                    continue

                # Estratégia Re-ID:
                # - Se track é novo: tenta Re-ID após min_samples
                # - Se track já promovido: apenas acumula embeddings
                if self.reid.is_promoted(track_id):
                    # Track já tem id_global — apenas atualiza Gallery
                    self.reid.on_track_active(track_id, emb, frame_index)
                else:
                    # Track novo — tenta Re-ID
                    id_global = self.reid.on_new_track(track_id, emb, frame_index)
                    if id_global is not None:
                        # Re-ID bem-sucedido
                        pass  # log já feito pelo ReIdentifier

            # Atualiza conjunto de tracks para próxima iteração
            self._previous_track_ids = current_track_ids

    # =========================================================================
    # DETECÇÃO DE TRACKS PERDIDOS
    # =========================================================================

    def _detect_lost_tracks(self, current_track_ids: Set[int], frame_index: int):
        """
        Detecta tracks que sumiram (estavam ativos e agora não estão).
        Consolida embeddings da Gallery → IdentityBank.
        
        Parâmetros
        ----------
        current_track_ids : set[int]
            IDs dos tracks atualmente ativos
        frame_index : int
            Frame atual
        """
        lost_track_ids = self._previous_track_ids - current_track_ids

        for track_id in lost_track_ids:
            id_global = self.reid.on_track_lost(track_id, frame_index)
            
            if id_global is not None:
                print(f"[ReID] Track {track_id} PERDIDO → consolidado como RID {id_global}")
            else:
                print(f"[ReID] Track {track_id} perdido (sem embeddings suficientes)")

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