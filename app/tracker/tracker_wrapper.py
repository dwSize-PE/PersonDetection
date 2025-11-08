"""
ByteTrack Wrapper — Integração YOLO → ByteTracker original + Re-ID

Responsabilidades:
- Converter detecções YOLO para formato ByteTracker
- Preservar keypoints + metadata (scale, had_pad) para cada track_id
- Exportar estados granulares: ACTIVE, PENDING, TEMP_LOST, LOST
- Fornecer interface limpa para sistema Re-ID
"""

from typing import List, Dict, Any, Tuple
import numpy as np

from .bytetrack.byte_tracker import BYTETracker, STrack

class Args:
    """Argumentos do ByteTracker"""
    def __init__(self, track_thresh=0.5, track_buffer=30, match_thresh=0.8, mot20=False):
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.mot20 = mot20


class ByteTrackWrapper:
    """
    Wrapper comercial do ByteTracker original.
    
    Input (detecções YOLO):
        Lista de [x1, y1, x2, y2, conf, cls, keypoints, scale, had_pad]
    
    Output:
        Dict com tracks separados por estado {
            'active': [...],      # ACTIVE: tracks confirmados e visíveis
            'pending': [...],     # PENDING: tracks novos (não confirmados)
            'temp_lost': [...],   # TEMP_LOST: perdidos recentemente (ainda no buffer)
            'lost': [...]         # LOST: removidos definitivamente
        }
    """

    def __init__(self, 
                 track_thresh: float = 0.5,
                 track_buffer: int = 30,
                 match_thresh: float = 0.8,
                 frame_rate: int = 30):
        
        args = Args(track_thresh=track_thresh, 
                   track_buffer=track_buffer,
                   match_thresh=match_thresh,
                   mot20=False)
        
        self.tracker = BYTETracker(args, frame_rate=frame_rate)
        self.frame_h = 0
        self.frame_w = 0
        self.frame_id = 0
        
        # ============================================================
        # CACHE: tracks do frame anterior (para detectar LOST)
        # ============================================================
        self._prev_active_ids = set()
        self._prev_lost_ids = set()

    def update(self, 
               detections: List[List[Any]], 
               frame_shape: Tuple[int, int]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Atualiza tracker com novas detecções.
        
        Parâmetros
        ----------
        detections : lista de [x1, y1, x2, y2, conf, cls, keypoints, scale, had_pad]
            Detecções do YOLO (somente pessoas, cls=0)
        frame_shape : (height, width)
            Dimensões do frame original
        
        Retorna
        -------
        tracks_by_state : dict
            {
                'active': [...],      # tracks confirmados
                'pending': [...],     # tracks novos
                'temp_lost': [...],   # perdidos temporários
                'lost': [...]         # perdidos definitivos
            }
        """
        
        self.frame_h, self.frame_w = frame_shape
        self.frame_id += 1
        
        if len(detections) == 0:
            # ============================================================
            # SEM DETECÇÕES: atualiza tracker com array vazio
            # ============================================================
            empty = np.empty((0, 5), dtype=np.float32)
            img_info = (self.frame_h, self.frame_w)
            img_size = (self.frame_h, self.frame_w)
            online_targets = self.tracker.update(empty, img_info, img_size)
            
            # ============================================================
            # DETECTA LOST (todos tracks anteriores viraram LOST)
            # ============================================================
            lost_tracks = self._extract_lost_tracks()
            
            # ============================================================
            # LOG: sem detecções
            # ============================================================
            #print(f"[BT_ASSOC] f={self.frame_id} | upd_high=0 upd_low=0 new=0 lost={len(lost_tracks)}")
            
            return {
                'active': [],
                'pending': [],
                'temp_lost': list(self.tracker.lost_stracks),
                'lost': lost_tracks
            }
        
        # ============================================================
        # 1) SEPARA BBOXES/SCORES E METADATA
        # ============================================================
        bboxes_scores = []
        metadata_map = {}  # idx -> {keypoints, scale, had_pad}
        
        for idx, det in enumerate(detections):
            x1, y1, x2, y2, conf, cls, keypoints, scale, had_pad = det
            bboxes_scores.append([x1, y1, x2, y2, conf])
            metadata_map[idx] = {
                'keypoints': keypoints,
                'scale': scale,
                'had_pad': had_pad
            }
        
        # Converte para numpy
        dets_array = np.array(bboxes_scores, dtype=np.float32)
        
        # ============================================================
        # 2) ATUALIZA BYTETRACKER (2 estágios + Kalman 8D)
        # ============================================================
        img_info = (self.frame_h, self.frame_w)
        img_size = (self.frame_h, self.frame_w)
        
        online_targets = self.tracker.update(dets_array, img_info, img_size)
        
        # ============================================================
        # 3) ASSOCIA METADATA (keypoints, scale, pad) VIA IoU
        # ============================================================
        active_tracks = []
        pending_tracks = []
        
        for track in online_targets:
            track_bbox = track.tlbr  # [x1, y1, x2, y2]
            track_id = track.track_id
            track_score = track.score
            
            # ============================================================
            # ENCONTRA DETECÇÃO MAIS PRÓXIMA (via IoU)
            # ============================================================
            best_idx = None
            best_iou = 0.0
            
            for idx, det in enumerate(detections):
                det_bbox = det[:4]
                iou = self._compute_iou(track_bbox, det_bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx
            
            # ============================================================
            # EXTRAI METADATA (ou None se não encontrou match)
            # ============================================================
            if best_idx is not None:
                meta = metadata_map[best_idx]
                kps = meta['keypoints']
                scale = meta['scale']
                had_pad = meta['had_pad']
            else:
                kps = None
                scale = "UNKNOWN"
                had_pad = False
            
            # ============================================================
            # MONTA TRACK COM METADATA
            # ============================================================
            track_dict = {
                'track_id': track_id,
                'bbox': tuple(track_bbox.tolist()),
                'score': float(track_score),
                'keypoints': kps,
                'scale': scale,
                'had_pad': had_pad,
                'is_activated': track.is_activated,
                'tracklet_len': track.tracklet_len
            }
            
            # ============================================================
            # CLASSIFICA ESTADO: ACTIVE vs PENDING
            # ============================================================
            # ACTIVE: is_activated=True (confirmado pelo ByteTracker)
            # PENDING: is_activated=False (novo, aguardando confirmação)
            if track.is_activated:
                active_tracks.append(track_dict)
            else:
                pending_tracks.append(track_dict)
        
        # ============================================================
        # 4) EXTRAI TEMP_LOST (lost_stracks do ByteTracker)
        # ============================================================
        temp_lost_tracks = []
        for track in self.tracker.lost_stracks:
            temp_lost_tracks.append({
                'track_id': track.track_id,
                'bbox': tuple(track.tlbr.tolist()),
                'score': float(track.score),
                'keypoints': None,  # sem detecção → sem keypoints
                'scale': "UNKNOWN",
                'had_pad': False,
                'is_activated': track.is_activated,
                'tracklet_len': track.tracklet_len,
                'time_since_update': track.time_since_update
            })
        
        # ============================================================
        # 5) EXTRAI LOST (removed_stracks do ByteTracker)
        # ============================================================
        lost_tracks = self._extract_lost_tracks()
        
        # ============================================================
        # LOG: [BT_ASSOC] estatísticas do update
        # ============================================================
        n_upd_high = len([t for t in active_tracks if t['score'] >= 0.55])
        n_upd_low = len([t for t in active_tracks if t['score'] < 0.55])
        n_new = len(pending_tracks)
        n_lost = len(lost_tracks)
        
        #print(f"[BT_ASSOC] f={self.frame_id} | upd_high={n_upd_high} upd_low={n_upd_low} new={n_new} lost={n_lost}")
        
        # ============================================================
        # ATUALIZA CACHE PARA PRÓXIMO FRAME
        # ============================================================
        self._prev_active_ids = {t['track_id'] for t in active_tracks}
        self._prev_lost_ids = {t['track_id'] for t in lost_tracks}
        
        return {
            'active': active_tracks,
            'pending': pending_tracks,
            'temp_lost': temp_lost_tracks,
            'lost': lost_tracks
        }

    def _extract_lost_tracks(self) -> List[Dict[str, Any]]:
        """
        Extrai tracks LOST (removed_stracks).
        Só retorna tracks que foram ativos no frame anterior.
        
        Retorna
        -------
        lost_tracks : list[dict]
            Tracks que transitaram para LOST neste frame
        """
        lost_tracks = []
        
        current_lost_ids = {t.track_id for t in self.tracker.removed_stracks}
        
        # ============================================================
        # NOVOS LOST: tracks que não estavam em _prev_lost_ids
        # ============================================================
        new_lost_ids = current_lost_ids - self._prev_lost_ids
        
        for track in self.tracker.removed_stracks:
            if track.track_id in new_lost_ids:
                lost_tracks.append({
                    'track_id': track.track_id,
                    'bbox': tuple(track.tlbr.tolist()),
                    'score': float(track.score),
                    'keypoints': None,
                    'scale': "UNKNOWN",
                    'had_pad': False,
                    'is_activated': track.is_activated,
                    'tracklet_len': track.tracklet_len,
                    'end_frame': track.end_frame
                })
        
        return lost_tracks

    @staticmethod
    def _compute_iou(bbox_a, bbox_b) -> float:
        """Calcula IoU entre duas bboxes [x1,y1,x2,y2]"""
        ax1, ay1, ax2, ay2 = bbox_a
        bx1, by1, bx2, by2 = bbox_b
        
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        
        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        
        union = area_a + area_b - inter_area + 1e-6
        return inter_area / union

    def get_active_count(self) -> int:
        """Retorna quantidade de tracks ativos"""
        return len([t for t in self.tracker.tracked_stracks if t.is_activated])

    def get_density(self) -> float:
        """
        Calcula densidade de tracks por área visível.
        Usado para lock adaptativo.
        
        Retorna
        -------
        density : float
            n_active_tracks / (H*W / 1000)
        """
        n_active = self.get_active_count()
        area = (self.frame_h * self.frame_w) / 1000.0
        return n_active / area if area > 0 else 0.0

    def reset(self):
        """Reset completo do tracker"""
        args = Args(track_thresh=0.5, track_buffer=30, match_thresh=0.8)
        self.tracker = BYTETracker(args, frame_rate=30)
        self.frame_id = 0
        self._prev_active_ids = set()
        self._prev_lost_ids = set()