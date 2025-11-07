"""
ByteTrack Wrapper — Integração YOLO → ByteTracker original + Re-ID

Responsabilidades:
- Converter detecções YOLO para formato ByteTracker
- Preservar keypoints para cada track_id
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
        Lista de [x1, y1, x2, y2, conf, cls, keypoints]
    
    Output:
        Lista de dicts {
            'track_id': int,
            'bbox': (x1, y1, x2, y2),
            'score': float,
            'keypoints': [(x,y,conf), ...],
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

    def update(self, 
               detections: List[List[Any]], 
               frame_shape: Tuple[int, int]) -> List[Dict[str, Any]]:
        """
        Atualiza tracker com novas detecções.
        
        Parâmetros
        ----------
        detections : lista de [x1, y1, x2, y2, conf, cls, keypoints]
            Detecções do YOLO (somente pessoas, cls=0)
        frame_shape : (height, width)
            Dimensões do frame original
        
        Retorna
        -------
        tracks : lista de dicts
            Tracks ativos com track_id, bbox, score, keypoints
        """
        
        self.frame_h, self.frame_w = frame_shape
        
        if len(detections) == 0:
            # sem detecções → atualiza tracker com array vazio
            empty = np.empty((0, 5), dtype=np.float32)
            img_info = (self.frame_h, self.frame_w)
            img_size = (self.frame_h, self.frame_w)
            online_targets = self.tracker.update(empty, img_info, img_size)
            return []
        
        # -------------------------------
        # 1) Separa bboxes/scores e keypoints
        # -------------------------------
        bboxes_scores = []
        keypoints_map = {}
        
        for idx, det in enumerate(detections):
            x1, y1, x2, y2, conf, cls, keypoints = det
            bboxes_scores.append([x1, y1, x2, y2, conf])
            keypoints_map[idx] = keypoints
        
        # Converte para numpy
        dets_array = np.array(bboxes_scores, dtype=np.float32)
        
        # -------------------------------
        # 2) Atualiza ByteTracker
        # -------------------------------
        img_info = (self.frame_h, self.frame_w)
        img_size = (self.frame_h, self.frame_w)
        
        online_targets = self.tracker.update(dets_array, img_info, img_size)
        
        # -------------------------------
        # 3) Associa keypoints aos tracks via IoU
        # -------------------------------
        output_tracks = []
        
        for track in online_targets:
            track_bbox = track.tlbr  # [x1, y1, x2, y2]
            track_id = track.track_id
            track_score = track.score
            
            # Encontra detecção mais próxima via IoU
            best_idx = None
            best_iou = 0.0
            
            for idx, det in enumerate(detections):
                det_bbox = det[:4]
                iou = self._compute_iou(track_bbox, det_bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx
            
            # Atribui keypoints (ou None se não encontrou match)
            kps = keypoints_map.get(best_idx) if best_idx is not None else None
            
            output_tracks.append({
                'track_id': track_id,
                'bbox': tuple(track_bbox.tolist()),
                'score': float(track_score),
                'keypoints': kps,
            })
        
        return output_tracks

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

    def get_lost_tracks(self) -> List[int]:
        """
        Retorna IDs dos tracks que foram perdidos (para Re-ID).
        """
        return [t.track_id for t in self.tracker.removed_stracks]

    def reset(self):
        """Reset completo do tracker"""
        args = Args(track_thresh=0.5, track_buffer=30, match_thresh=0.8)
        self.tracker = BYTETracker(args, frame_rate=30)