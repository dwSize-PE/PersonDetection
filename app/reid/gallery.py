"""
Gallery — Buffer temporal de embeddings de qualidade por track_id

Armazena histórico de embeddings bons (pós-gate) enquanto track está ativo.
Quando track some, esses embeddings são consolidados para Re-ID.

Estratégia comercial:
- Buffer FIFO de 10 embeddings por track_id (cap reduzida para eficiência)
- Stride temporal ≥50ms (evita redundância)
- Aceita apenas embeddings que passaram pelo gate de qualidade
- Metadata completa: scale, blur_z, aspect_ratio, HSV, timestamp
- Suporta análise de diversidade (multi-escala, variação aspect/HSV)
"""

from __future__ import annotations
import torch
import time
import numpy as np
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

# ============================================================
# CONFIGURAÇÕES DO BUFFER
# ============================================================
BUFFER_SIZE = 10        # cap reduzida (vs 15 original) para eficiência
STRIDE_MS = 50          # stride mínimo entre embeddings (ms)


class Gallery:
    """
    Galeria de embeddings por pessoa (track_id).
    
    Cada track_id mantém histórico de até 10 embeddings recentes.
    Usado durante tracking ativo, antes de consolidar no banco.
    """
    
    def __init__(self, buffer_size: int = BUFFER_SIZE):
        self.buffer_size = buffer_size
        
        # ============================================================
        # STORAGE: track_id -> lista de items
        # ============================================================
        # item = {
        #     'emb': Tensor(512,),
        #     'scale': str,
        #     'blur_z': float,
        #     'aspect_ratio': float,
        #     'hsv_mean': np.ndarray(3,),
        #     'tstamp': float,
        #     'frame': int
        # }
        self.data: Dict[int, List[Dict[str, object]]] = {}

    def add(self, 
            track_id: int, 
            embedding: Optional[torch.Tensor],
            scale: str,
            blur_z: float,
            aspect_ratio: float,
            hsv_mean: np.ndarray,
            frame_index: int) -> bool:
        """
        Adiciona embedding ao histórico de um track (com stride temporal).
        
        Parâmetros
        ----------
        track_id : int
            ID do track ativo (ByteTracker)
        embedding : torch.Tensor | None
            Embedding (512,) L2-normalizado do OSNet
        scale : str
            "NEAR", "MID", "FAR", "DESC"
        blur_z : float
            Z-score de blur (qualidade relativa ao frame)
        aspect_ratio : float
            w/h do crop
        hsv_mean : np.ndarray
            Média HSV do crop (3,)
        frame_index : int
            Frame atual
        
        Retorna
        -------
        added : bool
            True se adicionado, False se rejeitado por stride
        """
        if embedding is None:
            return False

        emb_cpu = embedding.detach().cpu()
        tstamp = time.time()

        if track_id not in self.data:
            self.data[track_id] = []

        lst = self.data[track_id]
        
        # ============================================================
        # STRIDE TEMPORAL: rejeita se muito próximo do último
        # EXCEÇÃO: permite se escala diferente (diversidade)
        # ============================================================
        if len(lst) > 0:
            last_item = lst[-1]
            last_tstamp = last_item['tstamp']  # type: ignore
            last_scale = last_item['scale']    # type: ignore
            
            dt_ms = (tstamp - last_tstamp) * 1000.0
            
            # Rejeita se stride < 50ms E mesma escala
            if dt_ms < STRIDE_MS and scale == last_scale:
                return False

        # ============================================================
        # ADICIONA ITEM COM METADATA COMPLETA
        # ============================================================
        item = {
            'emb': emb_cpu,
            'scale': scale,
            'blur_z': blur_z,
            'aspect_ratio': aspect_ratio,
            'hsv_mean': hsv_mean.copy() if isinstance(hsv_mean, np.ndarray) else hsv_mean,
            'tstamp': tstamp,
            'frame': frame_index
        }
        
        lst.append(item)

        # ============================================================
        # FIFO: mantém somente N mais recentes
        # ============================================================
        if len(lst) > self.buffer_size:
            lst.pop(0)

        return True

    def get(self, track_id: int) -> Optional[torch.Tensor]:
        """
        Retorna embedding médio (consolidado) do track.
        
        Usado para Re-ID quando track some ou atinge min_samples.
        
        Retorna
        -------
        embedding : torch.Tensor | None
            Embedding médio (512,) ou None se track não existe
        """
        if track_id not in self.data:
            return None
        
        lst = self.data[track_id]
        if not lst:
            return None

        # ============================================================
        # EXTRAI EMBEDDINGS E EMPILHA
        # ============================================================
        embs: List[torch.Tensor] = [d['emb'] for d in lst]  # type: ignore
        stacked = torch.stack(embs, dim=0)  # (N, 512)
        
        # ============================================================
        # MÉDIA: mais estável que mediana para Re-ID
        # ============================================================
        mean_emb = stacked.mean(dim=0)
        mean_emb = F.normalize(mean_emb, p=2, dim=0)
        return mean_emb

    def get_all(self, track_id: int) -> List[torch.Tensor]:
        """
        Retorna todos embeddings crus (sem consolidar).
        
        Útil para análise de qualidade ou estratégias alternativas.
        """
        if track_id not in self.data:
            return []
        return [d['emb'] for d in self.data[track_id]]  # type: ignore

    def get_by_scale(self, track_id: int) -> Dict[str, List[torch.Tensor]]:
        """
        Retorna embeddings agrupados por escala.
        
        Usado para criar protótipos multi-escala no PID.
        
        Retorna
        -------
        by_scale : dict
            {"NEAR": [...], "MID": [...], "FAR": [...]}
        """
        if track_id not in self.data:
            return {}
        
        by_scale = {"NEAR": [], "MID": [], "FAR": [], "DESC": []}
        
        for item in self.data[track_id]:
            scale = item['scale']  # type: ignore
            emb = item['emb']      # type: ignore
            
            if scale in by_scale:
                by_scale[scale].append(emb)
        
        return by_scale

    def check_diversity(self, track_id: int) -> Tuple[bool, Dict[str, any]]:
        """
        Verifica se buffer tem diversidade suficiente para criar PID.
        
        Critérios (qualquer um válido):
        - ≥2 escalas diferentes
        - Δaspect_ratio ≥ 0.06
        - Variação HSV > 15 (qualquer canal)
        
        Retorna
        -------
        has_diversity : bool
            True se atende critério de diversidade
        stats : dict
            Estatísticas de diversidade (para debug/log)
        """
        if track_id not in self.data:
            return False, {}
        
        lst = self.data[track_id]
        if len(lst) < 2:
            return False, {'reason': 'insufficient_samples'}
        
        # ============================================================
        # CRITÉRIO 1: ≥2 ESCALAS DIFERENTES
        # ============================================================
        scales = set(item['scale'] for item in lst)  # type: ignore
        scales.discard("DESC")  # DESC não conta como escala válida
        
        if len(scales) >= 2:
            return True, {
                'criterion': 'multi_scale',
                'scales': list(scales),
                'n_scales': len(scales)
            }
        
        # ============================================================
        # CRITÉRIO 2: VARIAÇÃO ASPECT RATIO
        # ============================================================
        aspects = [item['aspect_ratio'] for item in lst]  # type: ignore
        aspect_min = min(aspects)
        aspect_max = max(aspects)
        delta_aspect = aspect_max - aspect_min
        
        if delta_aspect >= 0.06:
            return True, {
                'criterion': 'aspect_variation',
                'delta_aspect': delta_aspect,
                'range': (aspect_min, aspect_max)
            }
        
        # ============================================================
        # CRITÉRIO 3: VARIAÇÃO HSV (qualquer canal)
        # ============================================================
        hsv_arrays = [item['hsv_mean'] for item in lst]  # type: ignore
        hsv_stacked = np.stack(hsv_arrays, axis=0)  # (N, 3)
        
        hsv_min = hsv_stacked.min(axis=0)
        hsv_max = hsv_stacked.max(axis=0)
        hsv_range = hsv_max - hsv_min
        
        max_hsv_var = hsv_range.max()
        
        if max_hsv_var > 15.0:
            return True, {
                'criterion': 'hsv_variation',
                'max_var': max_hsv_var,
                'channel': int(np.argmax(hsv_range))
            }
        
        # ============================================================
        # SEM DIVERSIDADE
        # ============================================================
        return False, {
            'reason': 'insufficient_diversity',
            'n_scales': len(scales),
            'delta_aspect': delta_aspect,
            'max_hsv_var': max_hsv_var
        }

    def exists(self, track_id: int) -> bool:
        """Verifica se track tem embeddings na galeria"""
        return track_id in self.data and len(self.data[track_id]) > 0

    def count(self, track_id: int) -> int:
        """Retorna quantidade de embeddings acumulados para um track"""
        if track_id not in self.data:
            return 0
        return len(self.data[track_id])

    def get_stats(self, track_id: int) -> Optional[Dict[str, any]]:
        """
        Retorna estatísticas do buffer (para debug).
        
        Retorna
        -------
        stats : dict | None
            {
                'count': int,
                'scales': dict,
                'blur_z_mean': float,
                'aspect_mean': float,
                'age_s': float
            }
        """
        if track_id not in self.data:
            return None
        
        lst = self.data[track_id]
        if not lst:
            return None
        
        # Contagem por escala
        scales_count = {"NEAR": 0, "MID": 0, "FAR": 0, "DESC": 0}
        for item in lst:
            scale = item['scale']  # type: ignore
            if scale in scales_count:
                scales_count[scale] += 1
        
        # Blur médio
        blur_zs = [item['blur_z'] for item in lst]  # type: ignore
        blur_z_mean = np.mean(blur_zs)
        
        # Aspect médio
        aspects = [item['aspect_ratio'] for item in lst]  # type: ignore
        aspect_mean = np.mean(aspects)
        
        # Idade (tempo desde primeiro embedding)
        first_tstamp = lst[0]['tstamp']  # type: ignore
        last_tstamp = lst[-1]['tstamp']  # type: ignore
        age_s = last_tstamp - first_tstamp
        
        return {
            'count': len(lst),
            'scales': scales_count,
            'blur_z_mean': float(blur_z_mean),
            'aspect_mean': float(aspect_mean),
            'age_s': float(age_s)
        }

    def reset(self, track_id: int) -> None:
        """Limpa embeddings de um track específico"""
        if track_id in self.data:
            self.data[track_id] = []

    def delete(self, track_id: int) -> None:
        """Remove track da galeria (após consolidar no banco)"""
        if track_id in self.data:
            del self.data[track_id]

    def clear_all(self) -> None:
        """Reset completo da galeria"""
        self.data = {}