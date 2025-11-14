"""
Gallery – Buffer temporal de embeddings de qualidade por track_id

Armazena histórico de embeddings bons (pós-gate) enquanto track está ativo.
Quando track some, esses embeddings são consolidados para Re-ID.

Estratégia comercial:
- Buffer FIFO de 10 embeddings por track_id (cap reduzida para eficiência)
- Stride temporal ≥50ms (evita redundância)
- Aceita apenas embeddings que passaram pelo gate de qualidade
- Metadata completa: scale, blur_z, aspect_ratio, HSV, timestamp
- Protótipos MULTI-ESCALA via medoid (não média simples)
- Rejeição automática de outliers (sim < 0.70)
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

# ============================================================
# OUTLIER REJECTION
# ============================================================
OUTLIER_SIM_THRESHOLD = 0.70  # rejeita embeddings com sim < 0.70 ao medoid


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
        [DEPRECATED] Retorna embedding médio (compatibilidade).
        
        Use get_prototypes() para multi-escala (recomendado).
        
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
        # MÉDIA: compatibilidade com código legado
        # ============================================================
        mean_emb = stacked.mean(dim=0)
        
        # ============================================================
        # NORMALIZA APÓS MEAN (mean quebra normalização)
        # ============================================================
        mean_emb = F.normalize(mean_emb, p=2, dim=0)
        return mean_emb

    def get_prototypes(self, track_id: int) -> Dict[str, torch.Tensor]:
        """
        Retorna protótipos multi-escala via medoid + outlier rejection.
        
        Estratégia comercial (Hikvision/Face++):
        - Agrupa embeddings por escala (NEAR/MID/FAR)
        - Para cada escala: calcula medoid (embedding mais central)
        - Rejeita outliers (sim < 0.70 ao medoid)
        - Retorna protótipos robustos
        
        Parâmetros
        ----------
        track_id : int
            ID do track
        
        Retorna
        -------
        prototypes : dict
            {"NEAR": Tensor(512,), "MID": Tensor(512,), "FAR": Tensor(512,)}
            Escalas sem embeddings suficientes são omitidas
        """
        if track_id not in self.data:
            return {}
        
        lst = self.data[track_id]
        if not lst:
            return {}

        # ============================================================
        # AGRUPA EMBEDDINGS POR ESCALA
        # ============================================================
        by_scale: Dict[str, List[torch.Tensor]] = {"NEAR": [], "MID": [], "FAR": []}
        
        for item in lst:
            scale = item['scale']  # type: ignore
            emb = item['emb']      # type: ignore
            
            if scale in by_scale:
                by_scale[scale].append(emb)
        
        # ============================================================
        # CALCULA MEDOID + OUTLIER REJECTION PARA CADA ESCALA
        # ============================================================
        prototypes = {}
        
        for scale, embs in by_scale.items():
            if len(embs) < 2:
                # Mínimo 2 embeddings para calcular medoid
                continue
            
            # Calcula medoid
            medoid = self._compute_medoid(embs)
            
            # Rejeita outliers
            filtered = self._reject_outliers(embs, medoid)
            
            if filtered:
                # Recalcula medoid com embeddings filtrados
                prototypes[scale] = self._compute_medoid(filtered)
        
        return prototypes

    def _compute_medoid(self, embeddings: List[torch.Tensor]) -> torch.Tensor:
        """
        Calcula medoid (embedding mais central).
        
        Medoid = embedding com maior soma de similaridades aos demais.
        Mais robusto que média simples (não dilui características).
        
        Parâmetros
        ----------
        embeddings : list[Tensor]
            Lista de embeddings (512,) normalizados
        
        Retorna
        -------
        medoid : Tensor
            Embedding mais central (512,)
        """
        if len(embeddings) == 1:
            return embeddings[0]
        
        # ============================================================
        # EMPILHA EMBEDDINGS: (N, 512)
        # ============================================================
        stacked = torch.stack(embeddings, dim=0)  # (N, 512)
        
        # ============================================================
        # MATRIZ DE SIMILARIDADE: (N, N)
        # ============================================================
        sim_matrix = torch.matmul(stacked, stacked.T)  # (N, N)
        
        # ============================================================
        # SOMA SIMILARIDADES POR EMBEDDING (excluindo diagonal)
        # ============================================================
        sim_sums = sim_matrix.sum(dim=1) - 1.0  # subtrai diagonal (sim consigo mesmo = 1.0)
        
        # ============================================================
        # RETORNA EMBEDDING COM MAIOR SOMA (mais central)
        # ============================================================
        medoid_idx = sim_sums.argmax().item()
        return embeddings[medoid_idx]

    def _reject_outliers(self, 
                        embeddings: List[torch.Tensor],
                        medoid: torch.Tensor) -> List[torch.Tensor]:
        """
        Rejeita embeddings com similaridade baixa ao medoid.
        
        Threshold: sim >= 0.70 (conservador para Re-ID comercial).
        
        Parâmetros
        ----------
        embeddings : list[Tensor]
            Lista de embeddings (512,)
        medoid : Tensor
            Embedding de referência (512,)
        
        Retorna
        -------
        filtered : list[Tensor]
            Embeddings coesos (sim >= 0.70 ao medoid)
        """
        filtered = []
        
        medoid_view = medoid.view(1, -1)  # (1, 512)
        
        for emb in embeddings:
            emb_view = emb.view(1, -1)  # (1, 512)
            sim = float(torch.matmul(emb_view, medoid_view.T).item())
            
            if sim >= OUTLIER_SIM_THRESHOLD:
                filtered.append(emb)
        
        return filtered

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