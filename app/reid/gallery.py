"""
Gallery — Buffer de embeddings por track_id

Armazena histórico de embeddings (bons e ruins) enquanto track está ativo.
Quando track some, esses embeddings serão consolidados para Re-ID.

Estratégia comercial:
- Buffer FIFO de 15 embeddings por track_id
- Aceita embeddings variados (perto/longe/ângulos) para robustez
- Retorna embedding médio (mais estável que mediana)
- Mantém metadata (frame_index) para debug
"""

from __future__ import annotations
import torch
from typing import Dict, List, Optional

# Buffer size padrão (15 embeddings = ~0.5s a 30fps)
BUFFER_SIZE = 15


class Gallery:
    """
    Galeria de embeddings por pessoa (track_id).
    
    Cada track_id mantém histórico de até 15 embeddings recentes.
    Usado durante tracking ativo, antes de consolidar no banco.
    """
    
    def __init__(self, buffer_size: int = BUFFER_SIZE):
        self.buffer_size = buffer_size
        # track_id -> lista de {'emb': Tensor(512,), 'frame': int}
        self.data: Dict[int, List[Dict[str, object]]] = {}

    def add(self, track_id: int, embedding: Optional[torch.Tensor], frame_index: int) -> None:
        """
        Adiciona embedding ao histórico de um track.
        
        Parâmetros
        ----------
        track_id : int
            ID do track ativo (ByteTracker)
        embedding : torch.Tensor | None
            Embedding (512,) L2-normalizado do OSNet
        frame_index : int
            Frame atual (para debug)
        """
        if embedding is None:
            return  # ignora embeddings inválidos (crop falhou, etc)

        emb_cpu = embedding.detach().cpu()

        if track_id not in self.data:
            self.data[track_id] = []

        lst = self.data[track_id]
        lst.append({'emb': emb_cpu, 'frame': frame_index})

        # FIFO — mantém somente N mais recentes
        if len(lst) > self.buffer_size:
            lst.pop(0)

    def get(self, track_id: int) -> Optional[torch.Tensor]:
        """
        Retorna embedding médio (consolidado) do track.
        
        Usado para Re-ID quando track some.
        
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

        # Extrai embeddings e empilha
        embs: List[torch.Tensor] = [d['emb'] for d in lst]  # type: ignore
        stacked = torch.stack(embs, dim=0)  # (N, 512)
        
        # Média — mais estável que mediana para Re-ID
        mean_emb = stacked.mean(dim=0)
        return mean_emb

    def get_all(self, track_id: int) -> List[torch.Tensor]:
        """
        Retorna todos embeddings crus (sem consolidar).
        
        Útil para análise de qualidade ou estratégias alternativas.
        """
        if track_id not in self.data:
            return []
        return [d['emb'] for d in self.data[track_id]]  # type: ignore

    def exists(self, track_id: int) -> bool:
        """Verifica se track tem embeddings na galeria"""
        return track_id in self.data and len(self.data[track_id]) > 0

    def count(self, track_id: int) -> int:
        """Retorna quantidade de embeddings acumulados para um track"""
        if track_id not in self.data:
            return 0
        return len(self.data[track_id])

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