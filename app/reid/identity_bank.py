"""
Identity Bank — Banco de identidades globais consolidadas

Armazena identidades de pessoas que saíram de cena para Re-ID futuro.
Cada identidade tem: embedding consolidado, cor persistente, metadata.

Fluxo comercial:
1. Track some → Gallery consolida embeddings → Banco armazena identidade
2. Novo track surge → Busca no banco via cosine similarity
3. Match encontrado → Reutiliza id_global + cor + atualiza embedding (EMA)
4. Sem match → Cria nova identidade global
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F


@dataclass
class Identity:
    """
    Identidade global consolidada de uma pessoa.
    
    Atributos
    ---------
    id_global : int
        ID único e persistente da pessoa (independente de track_id)
    embedding : torch.Tensor
        Embedding consolidado (512,) L2-normalizado
    color : tuple
        Cor BGR persistente (B, G, R)
    last_seen_frame : int
        Último frame onde pessoa foi vista
    n_updates : int
        Número de vezes que embedding foi atualizado (para debug)
    """
    id_global: int
    embedding: torch.Tensor  # (512,) L2-norm
    color: Tuple[int, int, int]  # BGR
    last_seen_frame: int
    n_updates: int = 1


class IdentityBank:
    """
    Banco de identidades globais para Re-ID.
    
    Gerencia ciclo de vida das identidades:
    - Criação quando track some pela primeira vez
    - Busca por similaridade quando novo track surge
    - Atualização via EMA quando pessoa é re-identificada
    """
    
    def __init__(self, 
                 match_threshold: float = 0.65,
                 ema_momentum: float = 0.20):
        """
        Parâmetros
        ----------
        match_threshold : float
            Threshold de cosine similarity para considerar match (0.65 = conservador)
        ema_momentum : float
            Momentum para atualização EMA do embedding (0.20 = atualização moderada)
        """
        self.match_threshold = match_threshold
        self.ema_momentum = ema_momentum
        
        # Banco de identidades: id_global -> Identity
        self.identities: Dict[int, Identity] = {}
        
        # Contador para gerar novos IDs globais
        self._next_id_global = 0

    def search(self, embedding: torch.Tensor) -> Optional[Tuple[int, float]]:
        """
        Busca identidade mais similar no banco via cosine similarity.
        
        Parâmetros
        ----------
        embedding : torch.Tensor
            Embedding (512,) L2-normalizado para buscar
        
        Retorna
        -------
        match : (id_global, similarity) | None
            Melhor match se similarity >= threshold, None caso contrário
        """
        if not self.identities:
            return None
        
        emb = F.normalize(embedding.detach().cpu(), p=2, dim=0).view(1, -1)  # (1, 512)
        
        best_id: Optional[int] = None
        best_sim = -1.0
        
        for id_global, identity in self.identities.items():
            proto = identity.embedding.view(1, -1)  # (1, 512)
            sim = float(torch.matmul(emb, proto.T).item())
            
            if sim > best_sim:
                best_sim = sim
                best_id = id_global
        
        # Retorna match apenas se passar threshold
        if best_id is not None and best_sim >= self.match_threshold:
            return (best_id, best_sim)
        
        return None

    def add(self, 
            embedding: torch.Tensor, 
            color: Tuple[int, int, int],
            frame_index: int) -> int:
        """
        Cria nova identidade global no banco.
        
        Parâmetros
        ----------
        embedding : torch.Tensor
            Embedding consolidado (512,)
        color : (B, G, R)
            Cor BGR da pessoa
        frame_index : int
            Frame onde pessoa sumiu
        
        Retorna
        -------
        id_global : int
            Novo ID global criado
        """
        id_global = self._next_id_global
        self._next_id_global += 1
        
        emb = F.normalize(embedding.detach().cpu(), p=2, dim=0)
        
        self.identities[id_global] = Identity(
            id_global=id_global,
            embedding=emb,
            color=color,
            last_seen_frame=frame_index,
            n_updates=1
        )
        
        return id_global

    def update(self, 
               id_global: int, 
               new_embedding: torch.Tensor,
               frame_index: int) -> None:
        """
        Atualiza embedding de identidade existente via EMA.
        
        Estratégia: embedding_novo = (1-α)*embedding_antigo + α*embedding_novo
        Preserva características históricas enquanto adapta a novas aparências.
        
        Parâmetros
        ----------
        id_global : int
            ID da identidade a atualizar
        new_embedding : torch.Tensor
            Novo embedding (512,) da pessoa re-identificada
        frame_index : int
            Frame atual
        """
        if id_global not in self.identities:
            return
        
        identity = self.identities[id_global]
        
        new_emb = F.normalize(new_embedding.detach().cpu(), p=2, dim=0)
        m = self.ema_momentum
        
        # EMA: suaviza mudanças bruscas de aparência
        updated_emb = (1 - m) * identity.embedding + m * new_emb
        identity.embedding = F.normalize(updated_emb, p=2, dim=0)
        
        identity.last_seen_frame = frame_index
        identity.n_updates += 1

    def get(self, id_global: int) -> Optional[Identity]:
        """Retorna identidade por ID global"""
        return self.identities.get(id_global)

    def get_color(self, id_global: int) -> Optional[Tuple[int, int, int]]:
        """Retorna cor BGR de uma identidade"""
        identity = self.identities.get(id_global)
        return identity.color if identity else None

    def exists(self, id_global: int) -> bool:
        """Verifica se ID global existe no banco"""
        return id_global in self.identities

    def remove(self, id_global: int) -> None:
        """Remove identidade do banco (raramente usado)"""
        if id_global in self.identities:
            del self.identities[id_global]

    def size(self) -> int:
        """Retorna quantidade de identidades no banco"""
        return len(self.identities)

    def clear(self) -> None:
        """Limpa todo o banco (reset completo)"""
        self.identities = {}
        self._next_id_global = 0