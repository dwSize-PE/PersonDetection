"""
Identity Bank — Banco de identidades globais consolidadas

Armazena identidades de pessoas que saíram de cena para Re-ID futuro.
Cada identidade tem: protótipos multi-escala, cor persistente, metadata completa.

Fluxo comercial:
1. Track some → Gallery consolida embeddings → Banco armazena identidade
2. Novo track surge → Busca no banco via cosine similarity multi-escala
3. Match encontrado → Reutiliza id_global + cor + atualiza embedding (EMA)
4. Sem match → Cria nova identidade global
5. TTL dinâmico: baseado em presence, confidence_mean, health, reentry_count
6. Health decay: -0.01/s (normal) ou -0.005/s (TTL<60s)
7. LRU: cap=64, remove menos relevante quando cheio
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import time
import torch
import torch.nn.functional as F

# ============================================================
# CONFIGURAÇÕES DO BANCO
# ============================================================
BANK_CAPACITY = 64          # máximo de identidades armazenadas
TTL_MIN = 30.0              # TTL mínimo (segundos)
TTL_MAX = 300.0             # TTL máximo (5 minutos)

# ============================================================
# HEALTH DECAY
# ============================================================
HEALTH_DECAY_NORMAL = 0.01  # decay/s normal
HEALTH_DECAY_SLOW = 0.005   # decay/s quando TTL<60s
HEALTH_DECAY_THRESHOLD = 60.0  # threshold para decay lento

# ============================================================
# THRESHOLDS DE QUALIDADE
# ============================================================
HEALTH_MIN = 0.30           # health mínimo (abaixo = rebaixado)
HEALTH_PRIORITY_PENALTY = 1.3  # penalidade no custo se health baixo


@dataclass
class Identity:
    """
    Identidade global consolidada de uma pessoa.
    
    Atributos
    ---------
    id_global : int
        ID único e persistente da pessoa (independente de track_id)
    prototypes : dict
        Protótipos por escala {"NEAR": Tensor, "MID": Tensor, "FAR": Tensor}
    color : tuple
        Cor BGR persistente (B, G, R)
    last_seen_frame : int
        Último frame onde pessoa foi vista
    created_at : float
        Timestamp de criação (para TTL)
    presence : float
        Tempo total de presença acumulado (segundos)
    confidence_mean : float
        Confiança média das detecções
    health : float
        Métrica de qualidade × consistência (0.0-1.0)
    reentry_count : int
        Número de vezes que pessoa reapareceu (re-entries)
    zone_out : int
        Zona de saída (grid 3×3: 0-8, -1=desconhecido)
    ttl : float
        Time-to-live dinâmico (segundos)
    last_update : float
        Timestamp do último update (para decay)
    n_updates : int
        Número de vezes que embedding foi atualizado
    """
    id_global: int
    prototypes: Dict[str, torch.Tensor] = field(default_factory=dict)
    color: Tuple[int, int, int] = (128, 128, 128)
    last_seen_frame: int = 0
    created_at: float = field(default_factory=time.time)
    presence: float = 0.0
    confidence_mean: float = 0.5
    health: float = 1.0
    reentry_count: int = 0
    zone_out: int = -1
    ttl: float = TTL_MIN
    last_update: float = field(default_factory=time.time)
    n_updates: int = 1
    
    # ============================================================
    # EMBEDDING PRINCIPAL (fallback se sem protótipos multi-escala)
    # ============================================================
    embedding: torch.Tensor = field(default_factory=lambda: torch.zeros(512))


class IdentityBank:
    """
    Banco de identidades globais para Re-ID.
    
    Gerencia ciclo de vida das identidades:
    - Criação quando track some pela primeira vez
    - Busca por similaridade multi-escala quando novo track surge
    - Atualização via EMA quando pessoa é re-identificada
    - TTL dinâmico + health decay + LRU
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
            Momentum base para atualização EMA (0.20, será adaptativo)
        """
        self.match_threshold = match_threshold
        self.ema_momentum = ema_momentum
        
        # ============================================================
        # BANCO DE IDENTIDADES: id_global -> Identity
        # ============================================================
        self.identities: Dict[int, Identity] = {}
        
        # ============================================================
        # CONTADOR PARA GERAR NOVOS IDs GLOBAIS
        # ============================================================
        self._next_id_global = 0

    # =========================================================================
    # BUSCA (multi-escala com fallback)
    # =========================================================================

    def search(self, 
               embedding: torch.Tensor,
               scale: str = "MID") -> Optional[Tuple[int, float]]:
        """
        Busca identidade mais similar no banco via cosine similarity.
        Estratégia multi-escala: tenta escala exata, depois escalas vizinhas.
        
        Parâmetros
        ----------
        embedding : torch.Tensor
            Embedding (512,) L2-normalizado para buscar
        scale : str
            Escala do track ("NEAR", "MID", "FAR")
        
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
        
        # ============================================================
        # ESTRATÉGIA 1: busca na escala exata
        # ============================================================
        for id_global, identity in self.identities.items():
            # Tenta protótipo da escala exata
            proto = identity.prototypes.get(scale)
            
            if proto is None:
                # Fallback: embedding principal
                proto = identity.embedding
            
            if proto is None or proto.numel() == 0:
                continue
            
            proto = proto.view(1, -1)  # (1, 512)
            sim = float(torch.matmul(emb, proto.T).item())
            
            if sim > best_sim:
                best_sim = sim
                best_id = id_global
        
        # ============================================================
        # ESTRATÉGIA 2: se sim < threshold + δ, tenta escalas vizinhas
        # ============================================================
        if best_sim < self.match_threshold + 0.10:  # δ = 0.10
            # Define escalas vizinhas
            neighbors = {
                "NEAR": ["MID"],
                "MID": ["NEAR", "FAR"],
                "FAR": ["MID"]
            }
            
            neighbor_scales = neighbors.get(scale, [])
            
            for neighbor_scale in neighbor_scales:
                for id_global, identity in self.identities.items():
                    proto = identity.prototypes.get(neighbor_scale)
                    
                    if proto is None:
                        continue
                    
                    proto = proto.view(1, -1)
                    sim = float(torch.matmul(emb, proto.T).item())
                    
                    if sim > best_sim:
                        best_sim = sim
                        best_id = id_global
        
        # ============================================================
        # VALIDA THRESHOLD
        # ============================================================
        if best_id is not None and best_sim >= self.match_threshold:
            return (best_id, best_sim)
        
        return None

    def search_all(self, embedding: torch.Tensor) -> List[Tuple[int, float]]:
        """
        Busca todas identidades acima do threshold (para Hungarian).
        
        Retorna
        -------
        matches : list[(id_global, similarity)]
            Lista ordenada por similaridade (maior primeiro)
        """
        if not self.identities:
            return []
        
        emb = F.normalize(embedding.detach().cpu(), p=2, dim=0).view(1, -1)
        
        results = []
        
        for id_global, identity in self.identities.items():
            # Usa embedding principal (ou primeiro protótipo disponível)
            proto = identity.embedding
            
            if proto is None or proto.numel() == 0:
                # Fallback: primeiro protótipo disponível
                if identity.prototypes:
                    proto = next(iter(identity.prototypes.values()))
                else:
                    continue
            
            proto = proto.view(1, -1)
            sim = float(torch.matmul(emb, proto.T).item())
            
            if sim >= self.match_threshold:
                results.append((id_global, sim))
        
        # Ordena por similaridade (maior primeiro)
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results

    # =========================================================================
    # CRIAÇÃO
    # =========================================================================

    def add(self, 
            embedding: torch.Tensor, 
            color: Tuple[int, int, int],
            frame_index: int,
            prototypes: Optional[Dict[str, torch.Tensor]] = None,
            presence: float = 0.0,
            confidence_mean: float = 0.5) -> int:
        """
        Cria nova identidade global no banco.
        
        Parâmetros
        ----------
        embedding : torch.Tensor
            Embedding consolidado (512,) - fallback se sem protótipos
        color : (B, G, R)
            Cor BGR da pessoa
        frame_index : int
            Frame onde pessoa sumiu
        prototypes : dict | None
            Protótipos multi-escala {"NEAR": Tensor, ...}
        presence : float
            Tempo total de presença (segundos)
        confidence_mean : float
            Confiança média das detecções
        
        Retorna
        -------
        id_global : int
            Novo ID global criado
        """
        id_global = self._next_id_global
        self._next_id_global += 1
        
        emb = F.normalize(embedding.detach().cpu(), p=2, dim=0)
        
        # ============================================================
        # NORMALIZA PROTÓTIPOS
        # ============================================================
        normalized_protos = {}
        if prototypes:
            for scale, proto in prototypes.items():
                normalized_protos[scale] = F.normalize(proto.detach().cpu(), p=2, dim=0)
        
        # ============================================================
        # CALCULA TTL INICIAL
        # ============================================================
        health_initial = 1.0  # nova identidade começa com saúde máxima
        reentry_initial = 0
        
        ttl = self._compute_ttl(
            presence=presence,
            confidence_mean=confidence_mean,
            health=health_initial,
            reentry_count=reentry_initial
        )
        
        # ============================================================
        # CRIA IDENTIDADE
        # ============================================================
        current_time = time.time()
        
        self.identities[id_global] = Identity(
            id_global=id_global,
            prototypes=normalized_protos,
            embedding=emb,
            color=color,
            last_seen_frame=frame_index,
            created_at=current_time,
            presence=presence,
            confidence_mean=confidence_mean,
            health=health_initial,
            reentry_count=reentry_initial,
            zone_out=-1,  # desconhecido
            ttl=ttl,
            last_update=current_time,
            n_updates=1
        )
        
        # ============================================================
        # LRU: remove excesso se capacidade ultrapassada
        # ============================================================
        self._enforce_capacity()
        
        # ============================================================
        # LOG: BANK_ADD
        # ============================================================
        n_protos = len(normalized_protos)
        print(f"[BANK_ADD] pid=P{id_global:02d} ttl={ttl:.0f}s zone=? health={health_initial:.2f} protos={n_protos}")
        
        return id_global

    # =========================================================================
    # ATUALIZAÇÃO
    # =========================================================================

    def update(self, 
               id_global: int, 
               new_embedding: torch.Tensor,
               frame_index: int,
               alpha: Optional[float] = None) -> None:
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
        alpha : float | None
            Momentum EMA (se None, usa padrão ou adaptativo)
        """
        if id_global not in self.identities:
            return
        
        identity = self.identities[id_global]
        
        new_emb = F.normalize(new_embedding.detach().cpu(), p=2, dim=0)
        
        # ============================================================
        # α ADAPTATIVO (baseado em health)
        # ============================================================
        if alpha is None:
            # α menor para identidades saudáveis (mais conservador)
            alpha = max(0.03, min(0.10, 0.10 - 0.05 * identity.health))
        
        # ============================================================
        # EMA: suaviza mudanças bruscas de aparência
        # ============================================================
        updated_emb = (1 - alpha) * identity.embedding + alpha * new_emb
        identity.embedding = F.normalize(updated_emb, p=2, dim=0)
        
        # ============================================================
        # ATUALIZA METADATA
        # ============================================================
        identity.last_seen_frame = frame_index
        identity.last_update = time.time()
        identity.n_updates += 1
        identity.reentry_count += 1  # re-entry detectado
        
        # ============================================================
        # RECALCULA TTL (aumenta por re-entry)
        # ============================================================
        identity.ttl = self._compute_ttl(
            presence=identity.presence,
            confidence_mean=identity.confidence_mean,
            health=identity.health,
            reentry_count=identity.reentry_count
        )
        
        # ============================================================
        # RESETA HEALTH (re-entry = boa saúde)
        # ============================================================
        identity.health = min(1.0, identity.health + 0.2)  # boost

    def update_prototype(self,
                         id_global: int,
                         scale: str,
                         new_embedding: torch.Tensor,
                         alpha: float = 0.20) -> None:
        """
        Atualiza protótipo de uma escala específica.
        
        Parâmetros
        ----------
        id_global : int
            ID da identidade
        scale : str
            "NEAR", "MID", "FAR"
        new_embedding : torch.Tensor
            Novo embedding da escala
        alpha : float
            Momentum EMA
        """
        if id_global not in self.identities:
            return
        
        identity = self.identities[id_global]
        new_emb = F.normalize(new_embedding.detach().cpu(), p=2, dim=0)
        
        if scale in identity.prototypes:
            # ============================================================
            # EMA: atualiza protótipo existente
            # ============================================================
            old_proto = identity.prototypes[scale]
            updated_proto = (1 - alpha) * old_proto + alpha * new_emb
            identity.prototypes[scale] = F.normalize(updated_proto, p=2, dim=0)
        else:
            # ============================================================
            # CRIA NOVO PROTÓTIPO
            # ============================================================
            identity.prototypes[scale] = new_emb

    # =========================================================================
    # MANUTENÇÃO (TTL + HEALTH + LRU)
    # =========================================================================

    def tick(self, dt: float = 1.0) -> List[int]:
        """
        Atualiza TTL e health de todas identidades.
        Remove identidades expiradas.
        
        Parâmetros
        ----------
        dt : float
            Delta time (segundos) desde último tick
        
        Retorna
        -------
        expired_ids : list[int]
            IDs removidos por TTL
        """
        current_time = time.time()
        expired = []
        
        for id_global, identity in list(self.identities.items()):
            # ============================================================
            # CALCULA TEMPO DESDE ÚLTIMA ATUALIZAÇÃO
            # ============================================================
            time_since_update = current_time - identity.last_update
            
            # ============================================================
            # HEALTH DECAY (condicional por TTL restante)
            # ============================================================
            ttl_remaining = identity.ttl - time_since_update
            
            if ttl_remaining > HEALTH_DECAY_THRESHOLD:
                decay_rate = HEALTH_DECAY_NORMAL
            else:
                decay_rate = HEALTH_DECAY_SLOW  # mais lento perto de expirar
            
            identity.health -= decay_rate * dt
            identity.health = max(0.0, identity.health)
            
            # ============================================================
            # TTL: verifica expiração
            # ============================================================
            if ttl_remaining <= 0:
                # Expirou
                expired.append(id_global)
                del self.identities[id_global]
                print(f"[BANK_TTL] pid=P{id_global:02d} EXPIRED (ttl=0s)")
            else:
                # ============================================================
                # LOG: TTL restante
                # ============================================================
                if id_global % 10 == 0 or ttl_remaining < 30:  # log seletivo
                    print(f"[BANK_TTL] pid=P{id_global:02d} left={ttl_remaining:.0f}s health={identity.health:.2f} decay={decay_rate:.3f}")
        
        return expired

    def _enforce_capacity(self) -> None:
        """
        Aplica LRU quando banco ultrapassa capacidade.
        Remove identidade menos relevante (menor health × tempo restante).
        """
        if len(self.identities) <= BANK_CAPACITY:
            return
        
        # ============================================================
        # CALCULA SCORE DE RELEVÂNCIA (health × TTL_norm)
        # ============================================================
        current_time = time.time()
        scores = {}
        
        for id_global, identity in self.identities.items():
            time_since_update = current_time - identity.last_update
            ttl_remaining = identity.ttl - time_since_update
            ttl_norm = ttl_remaining / TTL_MAX
            
            score = identity.health * ttl_norm
            scores[id_global] = score
        
        # ============================================================
        # REMOVE MENOR SCORE (LRU)
        # ============================================================
        lru_id = min(scores, key=scores.get)  # type: ignore
        
        del self.identities[lru_id]
        print(f"[BANK_LRU] pid=P{lru_id:02d} DROPPED (capacity={BANK_CAPACITY}) score={scores[lru_id]:.2f}")

    def _compute_ttl(self,
                     presence: float,
                     confidence_mean: float,
                     health: float,
                     reentry_count: int) -> float:
        """
        Calcula TTL dinâmico.
        
        Fórmula:
        ttl = clamp(0.1*presence + 10*conf_mean + 40*health + 20*reentry, 30, 300)
        
        Retorna
        -------
        ttl : float
            Time-to-live em segundos
        """
        ttl = (0.1 * presence + 
               10.0 * confidence_mean + 
               40.0 * health + 
               20.0 * reentry_count)
        
        ttl = max(TTL_MIN, min(TTL_MAX, ttl))
        
        return ttl

    # =========================================================================
    # QUERIES
    # =========================================================================

    def get(self, id_global: int) -> Optional[Identity]:
        """Retorna identidade por ID global"""
        return self.identities.get(id_global)

    def get_color(self, id_global: int) -> Optional[Tuple[int, int, int]]:
        """Retorna cor BGR de uma identidade"""
        identity = self.identities.get(id_global)
        return identity.color if identity else None

    def get_health(self, id_global: int) -> Optional[float]:
        """Retorna health de uma identidade"""
        identity = self.identities.get(id_global)
        return identity.health if identity else None

    def exists(self, id_global: int) -> bool:
        """Verifica se ID global existe no banco"""
        return id_global in self.identities

    def remove(self, id_global: int) -> None:
        """Remove identidade do banco (raramente usado)"""
        if id_global in self.identities:
            del self.identities[id_global]
            print(f"[BANK_DEL] pid=P{id_global:02d} removed manually")

    def size(self) -> int:
        """Retorna quantidade de identidades no banco"""
        return len(self.identities)

    def get_stats(self) -> Dict[str, any]:
        """
        Retorna estatísticas do banco.
        
        Retorna
        -------
        stats : dict
        """
        if not self.identities:
            return {
                'count': 0,
                'avg_health': 0.0,
                'avg_ttl': 0.0
            }
        
        current_time = time.time()
        
        healths = [i.health for i in self.identities.values()]
        ttls = [i.ttl - (current_time - i.last_update) for i in self.identities.values()]
        
        return {
            'count': len(self.identities),
            'avg_health': sum(healths) / len(healths),
            'avg_ttl': sum(ttls) / len(ttls),
            'min_health': min(healths),
            'max_health': max(healths)
        }

    # =========================================================================
    # RESET
    # =========================================================================

    def clear(self) -> None:
        """Limpa todo o banco (reset completo)"""
        self.identities = {}
        self._next_id_global = 0
        print("[BANK_CLEAR] All identities removed")