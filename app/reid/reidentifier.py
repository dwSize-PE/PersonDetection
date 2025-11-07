"""
ReIdentifier — Orquestrador de Re-ID (Gallery ↔ IdentityBank)

Responsabilidades:
1. Mapeia track_id (temporário) → id_global (persistente)
2. Acumula embeddings na Gallery enquanto track ativo
3. Quando track some: consolida Gallery → IdentityBank
4. Quando novo track surge: busca no Bank → reutiliza ou cria identidade
5. Gerencia cores persistentes por id_global
"""

from __future__ import annotations
from typing import Dict, Optional, Tuple
import random

import torch
import torch.nn.functional as F

from .gallery import Gallery
from .identity_bank import IdentityBank


class ReIdentifier:
    """
    Motor de Re-ID — ponte entre tracking temporário e identidades globais.
    
    Fluxo operacional:
    - Track ativo: acumula embeddings na Gallery
    - Track some: consolida no IdentityBank
    - Track novo: busca no Bank por similaridade
    """
    
    def __init__(self,
                 gallery: Gallery,
                 min_samples: int = 8,
                 match_threshold: float = 0.65,
                 ema_momentum: float = 0.20):
        """
        Parâmetros
        ----------
        gallery : Gallery
            Instância da galeria de embeddings
        min_samples : int
            Mínimo de embeddings para consolidar identidade (8 = ~0.25s a 30fps)
        match_threshold : float
            Threshold cosine similarity para Re-ID (0.65 = conservador comercial)
        ema_momentum : float
            Momentum para atualização EMA quando re-identificar
        """
        self.gallery = gallery
        self.min_samples = min_samples
        
        self.bank = IdentityBank(
            match_threshold=match_threshold,
            ema_momentum=ema_momentum
        )
        
        # Mapeamento: track_id → id_global (durante tracking ativo)
        self._track_to_global: Dict[int, int] = {}
        
        # Mapeamento: track_id → cor (tracks que ainda não foram promovidos)
        self._track_colors: Dict[int, Tuple[int, int, int]] = {}

    # =========================================================================
    # API PRINCIPAL
    # =========================================================================

    def on_track_active(self, track_id: int, embedding: torch.Tensor, frame_index: int) -> None:
        """
        Chamado frame a frame para tracks ativos.
        Acumula embeddings na Gallery.
        
        Parâmetros
        ----------
        track_id : int
            ID do track ativo (ByteTracker)
        embedding : torch.Tensor
            Embedding (512,) do OSNet
        frame_index : int
            Frame atual
        """
        # Apenas acumula na Gallery — Re-ID só quando track sumir
        self.gallery.add(track_id, embedding, frame_index)

    def on_track_lost(self, track_id: int, frame_index: int) -> Optional[int]:
        """
        Chamado quando track some (ByteTracker.removed_stracks).
        Consolida embeddings da Gallery → IdentityBank.
        
        Parâmetros
        ----------
        track_id : int
            ID do track que sumiu
        frame_index : int
            Frame onde track sumiu
        
        Retorna
        -------
        id_global : int | None
            ID global consolidado ou None se não tinha embeddings suficientes
        """
        # Se track já tinha id_global, atualiza banco com novos embeddings
        if track_id in self._track_to_global:
            id_global = self._track_to_global[track_id]
            self._update_bank_from_gallery(track_id, id_global, frame_index)
            
            # Limpa mapeamentos
            del self._track_to_global[track_id]
            if track_id in self._track_colors:
                del self._track_colors[track_id]
            
            self.gallery.delete(track_id)
            return id_global
        
        # Track novo que nunca foi promovido — tenta consolidar no banco
        if not self.gallery.exists(track_id):
            return None
        
        # Verifica se tem embeddings suficientes
        if self.gallery.count(track_id) < self.min_samples:
            print(f"[ReID] Track {track_id} perdido com poucos samples ({self.gallery.count(track_id)})")
            self.gallery.delete(track_id)
            return None
        
        # Consolida embedding médio
        consolidated_emb = self.gallery.get(track_id)
        if consolidated_emb is None:
            self.gallery.delete(track_id)
            return None
        
        # Busca no banco
        match = self.bank.search(consolidated_emb)
        
        if match is not None:
            # Re-identificou pessoa existente
            id_global, similarity = match
            self.bank.update(id_global, consolidated_emb, frame_index)
            print(f"[ReID] Track {track_id} consolidado → RID {id_global} (sim={similarity:.3f})")
        else:
            # Nova pessoa — cria identidade no banco
            color = self._track_colors.get(track_id, self._generate_color())
            id_global = self.bank.add(consolidated_emb, color, frame_index)
            print(f"[ReID] Track {track_id} → NOVA identidade RID {id_global}")
        
        # Limpa galeria
        self.gallery.delete(track_id)
        if track_id in self._track_colors:
            del self._track_colors[track_id]
        
        return id_global

    def on_new_track(self, track_id: int, embedding: torch.Tensor, frame_index: int) -> Optional[int]:
        """
        Chamado quando novo track surge pela primeira vez.
        Tenta Re-ID imediato se já tiver embeddings bons na Gallery.
        
        Estratégia: aguarda min_samples antes de tentar Re-ID.
        
        Parâmetros
        ----------
        track_id : int
            ID do novo track
        embedding : torch.Tensor
            Primeiro embedding coletado
        frame_index : int
            Frame atual
        
        Retorna
        -------
        id_global : int | None
            ID global se Re-ID bem-sucedido, None se ainda coletando
        """
        # Acumula na Gallery
        self.gallery.add(track_id, embedding, frame_index)
        
        # Aguarda amostras suficientes
        if self.gallery.count(track_id) < self.min_samples:
            # Gera cor temporária se ainda não tem
            if track_id not in self._track_colors:
                self._track_colors[track_id] = self._generate_color()
            return None
        
        # Tenta Re-ID
        consolidated_emb = self.gallery.get(track_id)
        if consolidated_emb is None:
            return None
        
        match = self.bank.search(consolidated_emb)
        
        if match is not None:
            # Re-identificou — mapeia track → id_global
            id_global, similarity = match
            self._track_to_global[track_id] = id_global
            
            # Remove cor temporária (usa cor do banco)
            if track_id in self._track_colors:
                del self._track_colors[track_id]
            
            print(f"[ReID] Track {track_id} RE-IDENTIFICADO → RID {id_global} (sim={similarity:.3f})")
            return id_global
        
        # Ainda sem match — continua coletando
        return None

    # =========================================================================
    # QUERIES
    # =========================================================================

    def get_global_id(self, track_id: int) -> Optional[int]:
        """Retorna id_global de um track (se já promovido)"""
        return self._track_to_global.get(track_id)

    def get_color(self, track_id: int) -> Tuple[int, int, int]:
        """
        Retorna cor BGR para um track.
        Se track tem id_global: usa cor do banco (persistente).
        Senão: usa cor temporária do track.
        """
        # Tem id_global? Usa cor do banco
        if track_id in self._track_to_global:
            id_global = self._track_to_global[track_id]
            color = self.bank.get_color(id_global)
            if color is not None:
                return color
        
        # Usa cor temporária ou gera nova
        if track_id not in self._track_colors:
            self._track_colors[track_id] = self._generate_color()
        
        return self._track_colors[track_id]

    def is_promoted(self, track_id: int) -> bool:
        """Verifica se track já tem id_global"""
        return track_id in self._track_to_global

    # =========================================================================
    # INTERNOS
    # =========================================================================

    def _update_bank_from_gallery(self, track_id: int, id_global: int, frame_index: int) -> None:
        """Atualiza embedding do banco com dados recentes da Gallery"""
        if not self.gallery.exists(track_id):
            return
        
        consolidated_emb = self.gallery.get(track_id)
        if consolidated_emb is not None:
            self.bank.update(id_global, consolidated_emb, frame_index)

    @staticmethod
    def _generate_color() -> Tuple[int, int, int]:
        """Gera cor BGR aleatória"""
        return (
            random.randint(60, 255),
            random.randint(60, 255),
            random.randint(60, 255),
        )

    # =========================================================================
    # RESET
    # =========================================================================

    def reset(self) -> None:
        """Reset completo do sistema Re-ID"""
        self.gallery.clear_all()
        self.bank.clear()
        self._track_to_global = {}
        self._track_colors = {}