# -*- coding: utf-8 -*-
"""
ReIdentifier – Orquestrador de Re-ID (Gallery ↔ IdentityBank)

Reestruturação completa com separação explícita de fases:
  Fase 1) Coleta → criação de PID (pessoa nova)
  Fase 2) Re-identificação → matching contra banco (Hungarian + MFSS + K + Anti-teleport)
  Fase 3) Consolidação → quando o track some

Compatibilidade mantida com:
  - Gallery (buffer temporal, diversidade, protótipos por escala)
  - IdentityBank (prototipagem multi-escala, EMA, TTL dinâmico, health)
  - EmbedderThread (loop assíncrono com gate de qualidade)
  - ByteTrackWrapper (estados ACTIVE/PENDING/LOST, densidade, bbox)
  - Stream (overlay, cores, manutenção do banco)

Principais melhorias:
  • on_new_track() agora só orquestra e chama _try_create_pid() e _try_reidentify()
  • on_track_active() passou a realizar EMA guardado (antes era vazio)
  • Criação de PID grava protótipos multi-escala no banco (antes era perdido)
  • MFSS e APPT usam top-2 reais (por PID), e K-window com timeout
  • Anti-teleport com cache de posição e tempo (Δt e distância em fração da altura)
  • NEGATE penalty pós-erro com decaimento exponencial
  • Matching usa protótipo da escala correta (NEAR com NEAR, FAR com FAR)
  • Logs cirúrgicos e coesos, visando depuração comercial (Hikvision/Face++-style)
"""

from __future__ import annotations
from typing import Dict, Optional, Tuple, List, Deque
from collections import deque, defaultdict
import random
import time
import math

import numpy as np
import torch
import torch.nn.functional as F

from .gallery import Gallery
from .identity_bank import IdentityBank

# ============================================================
# CONFIGURAÇÕES RE-ID (mantendo compatibilidade)
# ============================================================
MIN_SAMPLES = 8             # mínimo de embeddings para criar PID / tentar Re-ID
MATCH_THRESHOLD = 0.65      # threshold base de cosine similarity
EMA_MOMENTUM = 0.20         # momentum base para EMA (será adaptativo)

# ============================================================
# MFSS (Moving Frame Similarity Score) + APPT (margem top1-top2)
# ============================================================
MFSS_BETA = 0.70            # suavização temporal (0.6-0.8)
APPT_MARGIN = 0.06          # margem mínima top1-top2

# ============================================================
# K CONFIRMAÇÕES (janela deslizante)
# ============================================================
K_WINDOW_SIZE = 8           # últimos 8 frames
K_REQUIRED = 4              # 4 positivos necessários
K_RESET_TIMEOUT = 2.0       # segundos - mantém K se Δt < 2s

# ============================================================
# LOCK ADAPTATIVO
# ============================================================
LOCK_BASE = 8               # base mínima
LOCK_SPEED_FACTOR = 15      # peso para movimento
LOCK_CROWD_FACTOR = 8       # peso para densidade
LOCK_MIN = 8
LOCK_MAX = 30

# ============================================================
# ANTI-TELEPORT
# ============================================================
TELEPORT_DIST_THRESHOLD = 0.35  # 35% da altura do frame
TELEPORT_TIME_WINDOW = 2.0      # segundos
TELEPORT_SIM_OVERRIDE = 0.78    # se sim ≥ 0.78, permite teleport

# ============================================================
# EMA GUARD
# ============================================================
EMA_SIM_MIN_OFFSET = 0.05   # sim deve ser ≥ thr_in_pid + 0.05
EMA_HSV_MAX_DELTA = 12.0    # ΔHSV máximo permitido (placeholder)
EMA_ALPHA_MIN = 0.03        # α mínimo
EMA_ALPHA_MAX = 0.10        # α máximo

# ============================================================
# NEGATE (penalidade pós-erro)
# ============================================================
NEGATE_PENALTY_MAX = 0.20   # penalidade máxima
NEGATE_DECAY_TIME = 60.0    # decay em 60s

# ============================================================
# CONSTANTES E STRINGS (logs e marcadores)
# ============================================================
LOG_PREFIX_FLOW = "[REID_FLOW]"
LOG_PREFIX_PIDN = "[PID_NEW]"
LOG_PREFIX_PIDU = "[PID_UPD]"
LOG_PREFIX_CREATE_FAIL = "[PID_CREATE_FAIL]"
LOG_PREFIX_HUNG = "[HUNG_RES]"
LOG_PREFIX_CAND = "[REID_CAND]"
LOG_PREFIX_OK = "[REID_OK]"
LOG_PREFIX_TEL = "[ANTI_TEL]"
LOG_PREFIX_NEG = "[NEGATE]"
LOG_PREFIX_BANKSYNC = "[BANK_SYNC]"
LOG_PREFIX_WARN = "[REID_WARN]"


class ReIdentifier:
    """
    Motor de Re-ID – ponte entre tracking temporário e identidades globais.

    Fase 1) Coleta → criação de PID (pessoa nova)
    Fase 2) Re-ID  → matching (Hungarian + MFSS + K + Anti-teleport)
    Fase 3) Consolidação → quando o track some

    API pública (compatível):
      - on_new_track(track_id, embedding, frame_index, bbox=None, density=0.0, frame_height=1080) -> Optional[int]
      - on_track_active(track_id, embedding, frame_index, bbox=None, frame_height=1080) -> None
      - on_track_lost(track_id, frame_index, frame_height=1080) -> Optional[int]
      - get_global_id(track_id) -> Optional[int]
      - get_color(track_id) -> Tuple[int,int,int]
      - is_promoted(track_id) -> bool
      - update_prototype(pid, track_id, embedding, similarity, scale, hsv_mean, frame_index) -> bool
      - reset() -> None
    """

    def __init__(self,
                 gallery: Gallery,
                 min_samples: int = MIN_SAMPLES,
                 match_threshold: float = MATCH_THRESHOLD,
                 ema_momentum: float = EMA_MOMENTUM):
        """
        Parâmetros
        ----------
        gallery : Gallery
            Instância da galeria de embeddings
        min_samples : int
            Mínimo de embeddings para operar (PID/Re-ID)
        match_threshold : float
            Threshold base de cosine similarity (0.65 = conservador)
        ema_momentum : float
            Momentum base para atualização EMA (0.20) – (mantém compatibilidade com IdentityBank)
        """
        # Fonte de embeddings temporais
        self.gallery = gallery

        # Banco de identidades globais
        self.bank = IdentityBank(
            match_threshold=match_threshold,
            ema_momentum=ema_momentum
        )

        # Parâmetros
        self.min_samples = min_samples

        # Mapeamentos e caches
        self._track_to_global: Dict[int, int] = {}                          # track_id → id_global (promovidos)
        self._track_colors: Dict[int, Tuple[int, int, int]] = {}            # cores temporárias para tracks não promovidos
        self._mfss_cache: Dict[int, Dict[str, float]] = {}                  # track_id → {'mfss','mfss2','pid','similarities':[...]}
        self._k_window: Dict[int, Deque[bool]] = {}                         # track_id → janela K (deque de bool)
        self._k_last_update: Dict[int, float] = {}                          # track_id → timestamp último update da janela K
        self._lock_countdown: Dict[int, int] = {}                           # pid → frames restantes em lock
        self._negate_timestamp: Dict[int, float] = {}                       # pid → ts quando aplicou NEGATE
        self._last_position: Dict[int, Tuple[float, float]] = {}            # pid → (cx, cy) última posição conhecida
        self._last_pos_time: Dict[int, float] = {}                          # pid → timestamp da última posição
        self._frame_height = 1080                                           # cache p/ normalização de deslocamento

        # Métricas auxiliares (debug/telemetria)
        self._stats_attempts: Dict[str, int] = defaultdict(int)

    # =========================================================================
    # ORQUESTRADOR: TRACK ATIVO (frame a frame)
    # =========================================================================
    def on_track_active(self,
                        track_id: int,
                        embedding: torch.Tensor,
                        frame_index: int,
                        bbox: Optional[Tuple[float, float, float, float]] = None,
                        frame_height: int = 1080) -> None:
        """
        Chamado frame a frame para tracks ATIVOS.
        Objetivo: se já estiver promovido, tentar EMA guardado do protótipo.

        Observações:
          • Antes estava vazio (lacuna preenchida)
          • Agora detecta ESCALA dinâmica se bbox disponível (scale-aware)
          • Continua leve (não bloqueia pipeline)
        """
        if track_id not in self._track_to_global:
            return

        pid = self._track_to_global[track_id]
        identity = self.bank.get(pid)
        if identity is None:
            return

        # ============================================================
        # EMBEDDING JÁ NORMALIZADO (OSNet faz isso)
        # ============================================================
        emb = embedding.detach().cpu()
        
        # ============================================================
        # DETECTA ESCALA DINÂMICA (PROBLEMA A RESOLVIDO)
        # ============================================================
        track_scale = self._detect_scale(bbox, frame_height)
        
        # Similaridade com protótipo da ESCALA CORRETA (não embedding genérico)
        proto = identity.prototypes.get(track_scale)
        if proto is None:
            proto = identity.embedding
        
        sim = float(torch.matmul(emb.view(1, -1), proto.view(1, -1).T).item())

        updated = self.update_prototype(
            pid=pid,
            track_id=track_id,
            embedding=embedding,
            similarity=sim,
            scale=track_scale,  # AGORA DINÂMICO (escala real)
            hsv_mean=np.zeros(3, dtype=np.float32),
            frame_index=frame_index
        )

        if updated:
            print(f"{LOG_PREFIX_PIDU} pid=P{pid:02d} tid=T{track_id} scale={track_scale} sim={sim:.3f} (EMA guard ok)")
        else:
            # log enxuto; o detalhe do guard está dentro do update_prototype
            pass

    # =========================================================================
    # ORQUESTRADOR: TRACK NOVO (pendente)
    # =========================================================================
    def on_new_track(self,
                     track_id: int,
                     embedding: torch.Tensor,
                     frame_index: int,
                     bbox: Optional[Tuple[float, float, float, float]] = None,
                     density: float = 0.0,
                     frame_height: int = 1080) -> Optional[int]:
        """
        Chamado quando um track PENDING aparece/permanece.
        Orquestra duas rotas mutuamente exclusivas:

          Rota A) _try_reidentify(): se já há identidades no banco → tentar matching
          Rota B) _try_create_pid(): se não há match e o buffer atingiu critério → criar nova identidade

        Regras:
          • Exige pelo menos MIN_SAMPLES no buffer (coleta mínima)
          • Se o banco possui identidades, prioriza tentar Re-ID
          • Se não houver match e a diversidade permitir, cria PID
          • Se nenhum critério fechar, retorna None (continua coletando)
        """
        self._frame_height = frame_height

        # Garante cor temporária para rotulagem até promoção
        if track_id not in self._track_colors:
            self._track_colors[track_id] = self._generate_color()

        # Se já promovido, nada a fazer aqui
        if track_id in self._track_to_global:
            return self._track_to_global[track_id]

        # Coleta mínima
        count = self.gallery.count(track_id)
        if count < self.min_samples:
            print(f"{LOG_PREFIX_FLOW} T{track_id} → COLETA ({count}/{self.min_samples})")
            return None

        # Consolidado médio (para cálculo consistente)
        emb_cons = self.gallery.get(track_id)
        if emb_cons is None:
            print(f"{LOG_PREFIX_WARN} T{track_id} sem embedding consolidado após atingir min_samples.")
            return None

        # 1) Re-ID (matching) se houver identidades no banco
        if self.bank.size() > 0:
            pid, pending = self._try_reidentify(
                track_id=track_id,
                emb_cons=emb_cons,
                frame_index=frame_index,
                bbox=bbox,
                density=density
            )
            if pid is not None:
                # Match confirmado (K/anti-teleport/lock aplicados dentro)
                return pid
            
            if pending:
                # Há um candidato em progresso (MFSS/K-window ainda coletando)
                # Não tentar criar novo PID neste ciclo
                return None

        # 2) Criação de PID (se diversidade permitir)
        pid_new = self._try_create_pid(track_id, frame_index)
        if pid_new is not None:
            # PID criado, mas ainda não é "match re-ID", é identidade nova
            return None

        # Sem match e sem diversidade suficiente: segue coletando
        return None

    # =========================================================================
    # ORQUESTRADOR: TRACK PERDIDO (saiu de cena)
    # =========================================================================
    def on_track_lost(self,
                      track_id: int,
                      frame_index: int,
                      frame_height: int = 1080) -> Optional[int]:
        """
        Chamado quando o track some (ByteTracker.removed_stracks).
        Consolida embeddings da Gallery → IdentityBank.

        Casos:
          • Se já estava promovido: sincroniza banco com o consolidado e limpa caches
          • Se não estava promovido:
              - Se há consolidado suficiente e há match no banco → update()
              - Caso contrário, cria nova identidade (pessoa inédita)
        """
        self._frame_height = frame_height

        # Já promovido?
        if track_id in self._track_to_global:
            pid = self._track_to_global[track_id]
            self._update_bank_from_gallery(track_id, pid, frame_index)
            print(f"{LOG_PREFIX_BANKSYNC} tid=T{track_id} → pid=P{pid:02d} (on_lost)")
            self._cleanup_track_state(track_id)
            return pid

        # Não promovido → tentar consolidar como nova pessoa ou associar
        if not self.gallery.exists(track_id):
            # Não coletou nada útil
            self._cleanup_track_state(track_id)
            return None

        if self.gallery.count(track_id) < self.min_samples:
            count = self.gallery.count(track_id)
            print(f"[ReID] Track T{track_id} perdido com poucos samples ({count})")
            self.gallery.delete(track_id)
            self._cleanup_track_state(track_id)
            return None

        consolidated_emb = self.gallery.get(track_id)
        if consolidated_emb is None:
            self.gallery.delete(track_id)
            self._cleanup_track_state(track_id)
            return None

        # Buscar no banco (sem MFSS/K aqui, pois é consolidação de saída de cena)
        match = self.bank.search(consolidated_emb)
        if match is not None:
            pid, sim = match
            self.bank.update(pid, consolidated_emb, frame_index)
            print(f"[ReID] Track T{track_id} consolidado → RID P{pid:02d} (sim={sim:.3f})")
            self.gallery.delete(track_id)
            self._cleanup_track_state(track_id)
            return pid
        else:
            # Nova identidade
            color = self._track_colors.get(track_id, self._generate_color())
            
            # ============================================================
            # USA PROTÓTIPOS MULTI-ESCALA (novo)
            # ============================================================
            prototypes = self.gallery.get_prototypes(track_id)
            
            pid = self.bank.add(
                embedding=consolidated_emb,
                color=color,
                frame_index=frame_index,
                prototypes=prototypes
            )
            print(f"[ReID] Track T{track_id} → NOVA identidade RID P{pid:02d}")
            self.gallery.delete(track_id)
            self._cleanup_track_state(track_id)
            return pid

    # =========================================================================
    # ROTA A – CRIAÇÃO DE PID (pessoa nova)
    # =========================================================================
    def _try_create_pid(self, track_id: int, frame_index: int) -> Optional[int]:
        """
        Tenta criar uma nova identidade global a partir do buffer do track.
        Regras:
          • Exige diversidade (gallery.check_diversity), com timeout override
          • Gera protótipos multi-escala e grava no banco
        """
        # Diversidade
        has_diversity, diversity_stats = self.gallery.check_diversity(track_id)
        if not has_diversity:
            # Timeout override: se aguardou tempo suficiente, ainda assim cria
            stats = self.gallery.get_stats(track_id)
            if stats and stats.get('age_s', 0.0) > 2.0 and self.gallery.count(track_id) >= 5:
                diversity_stats = {'criterion': 'timeout_override', 'age_s': stats['age_s']}
                has_diversity = True
            else:
                print(f"{LOG_PREFIX_FLOW} T{track_id} → aguardando diversidade")
                return None

        # ============================================================
        # PROTÓTIPOS POR ESCALA (novo - usa medoid)
        # ============================================================
        prototypes = self.gallery.get_prototypes(track_id)
        
        if not prototypes:
            print(f"{LOG_PREFIX_CREATE_FAIL} tid=T{track_id} sem protótipos válidos")
            return None

        proto_scales = list(prototypes.keys())
        fragile = (diversity_stats.get('criterion') == 'timeout_override')

        color = self._track_colors.get(track_id, self._generate_color())
        
        # ============================================================
        # USA PRIMEIRO PROTÓTIPO COMO EMBEDDING PRINCIPAL
        # ============================================================
        main_proto = prototypes[proto_scales[0]]

        # ============================================================
        # GRAVA PROTÓTIPOS MULTI-ESCALA NO BANCO
        # ============================================================
        pid = self.bank.add(
            embedding=main_proto,
            color=color,
            frame_index=frame_index,
            prototypes=prototypes
        )

        # Mapeia track → pid (o track AINDA não é "reidentificado", mas já tem identidade registrada)
        self._track_to_global[track_id] = pid

        proto_str = ",".join(proto_scales)
        frag_str = "YES" if fragile else "NO"
        print(f"{LOG_PREFIX_PIDN} tid=T{track_id}→pid=P{pid:02d} protos:{proto_str} fragile={frag_str}")

        return pid

    # =========================================================================
    # ROTA B – RE-IDENTIFICAÇÃO (matching)
    # =========================================================================
    def _try_reidentify(self,
                        track_id: int,
                        emb_cons: torch.Tensor,
                        frame_index: int,
                        bbox: Optional[Tuple[float, float, float, float]],
                        density: float
                        ) -> Tuple[Optional[int], bool]:
        """
        Tenta re-identificar um track pendente contra o banco.
        Pipeline:
            Hungarian (custo multi-fator) → MFSS/APPT → K-window → Anti-teleport → lock
        Retorna:
            (pid, pending)
            pid: ID global se confirmado
            pending: True se ainda em confirmação (K-window não completo)
        """
        
        t0_reid = time.perf_counter()
        
        match = self._hungarian_search(
            track_id=track_id,
            embedding=emb_cons,
            bbox=bbox,
            frame_index=frame_index
        )

        if match is None:
            # Sem candidato com custo aceitável
            t1_reid = time.perf_counter()
            reid_ms = (t1_reid - t0_reid) * 1000.0
            print(f"[REID_TIME] tid=T{track_id} result=NO_CANDIDATE time={reid_ms:.1f}ms")
            return None, False

        pid, similarity, cost = match

        # MFSS + APPT (suavização + margem top1-top2)
        mfss_pass, mfss_stats = self._update_mfss(track_id, pid, similarity, emb_cons)

        if not mfss_pass:
            print(f"{LOG_PREFIX_CAND} tid=T{track_id} pid=P{pid:02d} pass=NO reason={mfss_stats.get('reason','unknown')}")
            t1_reid = time.perf_counter()
            reid_ms = (t1_reid - t0_reid) * 1000.0
            print(f"[REID_TIME] tid=T{track_id} pid=P{pid:02d} result=MFSS_FAIL time={reid_ms:.1f}ms")
            return None, False

        # K-window
        k_confirmed, k_stats = self._update_k_window(track_id, True)
        k_current = k_stats['positives']
        k_total = K_WINDOW_SIZE
        mfss = mfss_stats['mfss']
        margin = mfss_stats['margin']
        print(f"{LOG_PREFIX_CAND} tid=T{track_id} pid=P{pid:02d} K={k_current}/{k_total} MFSS={mfss:.2f} margin={margin:.2f} pass=YES")

        if not k_confirmed:
            t1_reid = time.perf_counter()
            reid_ms = (t1_reid - t0_reid) * 1000.0
            print(f"[REID_TIME] tid=T{track_id} pid=P{pid:02d} result=K_PENDING time={reid_ms:.1f}ms")
            return None, True

        # Anti-teleport
        teleport_ok = self._validate_antiteleport(pid, bbox, frame_index, similarity)
        if not teleport_ok:
            self._apply_negate(pid)
            # Reset janela K
            if track_id in self._k_window:
                self._k_window[track_id].clear()
            t1_reid = time.perf_counter()
            reid_ms = (t1_reid - t0_reid) * 1000.0
            print(f"[REID_TIME] tid=T{track_id} pid=P{pid:02d} result=TELEPORT_FAIL time={reid_ms:.1f}ms")
            return None, False

        # Confirmação final → promove
        self._track_to_global[track_id] = pid

        # Lock adaptativo
        lock_frames = self._compute_lock(pid, bbox, density)
        self._lock_countdown[pid] = lock_frames

        # Atualiza posição (anti-teleport futuro)
        if bbox is not None:
            cx = (bbox[0] + bbox[2]) / 2.0
            cy = (bbox[1] + bbox[3]) / 2.0
            self._last_position[pid] = (cx, cy)
            self._last_pos_time[pid] = time.time()

        print(f"{LOG_PREFIX_OK} tid=T{track_id} pid=P{pid:02d} K={k_current}/{k_total} lock={lock_frames} MFSS={mfss:.2f}")
        
        t1_reid = time.perf_counter()
        reid_ms = (t1_reid - t0_reid) * 1000.0
        print(f"[REID_TIME] tid=T{track_id} pid=P{pid:02d} result=SUCCESS time={reid_ms:.1f}ms")
        
        return pid, False

    # =========================================================================
    # HUNGARIAN SEARCH (custo multi-fator + multi-escala)
    # =========================================================================
    def _hungarian_search(self,
                          track_id: int,
                          embedding: torch.Tensor,
                          bbox: Optional[Tuple[float, float, float, float]],
                          frame_index: int) -> Optional[Tuple[int, float, float]]:
        """
        Busca no banco com custo multi-fator + matching multi-escala:

          C = w_e·(1-sim) + w_t·pen_t + w_z·pen_z + w_s·pen_s + w_h·(1-h) + NEGATE

        Usa protótipo da escala correta (NEAR com NEAR, FAR com FAR).

        Retorna
        -------
        (pid, similarity, cost) | None
        """
        # Pesos
        w_e = 0.80   # embedding similarity
        w_t = 0.05   # penalidade tempo
        w_z = 0.00   # zona (opcional, mantido 0.0)
        w_s = 0.05   # espacial (anti-teleport como custo)
        w_h = 0.10   # health

        if self.bank.size() == 0:
            return None

        # ============================================================
        # DETECTA ESCALA DO TRACK ATUAL
        # ============================================================
        track_scale = self._detect_scale(bbox, self._frame_height)

        best_pid = None
        best_sim = -1.0
        best_cost = float('inf')

        # ============================================================
        # EMBEDDING JÁ NORMALIZADO (OSNet faz isso)
        # ============================================================
        emb = embedding.detach().cpu().view(1, -1)  # (1, 512)

        # Varre todas identidades – custo manual
        for pid, identity in self.bank.identities.items():
            # ============================================================
            # USA PROTÓTIPO DA ESCALA CORRETA (novo)
            # ============================================================
            proto = identity.prototypes.get(track_scale)
            
            if proto is None:
                # Fallback: embedding principal
                proto = identity.embedding
            
            if proto is None or proto.numel() == 0:
                continue
            
            proto = proto.view(1, -1)  # (1, 512)
            sim = float(torch.matmul(emb, proto.T).item())

            # penalidade tempo (proxy simples por frames desde visto)
            time_since = max(0, frame_index - identity.last_seen_frame)
            pen_t = min(time_since / 150.0, 1.0)

            # penalidade zona (placeholder)
            pen_z = 0.0

            # penalidade espacial (salto grande entre última pos do pid e bbox atual)
            pen_s = 0.0
            if bbox is not None and pid in self._last_position:
                cx = (bbox[0] + bbox[2]) / 2.0
                cy = (bbox[1] + bbox[3]) / 2.0
                last_cx, last_cy = self._last_position[pid]
                dist = math.sqrt((cx - last_cx) ** 2 + (cy - last_cy) ** 2)
                dist_norm = dist / max(self._frame_height, 1)
                if dist_norm > TELEPORT_DIST_THRESHOLD:
                    pen_s = min(dist_norm / TELEPORT_DIST_THRESHOLD, 1.0)

            # penalidade health (quanto mais baixo, maior o custo)
            health = identity.health if identity.health is not None else 1.0
            pen_h = 1.0 - health

            negate_pen = self._get_negate_penalty(pid)

            cost = (w_e * (1.0 - sim) +
                    w_t * pen_t +
                    w_z * pen_z +
                    w_s * pen_s +
                    w_h * pen_h +
                    negate_pen)

            if cost < best_cost:
                best_cost = cost
                best_sim = sim
                best_pid = pid

        # valida threshold absoluto mínimo de similaridade
        if best_pid is None or best_sim < 0.75:
            return None

        print(f"{LOG_PREFIX_HUNG} assign: T{track_id}→P{best_pid:02d} cost={best_cost:.2f} sim={best_sim:.2f} scale={track_scale}")
        return best_pid, best_sim, best_cost

    def _detect_scale(self, 
                     bbox: Optional[Tuple[float, float, float, float]], 
                     frame_height: int) -> str:
        """
        Detecta escala do track baseado em bbox_height / frame_height.
        
        Thresholds (alinhado com detector.py):
        - NEAR: ratio > 0.45
        - MID: ratio > 0.20
        - FAR: ratio > 0.10
        - DESC: ratio <= 0.10
        
        Parâmetros
        ----------
        bbox : tuple | None
            (x1, y1, x2, y2)
        frame_height : int
            Altura do frame
        
        Retorna
        -------
        scale : str
            "NEAR", "MID", "FAR", "DESC"
        """
        if bbox is None:
            return "MID"  # default neutro
        
        x1, y1, x2, y2 = bbox
        bbox_height = y2 - y1
        ratio = bbox_height / max(frame_height, 1)
        
        # ============================================================
        # CLASSIFICAÇÃO (mesmo threshold do detector)
        # ============================================================
        if ratio > 0.45:
            return "NEAR"
        elif ratio > 0.20:
            return "MID"
        elif ratio > 0.10:
            return "FAR"
        else:
            return "DESC"

    # =========================================================================
    # MFSS + APPT (usando top-2 reais)
    # =========================================================================
    def _update_mfss(self,
                     track_id: int,
                     pid: int,
                     similarity: float,
                     emb_cons: torch.Tensor) -> Tuple[bool, Dict]:
        """
        Atualiza MFSS (EMA) e checa APPT (margem entre top-1 e top-2).
        Para obter um top-2 real, computamos similaridade contra TODOS PIDs.
        """
        # Inicialização do cache se necessário
        if track_id not in self._mfss_cache:
            self._mfss_cache[track_id] = {
                'mfss': similarity,
                'mfss2': 0.0,
                'pid': pid,
                'similarities': []
            }

        cache = self._mfss_cache[track_id]

        # Atualiza MFSS com a similaridade do candidato principal
        prev_mfss = cache['mfss']
        mfss = MFSS_BETA * prev_mfss + (1.0 - MFSS_BETA) * similarity
        cache['mfss'] = mfss

        # ---- top-2 reais (varre todas identidades para esse embedding)
        sim1, sim2 = self._compute_top2_sims(emb_cons)
        # atualiza MFSS2 com sim2
        prev2 = cache.get('mfss2', 0.0)
        mfss2 = MFSS_BETA * prev2 + (1.0 - MFSS_BETA) * sim2
        cache['mfss2'] = mfss2

        # Threshold adaptativo por PID
        thr_in_pid = self._compute_adaptive_threshold(pid)

        margin = mfss - mfss2
        passed = (mfss >= thr_in_pid) and (margin >= APPT_MARGIN)

        stats = {
            'mfss': mfss,
            'mfss2': mfss2,
            'margin': margin,
            'thr_in_pid': thr_in_pid,
            'reason': 'ok' if passed else ('low_mfss' if mfss < thr_in_pid else 'low_margin')
        }
        return passed, stats

    def _compute_top2_sims(self, embedding: torch.Tensor) -> Tuple[float, float]:
        """
        Computa top-2 similaridades do embedding contra todos os PIDs do banco.
        Não aplica threshold aqui para termos um APPT realista.
        """
        if self.bank.size() == 0:
            return 0.0, 0.0
        
        # ============================================================
        # EMBEDDING JÁ NORMALIZADO (OSNet faz isso)
        # ============================================================
        emb = embedding.detach().cpu().view(1, -1)  # (1, 512)
        
        sims = []
        for pid, identity in self.bank.identities.items():
            sim = float(torch.matmul(emb, identity.embedding.view(1, -1).T).item())
            sims.append(sim)
        
        if not sims:
            return 0.0, 0.0
        
        sims_sorted = sorted(sims, reverse=True)
        sim1 = sims_sorted[0]
        sim2 = sims_sorted[1] if len(sims_sorted) > 1 else 0.0
        return sim1, sim2

    def _compute_adaptive_threshold(self, pid: int) -> float:
        """
        Threshold adaptativo por PID. Placeholder conservador:
            thr_in_pid = max(0.62, 0.65)
        (Gancho para usar histórico μ/σ por PID no futuro.)
        """
        return max(0.62, 0.65)

    # =========================================================================
    # K CONFIRMAÇÕES (janela deslizante com timeout)
    # =========================================================================
    def _update_k_window(self,
                         track_id: int,
                         passed: bool) -> Tuple[bool, Dict]:
        """
        Atualiza janela K (últimos 8 frames). Mantém progresso se Δt < 2s.
        """
        current_time = time.time()

        if track_id not in self._k_window:
            self._k_window[track_id] = deque(maxlen=K_WINDOW_SIZE)
            self._k_last_update[track_id] = current_time

        last_update = self._k_last_update[track_id]
        dt = current_time - last_update
        if dt > K_RESET_TIMEOUT:
            self._k_window[track_id].clear()

        self._k_window[track_id].append(passed)
        self._k_last_update[track_id] = current_time

        window = self._k_window[track_id]
        positives = sum(1 for x in window if x)
        confirmed = positives >= K_REQUIRED

        stats = {'positives': positives, 'window_size': len(window), 'confirmed': confirmed}
        return confirmed, stats

    # =========================================================================
    # ANTI-TELEPORT (Δt + distância)
    # =========================================================================
    def _validate_antiteleport(self,
                               pid: int,
                               bbox: Optional[Tuple[float, float, float, float]],
                               frame_index: int,
                               similarity: float) -> bool:
        """
        Reprova re-ID se salto espacial impossível for detectado:
          • Δt < 2s E dist_norm > 0.35H E sim < 0.78
        """
        if bbox is None or pid not in self._last_position:
            # Sem referência de posição – aceita
            print(f"{LOG_PREFIX_TEL} pid=P{pid:02d} dist=?H ok=YES (no_ref)")
            return True

        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        last_cx, last_cy = self._last_position[pid]
        dist = math.sqrt((cx - last_cx) ** 2 + (cy - last_cy) ** 2)
        dist_norm = dist / max(self._frame_height, 1)

        last_ts = self._last_pos_time.get(pid, 0.0)
        dt_s = time.time() - last_ts if last_ts > 0 else float('inf')

        if dist_norm > TELEPORT_DIST_THRESHOLD and dt_s < TELEPORT_TIME_WINDOW:
            if similarity >= TELEPORT_SIM_OVERRIDE:
                print(f"{LOG_PREFIX_TEL} pid=P{pid:02d} dist={dist_norm:.2f}H Δt={dt_s:.2f}s ok=YES (override sim={similarity:.2f})")
                return True
            else:
                print(f"{LOG_PREFIX_TEL} pid=P{pid:02d} dist={dist_norm:.2f}H Δt={dt_s:.2f}s ok=REJECT")
                return False

        print(f"{LOG_PREFIX_TEL} pid=P{pid:02d} dist={dist_norm:.2f}H Δt={dt_s:.2f}s ok=YES")
        return True

    # =========================================================================
    # LOCK ADAPTATIVO
    # =========================================================================
    def _compute_lock(self,
                      pid: int,
                      bbox: Optional[Tuple[float, float, float, float]],
                      density: float) -> int:
        """
        lock = clamp(base + f_speed·Δpos_norm + f_crowd·density, 8, 30)
        """
        lock = LOCK_BASE
        if bbox is not None and pid in self._last_position:
            cx = (bbox[0] + bbox[2]) / 2.0
            cy = (bbox[1] + bbox[3]) / 2.0
            last_cx, last_cy = self._last_position[pid]
            dist = math.sqrt((cx - last_cx) ** 2 + (cy - last_cy) ** 2)
            dist_norm = dist / max(self._frame_height, 1)
            lock += LOCK_SPEED_FACTOR * dist_norm

        lock += LOCK_CROWD_FACTOR * max(0.0, density)
        lock = max(LOCK_MIN, min(LOCK_MAX, int(round(lock))))
        return lock

    # =========================================================================
    # NEGATE (penalidade pós-erro)
    # =========================================================================
    def _apply_negate(self, pid: int):
        self._negate_timestamp[pid] = time.time()
        print(f"{LOG_PREFIX_NEG} pid=P{pid:02d} penalty=0.20 (decay 60s)")

    def _get_negate_penalty(self, pid: int) -> float:
        if pid not in self._negate_timestamp:
            return 0.0
        dt = time.time() - self._negate_timestamp[pid]
        penalty = NEGATE_PENALTY_MAX * math.exp(-dt / NEGATE_DECAY_TIME)
        return penalty

    # =========================================================================
    # EMA UPDATE (guardado) – mantém assinatura idêntica
    # =========================================================================
    def update_prototype(self,
                         pid: int,
                         track_id: int,
                         embedding: torch.Tensor,
                         similarity: float,
                         scale: str,
                         hsv_mean: np.ndarray,
                         frame_index: int) -> bool:
        """
        Atualiza protótipo do PID com guards SCALE-AWARE:

          • lock countdown == 0
          • sim ≥ thr_in_pid + 0.05 (comparado contra protótipo da ESCALA CORRETA)
          • ΔHSV ≤ 12 (placeholder)
          • sem conflito com outro PID (placeholder)

        Em seguida, aplica update no banco com EMA por escala (IdentityBank.update_prototype()).
        """
        identity = self.bank.get(pid)
        if identity is None:
            return False
        
        # GUARD 1 – lock
        lock = self._lock_countdown.get(pid, 0)
        if lock > 0:
            self._lock_countdown[pid] = lock - 1
            print(f"[EMA_UPD] pid=P{pid:02d} scale={scale} sim={similarity:.2f} guards_ok=NO (lock={lock})")
            return False

        # GUARD 2 – similaridade mínima contra PROTÓTIPO DA ESCALA (PROBLEMA B RESOLVIDO)
        # Se a escala tem protótipo, valida contra aquele. Senão, usa threshold base.
        if scale in identity.prototypes:
            # Recomputa sim contra protótipo da escala para validação mais precisa
            proto_scale = identity.prototypes[scale]
            emb = embedding.detach().cpu().view(1, -1)
            similarity_scale = float(torch.matmul(emb, proto_scale.view(1, -1).T).item())
        else:
            # Primeira vez vendo essa escala, usa baseline
            similarity_scale = similarity
        
        thr_in_pid = self._compute_adaptive_threshold(pid)
        sim_min = thr_in_pid + EMA_SIM_MIN_OFFSET
        if similarity_scale < sim_min:
            print(f"[EMA_UPD] pid=P{pid:02d} scale={scale} sim={similarity_scale:.2f} guards_ok=NO (sim<{sim_min:.2f})")
            return False

        # GUARD 3 – ΔHSV (placeholder: aceita)
        # GUARD 4 – Oclusão (placeholder: aceita)
        # GUARD 5 – Conflito (placeholder: aceita)

        # α adaptativo por health
        health = self.bank.get_health(pid) or 1.0
        alpha = max(EMA_ALPHA_MIN, min(EMA_ALPHA_MAX, 0.10 - 0.05 * health))

        # Atualiza embedding principal (EMA no embedding geral) + protótipo por escala
        self.bank.update(pid, embedding, frame_index)
        self.bank.update_prototype(pid, scale, embedding, alpha)

        print(f"[EMA_UPD] pid=P{pid:02d} scale={scale} sim={similarity_scale:.2f} α={alpha:.2f} guards_ok=YES")
        return True

    # =========================================================================
    # SINCRONIZAÇÃO BANCO ← GALLERY (quando já promovido)
    # =========================================================================
    def _update_bank_from_gallery(self, track_id: int, id_global: int, frame_index: int) -> None:
        """
        Consolida embeddings multi-escala da gallery → IdentityBank (PROBLEMA C RESOLVIDO).
        
        Estratégia: se há protótipos por escala, atualiza cada um individualmente.
        Senão, usa embedding médio (fallback).
        """
        if not self.gallery.exists(track_id):
            return
        
        # Tenta consolidar com protótipos multi-escala
        prototypes_multi = self.gallery.get_prototypes(track_id)
        
        if prototypes_multi:
            # Consolida cada escala separadamente (scale-aware)
            for scale, proto in prototypes_multi.items():
                self.bank.update_prototype(id_global, scale, proto, alpha=0.15)
            print(f"[BANK_CONS] tid=T{track_id} pid=P{id_global:02d} consolidated {len(prototypes_multi)} scales")
        else:
            # Fallback: embedding médio (compatibilidade)
            consolidated_emb = self.gallery.get(track_id)
            if consolidated_emb is not None:
                self.bank.update(id_global, consolidated_emb, frame_index)
                print(f"[BANK_CONS] tid=T{track_id} pid=P{id_global:02d} consolidated via avg (no protos)")

    # =========================================================================
    # QUERIES (públicos)
    # =========================================================================
    def get_global_id(self, track_id: int) -> Optional[int]:
        """Retorna id_global de um track (se já promovido)"""
        return self._track_to_global.get(track_id)

    def get_color(self, track_id: int) -> Tuple[int, int, int]:
        """
        Retorna cor BGR para um track:
          - se promovido → cor persistente do banco
          - senão → cor temporária do track
        """
        if track_id in self._track_to_global:
            pid = self._track_to_global[track_id]
            color = self.bank.get_color(pid)
            if color is not None:
                return color

        if track_id not in self._track_colors:
            self._track_colors[track_id] = self._generate_color()
        return self._track_colors[track_id]

    def is_promoted(self, track_id: int) -> bool:
        """Verifica se track já tem id_global"""
        return track_id in self._track_to_global

    # =========================================================================
    # HOUSEKEEPING
    # =========================================================================
    def _cleanup_track_state(self, track_id: int) -> None:
        """Limpa estados auxiliares/caches de um track após consolidar/remover."""
        if track_id in self._track_to_global:
            del self._track_to_global[track_id]
        if track_id in self._track_colors:
            del self._track_colors[track_id]
        if track_id in self._mfss_cache:
            del self._mfss_cache[track_id]
        if track_id in self._k_window:
            del self._k_window[track_id]
        if track_id in self._k_last_update:
            del self._k_last_update[track_id]
        self.gallery.delete(track_id)

    def reset(self) -> None:
        """Reset completo do sistema de Re-ID"""
        self.gallery.clear_all()
        self.bank.clear()
        self._track_to_global.clear()
        self._track_colors.clear()
        self._mfss_cache.clear()
        self._k_window.clear()
        self._k_last_update.clear()
        self._lock_countdown.clear()
        self._negate_timestamp.clear()
        self._last_position.clear()
        self._last_pos_time.clear()

    def reset_dynamic_cache(self) -> None:
        """
        Reseta apenas caches dinâmicos entre loops de vídeo.
        Mantém banco de identidades intacto (para Re-ID em novo loop).
        
        DEVE ser chamado em stream.py quando novo loop começa.
        Evita que histerese e penalidades velhas contaminem novo loop.
        
        ============================================================
        IMPORTANTE: O que é resetado vs preservado
        ============================================================
        RESETADO (porque depende de track_id que muda a cada loop):
        - _track_to_global: mapeamento track_id → id_global (novo loop = novos track_ids)
        - _track_colors: cores temporárias de tracking
        - gallery: buffer temporal de embeddings (está associado a track_id ativo)
        - Todos os caches dinâmicos (MFSS, K-window, NEGATE, posição, etc)
        
        PRESERVADO (identidades globais consolidadas - válidas entre loops):
        - IdentityBank (self.bank): contém pessoas já identificadas com protótipos
          └─ Cada identidade: id_global, protótipos multi-escala, cor, health, TTL
          └─ Estas NÃO expiram imediatamente - TTL até 5 minutos
          └─ Permite Re-ID em novo loop: mesma pessoa = mesmo PID
        
        Fluxo esperado:
        1. Loop 1: pessoa aparece → gallery coleta embeddings → track some → consolida no banco (PID=1)
        2. Loop 2: pessoa reaparece → busca no banco → encontra PID=1 → reutiliza
        3. Loop 5 minutos depois: se pessoa não reapareceu → TTL expira → remove do banco
        """
        self._mfss_cache.clear()                 # similaridade temporal (MFSS)
        self._k_window.clear()                   # janela K (confirmações)
        self._k_last_update.clear()              # timestamp última atualização K
        self._lock_countdown.clear()             # countdown de lock adaptativo
        self._negate_timestamp.clear()           # penalidades de erro (muito importante!)
        self._last_position.clear()              # cache de posição para anti-teleport
        self._last_pos_time.clear()              # timestamp última posição
        self._track_to_global.clear()            # mapping track → id_global (NOVO cada loop)
        self._track_colors.clear()               # colors temporárias de tracking (NOVO cada loop)
        self.gallery.clear_all()                 # gallery = buffer TEMPORÁRIO (depende de track_id)
        bank_size = self.bank.size()
        print(f"{LOG_PREFIX_BANKSYNC} Dynamic cache resetado para novo loop (banco com {bank_size} identidades preservadas)")

    # =========================================================================
    # UTILITÁRIOS
    # =========================================================================
    @staticmethod
    def _generate_color() -> Tuple[int, int, int]:
        """Gera cor BGR aleatória (consistente com seu padrão atual)."""
        return (
            random.randint(60, 255),
            random.randint(60, 255),
            random.randint(60, 255),
        )