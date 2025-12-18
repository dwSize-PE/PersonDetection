"""
Stream — Pipeline principal de detecção + tracking + Re-ID

Responsabilidades:
- Orquestrar detector thread (YOLO assíncrono)
- ByteTracker mantém IDs estáveis via Kalman (fonte de verdade do tracking)
- Consolidar identidades via LOST (ByteTracker = gatilho único)
- Alimentar Re-ID thread com tracks ativos (ACTIVE + PENDING)
- Propagar metadata crítica (bbox, density, frame_height, temp_lost)
- Renderizar overlay com cores persistentes + estatísticas
- Manutenção do banco (TTL + health decay + LRU)

Fluxo:
1. YOLO detecta → 2. ByteTracker rastreia → 3. LOST consolida → 
4. Re-ID thread processa → 5. Overlay renderiza
"""

import cv2
import time
import os
import threading
from typing import List, Dict

import app.detector
from app.tracker.tracker_wrapper import ByteTrackWrapper
from app.reid.gallery import Gallery
from app.osnet.osnet_model import OsNetEmbedder
from app.reid.reidentifier import ReIdentifier
from app.reid.embedder import ReIDEmbedderThread

VIDEO_PATH = os.path.join("data", "video.mp4")

# ============================================================
# MANUTENÇÃO DO BANCO (intervalo em frames)
# ============================================================
BANK_TICK_INTERVAL = 30  # tick a cada 30 frames (~1s @ 30fps)


def start_stream():
    """
    Inicia pipeline completo de detecção + tracking + Re-ID.
    """
    
    # =========================================================================
    # 1) DETECTOR THREAD (YOLO)
    # =========================================================================
    detector_thread = threading.Thread(target=app.detector.detector_thread, daemon=True)
    detector_thread.start()
    print("✅ Detector thread iniciada (YOLO)")

    # =========================================================================
    # 2) BYTETRACKER (Kalman Filter + Two-stage matching)
    # =========================================================================
    tracker = ByteTrackWrapper(
        track_thresh=0.5,      # threshold mínimo para high-confidence detections
        track_buffer=30,       # buffer de frames para tracks perdidos
        match_thresh=0.8,      # threshold IoU para matching
        frame_rate=30
    )
    print("✅ ByteTracker inicializado")

    # =========================================================================
    # 3) RE-ID INFRASTRUCTURE
    # =========================================================================
    lock_frame = threading.Lock()
    lock_tracks = threading.Lock()
    
    shared_frame = {'frame': None, 'frame_index': 0}
    shared_tracks = {'tracks': []}

    gallery = Gallery(buffer_size=10)
    
    osnet = OsNetEmbedder(
        weight_path="models/best_model.pth",
        device="auto",
        half=False
    )
    
    reid = ReIdentifier(
        gallery=gallery,
        min_samples=8,          # 8 embeddings = ~0.25s a 30fps
        match_threshold=0.65,   # cosine similarity threshold (conservador)
        ema_momentum=0.20
    )

    embedder = ReIDEmbedderThread(
        gallery=gallery,
        osnet=osnet,
        reid=reid,
        lock_frame=lock_frame,
        shared_frame=shared_frame,
        lock_tracks=lock_tracks,
        shared_tracks=shared_tracks,
        sleep_ms=5
    )
    embedder.start()
    print("✅ Re-ID thread iniciada (OSNet + IdentityBank)")

    # =========================================================================
    # 4) ATRIBUTOS DE CONTROLE (idempotência + suavização)
    # =========================================================================
    lost_fired = set()       # garante on_track_lost dispara apenas 1x por tid
    density_smooth = 0.0     # densidade suavizada via EMA (α=0.2)

    # =========================================================================
    # 5) MAIN LOOP (Video streaming + tracking + visualization)
    # =========================================================================
    # Buffer de frames: guarda (frame_index: frame)
    # Permite recuperar o frame EXATO que o YOLO processou (mesmo com atraso)
    # 
    # DIMENSIONAMENTO:
    # - BUFFER_MAX_SIZE = (atraso_detector_em_segundos × FPS) + margem
    # - 120 frames @ 30fps = 4 segundos de atraso máximo suportado
    # - Se detector levar > 4s, frames expiram → [CRITICAL_SYNC]
    # - Se isso ocorre: aumentar buffer OU otimizar detector
    frame_buffer = {} 
    BUFFER_MAX_SIZE = 120  # Aumentado de 60 para 120 (suporta atraso maior do detector)
    
    while True:
        cap = cv2.VideoCapture(VIDEO_PATH)
        if not cap.isOpened():
            print("❌ Erro ao abrir vídeo:", VIDEO_PATH)
            time.sleep(1)
            continue

        # =========================================================================
        # RESET DE ESTADO ENTRE LOOPS (IMPORTANTE!)
        # =========================================================================
        # Reseta caches do detector (histerese de escala)
        app.detector.reset_detector_state()
        
        # Reseta caches dinâmicos do Re-ID (penalidades, MFSS, K-window)
        # Mantém banco de identidades para Re-ID em novo loop
        reid.reset_dynamic_cache()
        
        # Limpa tracks temporários (pendências de tracking)
        lost_fired.clear()
        
        # Log informativo sobre dimensionamento
        fps = 30.0
        buffer_time_seconds = BUFFER_MAX_SIZE / fps
        bank_count = reid.bank.size()
        print(f"[STREAM_RESET] Caches e estado resetados para novo loop")
        print(f"[CONFIG] BUFFER: {BUFFER_MAX_SIZE} frames = {buffer_time_seconds:.1f}s @ {fps}fps | BANK: {bank_count} identidades (TTL 30-300s)")
        
        frame_interval = 1.0 / fps
        next_frame_time = time.time()
        frame_idx = 0

        print(f"🎬 Iniciando vídeo (FPS={fps:.1f})")
        video_start_time = time.time()

        while True:
            ret, raw_frame = cap.read()
            if not ret:
                # ============================================================
                # TIMER: fim do processamento
                # ============================================================
                video_end_time = time.time()
                video_duration = video_end_time - video_start_time
                
                print("=" * 60)
                print(f"[VIDEO_END] Tempo total de processamento: {video_duration:.2f}s")
                print(f"[VIDEO_END] Frames processados: {frame_idx}")
                print(f"[VIDEO_END] FPS medio: {frame_idx / video_duration:.2f}")
                print("=" * 60)
                print("LOOP Reiniciando video...")
                break

            # ============================================================
            # BUFFER MANAGEMENT
            # ============================================================
            # 1. Adiciona frame atual ao buffer e envia para detectar
            frame_buffer[frame_idx] = raw_frame.copy()
            app.detector.submit_frame(raw_frame, frame_idx)
            
            # Limpeza de segurança do buffer
            if len(frame_buffer) > BUFFER_MAX_SIZE:
                # Remove frames muito antigos (safety net)
                oldest = min(frame_buffer.keys())
                del frame_buffer[oldest]
                
            # ============================================================
            # RECUPERA RESULTADO SINCRONIZADO
            # ============================================================
            # Não bloqueia loop de leitura da câmera, mas só atualiza tracker
            # se houver resultado do detector. 
            # Se detector for lento, o display acompanha a velocidade do detector (sync)
            
            det_result = app.detector.get_result()
            
            if det_result is not None:
                detections, det_frame_idx = det_result
                
                # ============================================================
                # VALIDAÇÃO OBRIGATÓRIA #1: Frame deve estar no buffer
                # Se não estiver, descarta resultado (detector muito lento)
                # ============================================================
                if det_frame_idx not in frame_buffer:
                    print(f"[CRITICAL_SYNC] Frame {det_frame_idx} expired from buffer! "
                          f"buffer_size={len(frame_buffer)} buffer_range=[{min(frame_buffer.keys()) if frame_buffer else 'empty'}-{max(frame_buffer.keys()) if frame_buffer else 'empty'}] "
                          f"max_size={BUFFER_MAX_SIZE} (detector muito lento)")
                    print(f"[CRITICAL_SYNC] Discarding detection result to maintain sync")
                    continue  # Pula este update – aguarda próximo detector result
                
                # Frame está válido no buffer
                frame = frame_buffer[det_frame_idx]
                
                # Limpa frames mais antigos que este (já processados ou pulados)
                keys_to_remove = [k for k in frame_buffer.keys() if k < det_frame_idx]
                for k in keys_to_remove:
                    del frame_buffer[k]

                frame_h, frame_w = frame.shape[:2]

                # ============================================================
                # STEP 2: BYTETRACKER UPDATE (estados granulares)
                # ============================================================
                tracks_by_state = tracker.update(detections, (frame_h, frame_w), external_frame_id=det_frame_idx)
                
                active_tracks = tracks_by_state['active']
                pending_tracks = tracks_by_state['pending']
                temp_lost_tracks = tracks_by_state['temp_lost']
                lost_tracks = tracks_by_state['lost']
                
                # Formato: {'track_id', 'bbox', 'score', 'keypoints', 'scale', 'had_pad', ...}

                # ============================================================
                # STEP 2.5: CONSOLIDAÇÃO VIA LOST (ByteTracker = fonte verdade)
                # ============================================================
                if lost_tracks:
                    ids_str = ",".join([f"T{t['track_id']}" for t in lost_tracks[:5]])
                    if len(lost_tracks) > 5:
                        ids_str += "..."
                    print(f"[STREAM_LOST] f={frame_idx} n={len(lost_tracks)} ids=[{ids_str}]")
                    
                    for track in lost_tracks:
                        tid = track['track_id']
                        
                        # Idempotência (dispara apenas uma vez por track_id)
                        if tid not in lost_fired:
                            reid.on_track_lost(tid, frame_idx, frame_h)
                            lost_fired.add(tid)
                
                # Limpeza periódica do set (evita crescimento infinito)
                if frame_idx % 1000 == 0:
                    lost_fired.clear()

                # ============================================================
                # STEP 3: ALIMENTA RE-ID THREAD (ACTIVE + PENDING)
                # ============================================================
                all_active = active_tracks + pending_tracks
                
                density_instant = len(all_active) / ((frame_h * frame_w) / 1000.0)
                density_smooth = 0.8 * density_smooth + 0.2 * density_instant
                
                # CRUCIAL: Atualiza shared_frame com o frame SINCRONIZADO
                # O Embedder vai ler este frame, garantindo que o crop bata com a bbox
                with lock_frame:
                    shared_frame['frame'] = frame.copy()
                    shared_frame['frame_index'] = det_frame_idx

                with lock_tracks:
                    shared_tracks['tracks'] = all_active
                    shared_tracks['temp_lost'] = temp_lost_tracks
                    shared_tracks['frame_index'] = det_frame_idx  # NOVO: sincronização mútua
                    shared_tracks['density'] = density_smooth
                    shared_tracks['frame_height'] = frame_h
                
                # ============================================================
                # STEP 4: PROCESSA PENDING PARA RE-ID (Hungarian + MFSS + K)
                # ============================================================
                density = tracker.get_density()
                
                for track in pending_tracks:
                    track_id = track['track_id']
                    
                    # Verifica se já tem PID ou se está em processo de Re-ID
                    if reid.is_promoted(track_id):
                        continue  # já promovido
                    
                    # Verifica se tem embeddings suficientes para tentar Re-ID
                    if gallery.count(track_id) < 8:
                        continue  # ainda coletando
                    
                    # Tenta Re-ID (Hungarian + MFSS + K confirmações)
                    # (embedder thread já faz isso em on_new_track)
                    # Aqui apenas garantimos que track está visível

                # ============================================================
                # STEP 5: OVERLAY (desenha tracks + IDs + cores)
                # ============================================================
                for track in all_active:
                    track_id = track['track_id']
                    bbox = track['bbox']
                    x1, y1, x2, y2 = bbox

                    # ============================================================
                    # OBTÉM ID_GLOBAL (se promovido) ou usa track_id temporário
                    # ============================================================
                    id_global = embedder.get_global_id(track_id)
                    
                    if id_global is not None:
                        # Re-ID bem-sucedido — mostra RID (id_global)
                        label = f"RID:P{id_global:02d}"
                    else:
                        # Ainda em tracking temporário
                        label = f"T:{track_id}"

                    # ============================================================
                    # COR PERSISTENTE (do banco se promovido, ou temporária)
                    # ============================================================
                    color = embedder.get_color(track_id)

                    # ============================================================
                    # DESENHA BBOX + LABEL
                    # ============================================================
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    
                    # Background para texto (melhor legibilidade)
                    (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(frame, (int(x1), int(y1) - text_h - 8), (int(x1) + text_w, int(y1)), color, -1)
                    cv2.putText(frame, label, (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    # ============================================================
                    # MOSTRA ESCALA (debug)
                    # ============================================================
                    scale = track.get('scale', 'UNKNOWN')
                    scale_label = f"{scale[0]}"  # primeira letra (N/M/F/D)
                    cv2.putText(frame, scale_label, (int(x2) - 15, int(y1) + 15), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

                # ============================================================
                # STEP 6: INFO OVERLAY (estatísticas)
                # ============================================================
                n_active = len(active_tracks)
                n_pending = len(pending_tracks)
                n_temp_lost = len(temp_lost_tracks)
                n_bank = reid.bank.size()
                
                info_text = f"Active:{n_active} Pending:{n_pending} TempLost:{n_temp_lost} Bank:{n_bank}"
                cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Densidade
                density_text = f"Density:{density_smooth:.2f}"
                cv2.putText(frame, density_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                # ============================================================
                # TIMER: tempo de processamento atual
                # ============================================================
                elapsed_time = time.time() - video_start_time
                timer_text = f"Time:{elapsed_time:.1f}s"
                cv2.putText(frame, timer_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                # ============================================================
                # STEP 7: MANUTENÇÃO DO BANCO (TTL + HEALTH + LRU)
                # ============================================================
                if frame_idx % BANK_TICK_INTERVAL == 0:
                    # Tick do banco (decay health, verifica TTL)
                    dt = BANK_TICK_INTERVAL / fps  # tempo decorrido em segundos
                    expired_ids = reid.bank.tick(dt)
                    
                    if expired_ids:
                        print(f"[BANK_TICK] Expired {len(expired_ids)} identities: {expired_ids}")

                # ============================================================
                # STEP 8: DISPLAY
                # ============================================================
                cv2.imshow("Person Detection + Re-ID", frame)

            # ============================================================
            # FPS CONTROL (mantém velocidade real do vídeo)
            # ============================================================
            next_frame_time += frame_interval
            sleep_time = next_frame_time - time.time()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                next_frame_time = time.time()

            frame_idx += 1

            # ============================================================
            # ESC PARA SAIR
            # ============================================================
            if cv2.waitKey(1) & 0xFF == 27:
                print("ℹ️  Encerrando...")
                app.detector.running = False
                embedder.stop()
                cap.release()
                cv2.destroyAllWindows()
                return

        cap.release()