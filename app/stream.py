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
        weight_path="models/osnet_ibn_x1_0_imagenet.pth",
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
    while True:
        cap = cv2.VideoCapture(VIDEO_PATH)
        if not cap.isOpened():
            print("❌ Erro ao abrir vídeo:", VIDEO_PATH)
            time.sleep(1)
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0
        
        frame_interval = 1.0 / fps
        next_frame_time = time.time()
        frame_idx = 0

        print(f"🎬 Iniciando vídeo (FPS={fps:.1f})")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("🔄 Reiniciando vídeo...")
                break

            frame_h, frame_w = frame.shape[:2]

            # ============================================================
            # STEP 1: YOLO DETECTIONS (com escala)
            # ============================================================
            app.detector.set_frame(frame)
            detections = app.detector.get_detections()
            
            # Formato: [x1, y1, x2, y2, conf, cls, keypoints, scale, had_pad]

            # ============================================================
            # STEP 2: BYTETRACKER UPDATE (estados granulares)
            # ============================================================
            tracks_by_state = tracker.update(detections, (frame_h, frame_w))
            
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
            
            # ============================================================
            # DENSITY SUAVIZADA (EMA α=0.2)
            # ============================================================
            density_instant = len(all_active) / ((frame_h * frame_w) / 1000.0)
            density_smooth = 0.8 * density_smooth + 0.2 * density_instant
            
            with lock_frame:
                shared_frame['frame'] = frame.copy()
                shared_frame['frame_index'] = frame_idx

            with lock_tracks:
                shared_tracks['tracks'] = all_active
                shared_tracks['temp_lost'] = temp_lost_tracks
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