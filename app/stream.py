"""
Stream — Pipeline principal de detecção + tracking + Re-ID

Orquestra todo o sistema:
1. YOLO detecta pessoas (detector thread)
2. ByteTracker mantém IDs estáveis via Kalman
3. Re-ID identifica pessoas globalmente (quando reaparecem)
4. Renderiza overlay com cores persistentes

Arquitetura:
- Detector thread: YOLO assíncrono
- Embedder thread: Re-ID assíncrono  
- Main loop: streaming + tracking + visualização
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

    gallery = Gallery(buffer_size=15)
    
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
    # 4) MAIN LOOP (Video streaming + tracking + visualization)
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

            # -----------------------------------------------------------------
            # STEP 1: YOLO detections
            # -----------------------------------------------------------------
            app.detector.set_frame(frame)
            detections = app.detector.get_detections()
            
            # Formato: [x1, y1, x2, y2, conf, cls, keypoints]

            # -----------------------------------------------------------------
            # STEP 2: ByteTracker update
            # -----------------------------------------------------------------
            tracks = tracker.update(detections, (frame_h, frame_w))
            
            # Formato: [{'track_id', 'bbox', 'score', 'keypoints'}, ...]

            # -----------------------------------------------------------------
            # STEP 3: Alimenta Re-ID thread
            # -----------------------------------------------------------------
            with lock_frame:
                shared_frame['frame'] = frame.copy()
                shared_frame['frame_index'] = frame_idx

            with lock_tracks:
                shared_tracks['tracks'] = tracks

            # -----------------------------------------------------------------
            # STEP 4: Overlay (desenha tracks + IDs + cores)
            # -----------------------------------------------------------------
            for track in tracks:
                track_id = track['track_id']
                bbox = track['bbox']
                x1, y1, x2, y2 = bbox

                # Obtém id_global (se promovido) ou usa track_id temporário
                id_global = embedder.get_global_id(track_id)
                
                if id_global is not None:
                    # Re-ID bem-sucedido — mostra RID (id_global)
                    label = f"RID:{id_global}"
                else:
                    # Ainda em tracking temporário
                    label = f"T:{track_id}"

                # Cor persistente (do banco se promovido, ou temporária)
                color = embedder.get_color(track_id)

                # Desenha bbox + label
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                
                # Background para texto (melhor legibilidade)
                (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (int(x1), int(y1) - text_h - 8), (int(x1) + text_w, int(y1)), color, -1)
                cv2.putText(frame, label, (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # -----------------------------------------------------------------
            # STEP 5: Display
            # -----------------------------------------------------------------
            cv2.imshow("Person Detection + Re-ID", frame)

            # FPS control (mantém velocidade real do vídeo)
            next_frame_time += frame_interval
            sleep_time = next_frame_time - time.time()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                next_frame_time = time.time()

            frame_idx += 1

            # ESC para sair
            if cv2.waitKey(1) & 0xFF == 27:
                print("⏹️  Encerrando...")
                app.detector.running = False
                embedder.stop()
                cap.release()
                cv2.destroyAllWindows()
                return

        cap.release()