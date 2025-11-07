# detector.py
import threading
import time
import cv2
from ultralytics import YOLO

latest_frame = None
latest_detections = []
lock_frame = threading.Lock()
lock_det = threading.Lock()

running = True

def set_frame(frame):
    global latest_frame
    with lock_frame:
        latest_frame = frame.copy()

def get_detections():
    with lock_det:
        return list(latest_detections)

def person_is_valid(keypoints):
    """
    Retorna True se a pessoa for válida (tem corpo).
    """
    head_ids = [0,1,2,3,4]
    body_ids = [5,6,11,12]

    head_ok = any(keypoints[i][2] > 0.2 for i in head_ids)   # confiança > 0.2
    body_ok = any(keypoints[i][2] > 0.2 for i in body_ids)

    return head_ok and body_ok

def detector_thread():
    global latest_frame, latest_detections, running

    model = YOLO("models/yolov11n-pose.pt")

    while running:
        frame = None
        with lock_frame:
            if latest_frame is not None:
                frame = latest_frame.copy()

        if frame is None:
            time.sleep(0.001)
            continue

        results = model(frame, verbose=False)
        dets = []

        for r in results:
            for box, kp in zip(r.boxes, r.keypoints):
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls = int(box.cls[0])

                # só pessoa
                if cls != 0:
                    continue

                # kp.xy: (17, 2) / kp.conf: (17)
                keypoints = []
                for (x,y),c in zip(kp.xy[0].tolist(), kp.conf[0].tolist()):
                    keypoints.append((x,y,c))

                # valida pessoa real
                if not person_is_valid(keypoints):
                    continue

                dets.append([x1, y1, x2, y2, conf, cls, keypoints])

        with lock_det:
            latest_detections = dets

        time.sleep(0.001)
