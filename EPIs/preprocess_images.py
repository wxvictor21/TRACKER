from ultralytics import YOLO
import cv2
import numpy as np
from constants import KNOWN_VEST_HEIGHT_CM, FOCAL_LENGTH_PIXELS

# Cargar el modelo YOLO
model = YOLO(r"C:\Users\Victor Navarro\EPIs\runs\detect\train11\weights\best.pt")



cap = cv2.VideoCapture(0)
tracker = cv2.legacy.TrackerCSRT_create()
init = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # --- Detección YOLO ---
    results = model.predict(frame, conf=0.4, verbose=False)
    detections = results[0].boxes

    if len(detections) > 0:
        # Escoge el bounding box más grande (el más cercano)
        det = max(detections, key=lambda b: (b.xywh[0][2] * b.xywh[0][3]))
        x1, y1, x2, y2 = map(int, det.xyxy[0])
        bbox = (x1, y1, x2 - x1, y2 - y1)

        if not init:
            tracker.init(frame, bbox)
            init = True

    elif init:
        # Si no hay detección, intentar seguir con el tracker
        success, bbox = tracker.update(frame)
        if not success:
            init = False
            continue

    if init:
        x, y, w, h = [int(v) for v in bbox]
        distance_cm = (KNOWN_VEST_HEIGHT_CM * FOCAL_LENGTH_PIXELS) / h

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(frame, f"{distance_cm:.1f} cm", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    cv2.imshow("YOLO Vest Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
