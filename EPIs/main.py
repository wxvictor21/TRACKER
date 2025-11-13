import cv2
from ultralytics import YOLO

# Cargar modelo entrenado
#model = YOLO(r"C:\Users\Victor Navarro\MERGE\EPIs\best.pt")  # Ajusta la ruta
model = YOLO(r"C:\Users\Victor Navarro\EPIs\runs\detect\train13\weights\best.pt")  # Ajusta la ruta

# Abrir cámara
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: No se pudo abrir la cámara")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Ejecutar detección
    results = model(frame)

    annotated_frame = frame.copy()

    # Dibujar bounding boxes
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # coordenadas
            conf = box.conf[0]  # confianza
            cls = int(box.cls[0])  # clase

            # Definir colores por clase
            colors = [(0, 0, 255), (255, 0, 0), (0, 255, 0)]  # Red, Blue, Green
            color = colors[cls]

            label = f"{model.names[cls]} {conf:.2f}"

            # Dibujar caja
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

            # Texto
            cv2.putText(
                annotated_frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

    # Mostrar frame
    cv2.imshow("Display", annotated_frame)

    # Salir con tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
