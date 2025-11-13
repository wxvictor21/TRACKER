"""
Flask API para Gantry (GRBL + Cámara)
- Manejo CORS manual en after_request (no requiere flask-cors).
- Si PREFER_RESTRICTED_CORS=True (env), solo permite ORIGIN_ALLOWED.
- Si ENV=production (env var), sirve una carpeta static/www (build de React).
"""

import os
import sys
from ultralytics import YOLO

# Add project subdirectories to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'EPIs')))
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', r"EPIs\runs\detect\train13\weights\best.pt"))
model = YOLO(MODEL_PATH)

# Add Hikvision SDK library path to PATH
os.environ['PATH'] = r"C:\Program Files (x86)\Common Files\MVS\Runtime\Win64_x64" + os.pathsep + os.environ['PATH']
from  CAMERAS.cameras.camera_brands.hik import HikCamera
import cv2

import time
from flask import Flask, Response, request, jsonify, send_from_directory, abort, make_response
from flask_cors import CORS
import grbl_module



def gen_frames():
    while True:
        frame = camera.capture_frame()  # Get raw frame
        if frame is not None:
            # Bayer to BGR conversion
            if len(frame.shape) == 2:
                try:
                    # NOTE: The Bayer pattern (e.g., COLOR_BAYER_RG2BGR) is a guess.
                    # It might need to be changed based on the camera's specific sensor.
                    color_frame = cv2.cvtColor(frame, cv2.COLOR_BAYER_RG2BGR)
                except cv2.error:
                    color_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            else:
                color_frame = frame

            # Convert to grayscale
            gray_frame = cv2.cvtColor(color_frame, cv2.COLOR_BGR2GRAY)
            
            # Convert grayscale to 3-channel for YOLO input (duplicate channel)
            yolo_input_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)

            # Run Inference on the 3-channel grayscale image
            results = model(yolo_input_frame, verbose=False)
            detections = results[0].boxes

            # Create a BGR version of the grayscale frame for drawing (to use color drawing functions)
            display_frame_bgr = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)

            # Draw boxes
            if detections:
                for det in detections:
                    box = det.xywh[0]
                    x_center, y_center, w, h = box.tolist()
                    x1, y1 = int(x_center - w / 2), int(y_center - h / 2)
                    x2, y2 = int(x_center + w / 2), int(y_center + h / 2)
                    class_id = int(det.cls.item())
                    label = model.names[class_id]
                    # Draw in white on the BGR version of the grayscale frame
                    cv2.rectangle(display_frame_bgr, (x1, y1), (x2, y2), (255, 255, 255), 2) # White color
                    cv2.putText(display_frame_bgr, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2) # White color

            # Convert the drawn frame back to grayscale for output
            final_output_frame = cv2.cvtColor(display_frame_bgr, cv2.COLOR_BGR2GRAY)

            # Encode to JPEG
            ret, buffer = cv2.imencode('.jpg', final_output_frame)
            frame_bytes = buffer.tobytes()

            yield (b'--frame\n'
                   b'Content-Type: image/jpeg\n\n' + frame_bytes + b'\n')
        else:
            # If no frame is available, wait a bit before trying again
            time.sleep(0.1)


# Importa tus controladores existentes
# Asegúrate de que camera_module.py y grbl_module.py estén en el mismo paquete/path
from CAMERAS.cameras.camera_brands.hik import HikCamera
from grbl_module import GRBLController

GRBL_PORT = "COM8" #linux use: "/dev/ttyUSB0"
GRBL_BAUD = 115200

CAMERA_IP = "192.168.2.240"
# Directorio para imágenes y para servir React build en producción
BASE_DIR = os.path.dirname(__file__)
PHOTOS_DIR = os.path.join(BASE_DIR, "static", "photos")
# REACT_BUILD_DIR = os.path.join(BASE_DIR, "static", "www")  # sitúa aquí tu build de React si quieres servirla

# Crear carpetas si faltan
os.makedirs(PHOTOS_DIR, exist_ok=True)
# if ENV == "production":
#     os.makedirs(REACT_BUILD_DIR, exist_ok=True)

# Instanciar controladores (puede lanzar excepciones si no hay hardware; en dev puedes mockear)
camera_config = {"serial_number": "your_serial_number", "ip": CAMERA_IP}
camera = HikCamera(camera_config)
camera.start_camera()
camera.set_parameter("GainAuto", "Once")
camera.start_grabbing()
grbl = GRBLController(GRBL_PORT, GRBL_BAUD)

# App Flask
app = Flask(__name__)
CORS(app)

# ----------------------------
# Endpoints API
# ----------------------------
@app.get("/api/status")
def status():
    """Devuelve estado de GRBL (respuesta de '?')"""
    try:
        resp = grbl.status()
    except Exception as e:
        return jsonify({"error": "GRBL error", "message": str(e)}), 500
    return jsonify({"grbl": resp})

@app.post("/api/move")
def move():
    """Mover X,Y (POST JSON {x, y, f?})"""
    data = request.get_json() or {}
    x = data.get("x")
    y = data.get("y")
    f = data.get("f", 1500)
    if x is None or y is None:
        return jsonify({"error": "x and y required"}), 400
    try:
        cmd, resp = grbl.move(x, y, f=f)
    except Exception as e:
        return jsonify({"error": "GRBL move failed", "message": str(e)}), 500
    return jsonify({"cmd": cmd, "response": resp})

@app.post("/api/capture")
def capture():
    """Dispara la cámara y devuelve la URL pública de la imagen guardada"""
    try:
        img = camera.capture_frame()
        if img is not None:
            filename = f"{time.strftime('%Y%m%d_%H%M%S')}.jpg"
            filepath = os.path.join(PHOTOS_DIR, filename)
            cv2.imwrite(filepath, img)
        else:
            filename = None
    except Exception as e:
        return jsonify({"error": "camera error", "message": str(e)}), 500

    if filename:
        return jsonify({"status": "captured", "file": f"/api/photos/{filename}"})
    else:
        return jsonify({"status": "error", "message": "No frame received"}), 500

@app.get("/api/gallery")
def gallery():
    """Lista las fotos guardadas (URLs relativas)"""
    try:
        files = sorted([f for f in os.listdir(PHOTOS_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    except Exception as e:
        return jsonify({"error": "listing photos", "message": str(e)}), 500
    urls = [f"/api/photos/{fname}" for fname in files]
    return jsonify({"photos": urls})

@app.get("/api/photos/<path:filename>")
def photos(filename):
    """Sirve las imágenes guardadas"""
    try:
        return send_from_directory(PHOTOS_DIR, filename, as_attachment=False)
    except Exception as e:
        return jsonify({"error": "file not found", "message": str(e)}), 404

@app.post("/api/sequence")
def sequence():
    """Ejecuta una secuencia simple: movimientos + capturas"""
    data = request.get_json() or {}
    num_shots = int(data.get("num_shots", 1))
    step_x = float(data.get("step_x", 1.0))
    step_y = float(data.get("step_y", 1.0))
    start_x = float(data.get("start_x", 0))
    start_y = float(data.get("start_y", 0))

    photos = []
    shot_count = 0
    try:
        for i in range(num_shots):
            # ejemplo raster (puedes adaptar a tu lógica)
            x = start_x + (i % max(1, int(num_shots))) * step_x
            y = start_y + (i // max(1, int(num_shots))) * step_y
            grbl.move(x, y)
            img = camera.capture_frame()
            if img is not None:
                filename = f"{time.strftime('%Y%m%d_%H%M%S')}.jpg"
                filepath = os.path.join(PHOTOS_DIR, filename)
                cv2.imwrite(filepath, img)
                photos.append(f"/api/photos/{filename}")
            shot_count += 1
            time.sleep(0.1)
    except Exception as e:
        return jsonify({"error": "sequence failed", "message": str(e)}), 500

    return jsonify({"num_shots": shot_count, "photos": photos})

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# ----------------------------
# (Opcional) Servir React build en producción
# ----------------------------
if __name__ == "__main__":
    # The host='0.0.0.0' parameter makes the app publicly available on your network.
    # Note: The Flask development server is not recommended for production use.
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)
