import sys
import os
import cv2
import time
import threading
from ultralytics import YOLO

# Add HIKvision SDK to PATH
hik_sdk_path = "C:\\Program Files (x86)\\Common Files\\MVS\\Runtime\\Win64_x64"
if os.path.exists(hik_sdk_path):
    os.environ['PATH'] = f"{hik_sdk_path};{os.environ['PATH']}"
else:
    print(f"Warning: HIKvision SDK path not found at {hik_sdk_path}")
    print("Please make sure the HIKvision MVS SDK is installed and the path is correct.")

# Add project subdirectories to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'gantryApp', 'gantry-main')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'EPIs')))

try:
    from cameras.camera_brands.hik import HikCamera
    from grbl_module import GRBLController
except ImportError as e:
    print(f"Error: Failed to import necessary modules. Make sure paths are correct.")
    print(f"Details: {e}")
    sys.exit(1)

# --- Constants and Configuration ---
PIXELS_TO_MM = 0.1  # TODO: Calibrate this value
MODEL_PATH = r"C:\Users\Victor Navarro\MERGE\EPIs\runs\detect\train13\weights\best.pt"
GRBL_PORT = "COM7"  # IMPORTANT: Change to your GRBL controller's serial port
GRBL_BAUDRATE = 115200
CAMERA_IP = "192.168.2.202"
STREAM_FPS = 30 # Target FPS for the video window

# --- Global variables for threads and data sharing ---
frame_grabber = None
processing_thread = None

# --- Thread for grabbing frames from the camera ---
class FrameGrabber(threading.Thread):
    def __init__(self, cam, target_size=(1280, 720)):
        super().__init__()
        self.cam = cam
        self.target_size = target_size
        self.latest_frame = None
        self.lock = threading.Lock()
        self.daemon = True
        self.running = True

    def run(self):
        frame_count = 0
        while self.running:
            try:
                frame = self.cam.capture_frame()

                if frame is not None:
                    # Convert Bayer pattern to BGR color image
                    if len(frame.shape) == 2:
                        # NOTE: The Bayer pattern (e.g., COLOR_BAYER_RG2BGR) is a guess.
                        # It might need to be changed based on the camera's specific sensor.
                        # Common patterns: _RG, _BG, _GR, _GB
                        try:
                            color_frame = cv2.cvtColor(frame, cv2.COLOR_BAYER_RG2BGR)
                        except cv2.error:
                            # Fallback to simple grayscale conversion if debayering fails
                            color_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                    else:
                        color_frame = frame

                    resized_frame = cv2.resize(color_frame, self.target_size, interpolation=cv2.INTER_AREA)
                    with self.lock:
                        self.latest_frame = resized_frame
                    frame_count += 1
                else:
                    print("[FrameGrabber] Failed to grab frame.")
                    time.sleep(0.5) # Wait before retrying

            except Exception as e:
                print(f"!!! EXCEPTION in FrameGrabber thread: {e}")
                self.running = False

    def get_frame(self):
        with self.lock:
            if self.latest_frame is None:
                return None
            return self.latest_frame.copy()

    def stop(self):
        self.running = False

# --- Thread for processing frames (YOLO, GRBL) ---
class ProcessingThread(threading.Thread):
    def __init__(self, frame_grabber, grbl, model, image_center, image_size):
        super().__init__()
        self.frame_grabber = frame_grabber
        self.grbl = grbl
        self.model = model
        self.image_center_x, self.image_center_y = image_center
        self.image_width, self.image_height = image_size
        
        self.latest_detections = []
        self.detection_lock = threading.Lock()
        
        self.current_x = 0.0
        self.current_y = 0.0
        self.position_lock = threading.Lock()

        self.daemon = True
        self.running = True

    def run(self):
        # Get initial GRBL position
        self._update_initial_position()

        while self.running:
            frame = self.frame_grabber.get_frame()
            if frame is None:
                time.sleep(0.02) # Wait for the first frame
                continue

            # --- Run Inference ---
            results = self.model(frame, verbose=False)
            with self.detection_lock:
                self.latest_detections = results[0].boxes

            # --- Move Gantry ---
            if len(self.latest_detections) > 0:
                self._move_gantry()
            
            # This sleep prevents the thread from hogging the CPU
            # and gives other threads time to run.
            time.sleep(0.01)

    def _update_initial_position(self):
        with self.position_lock:
            status_lines = self.grbl.status()
            if status_lines:
                for line in status_lines:
                    if line.startswith('<') and 'WPos:' in line:
                        try:
                            wpos_str = line.split('WPos:')[1].split('|')[0]
                            pos_parts = wpos_str.split(',')
                            self.current_x = float(pos_parts[0])
                            self.current_y = float(pos_parts[1])
                            print(f"Initial GRBL position: X={self.current_x}, Y={self.current_y}")
                        except (IndexError, ValueError) as parse_error:
                            print(f"Could not parse GRBL position: {parse_error}. Assuming (0,0).")
                        return # Found position line

    def _move_gantry(self):
        with self.detection_lock, self.position_lock:
            if not self.latest_detections:
                return

            largest_detection = max(self.latest_detections, key=lambda det: det.xywh[0][2] * det.xywh[0][3])
            box = largest_detection.xywh[0] # box is a tensor [x, y, w, h]
            x_center = box[0].item()
            y_center = box[1].item()

            error_x_px = x_center - self.image_center_x
            error_y_px = y_center - self.image_center_y
            move_x_mm = error_x_px * PIXELS_TO_MM
            move_y_mm = -error_y_px * PIXELS_TO_MM

            target_x = self.current_x + move_x_mm
            target_y = self.current_y + move_y_mm

            clamped_target_x = max(0, min(150, target_x))
            clamped_target_y = max(0, min(170, target_y))

            clamped_move_x = clamped_target_x - self.current_x
            clamped_move_y = clamped_target_y - self.current_y

            if abs(clamped_move_x) > 0.1 or abs(clamped_move_y) > 0.1:
                self.grbl.send('G91')
                move_command = f'G1 X{clamped_move_x:.2f} Y{clamped_move_y:.2f} F2500'
                self.grbl.send(move_command)
                self.grbl.send('G90')
                self.current_x += clamped_move_x
                self.current_y += clamped_move_y

    def get_latest_detections(self):
        with self.detection_lock:
            return self.latest_detections

    def stop(self):
        self.running = False

# --- Main Control Function ---
def main():
    global frame_grabber, processing_thread
    
    cam, grbl, model = None, None, None
    
    try:
        print("Initializing modules...")
        config = {"ip": CAMERA_IP, "serial_number": ""}  # Serial number can be empty if using IP
        cam = HikCamera(config)
        if not cam.start_camera():
            print("Could not initialize camera. Exiting.")
            return
        cam.start_grabbing()

        # Set camera exposure using the specific MVS SDK function
        EXPOSURE_TIME = 50000.0
        try:
            # The strKey for exposure time in GenICam standard is "ExposureTime"
            cam.set_parameter("ExposureTime", EXPOSURE_TIME)
            print(f"Attempted to set camera exposure to {EXPOSURE_TIME} us.")
        except Exception as e:
            print(f"Warning: Could not set camera exposure. {e}")

        grbl = GRBLController(port=GRBL_PORT, baudrate=GRBL_BAUDRATE)
        model = YOLO(MODEL_PATH)
        print("All modules initialized successfully.")

        # Get a single frame to determine dimensions before starting threads
        print("Getting frame dimensions from camera...")
        frame = None
        while frame is None:
            frame = cam.capture_frame()
            if frame is None:
                time.sleep(0.1) # Wait briefly before retrying
        
        shape = frame.shape
        if len(shape) == 3:
            original_height, original_width, _ = shape
        else:
            original_height, original_width = shape
        print(f"Original frame dimensions: {original_width}x{original_height}")

        # Define the target processing resolution and calculate aspect-correct height
        PROCESSING_WIDTH = 1280
        if original_width > 0:
            PROCESSING_HEIGHT = int(original_height * (PROCESSING_WIDTH / original_width))
        else:
            PROCESSING_HEIGHT = 720 # Fallback in case of error
        
        processing_size = (PROCESSING_WIDTH, PROCESSING_HEIGHT)
        processing_center = (PROCESSING_WIDTH / 2, PROCESSING_HEIGHT / 2)
        print(f"Processing frames at {processing_size[0]}x{processing_size[1]}")

        # Now, initialize and start the threads with the correct dimensions
        frame_grabber = FrameGrabber(cam, target_size=processing_size)
        processing_thread = ProcessingThread(frame_grabber, grbl, model, processing_center, processing_size)

        frame_grabber.start()
        processing_thread.start()

        print("\n--- System is running ---")
        print("Press 'q' in the video window to quit.")

        while True:
            frame = frame_grabber.get_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            detections = processing_thread.get_latest_detections()

            if detections:
                for det in detections:
                    box = det.xywh[0]
                    x_center, y_center, w, h = box.tolist()
                    x1, y1 = int(x_center - w/2), int(y_center - h/2)
                    x2, y2 = int(x_center + w/2), int(y_center + h/2)
                    # Get the class ID and name
                    class_id = int(det.cls.item())
                    label = model.names[class_id]

                    # Draw bounding box and label
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

            cv2.imshow('Gantry Live Feed', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            # Pace the loop to roughly match the target FPS
            time.sleep(1 / STREAM_FPS)


    except KeyboardInterrupt:
        print("\nCaught KeyboardInterrupt. Shutting down gracefully.")
    except Exception as e:
        print(f"An error occurred in the main thread: {e}")
    finally:
        print("--- Shutting down all threads ---")
        if frame_grabber:
            frame_grabber.stop()
            frame_grabber.join()
            print("Frame grabber stopped.")
        if processing_thread:
            processing_thread.stop()
            processing_thread.join()
            print("Processing thread stopped.")
        if cam:
            cam.stop_grabbing()
            cam.close()
            print("Camera connection closed.")
        if grbl and grbl.ser:
            grbl.ser.close()
            print("GRBL connection closed.")
        
        cv2.destroyAllWindows()
        print("OpenCV windows closed.")

if __name__ == "__main__":
    main()
