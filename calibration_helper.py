
import sys
import os
import cv2
import time

# --- Configuration ---
# Ensure this matches the IP in your main script
CAMERA_IP = "192.168.2.202"
# The window name
WINDOW_NAME = 'Calibration Helper'

# --- Path Setup ---
# Add HIKvision SDK to PATH
hik_sdk_path = "C:\\Program Files (x86)\\Common Files\\MVS\\Runtime\\Win64_x64"
if os.path.exists(hik_sdk_path):
    os.environ['PATH'] = f"{hik_sdk_path};{os.environ['PATH']}"
else:
    print(f"Warning: HIKvision SDK path not found at {hik_sdk_path}")

# Add project subdirectory to the Python path to find HikCamera
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'gantryApp', 'gantry-main')))

try:
    from cameras.camera_brands.hik import HikCamera
except ImportError as e:
    print("Error: Failed to import HikCamera.")
    print("Please ensure this script is in the same directory as 'gantry_centering.py'")
    print(f"Details: {e}")
    sys.exit(1)

# --- Mouse Callback Function ---
def print_pixel_coords(event, x, y, flags, param):
    """
    This function is called by OpenCV whenever a mouse event occurs in the window.
    It prints the coordinates of a left-button click.
    """
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Pixel Coordinate (X, Y): ({x}, {y})")

# --- Main Function ---
def main():
    print("Initializing camera...")
    cam = None
    try:
        config = {"ip": CAMERA_IP, "serial_number": ""}
        cam = HikCamera(config)
        if not cam.start_camera():
            print("Could not initialize camera. Exiting.")
            return
        cam.start_grabbing()
        print("Camera initialized successfully.")

        cv2.namedWindow(WINDOW_NAME)
        cv2.setMouseCallback(WINDOW_NAME, print_pixel_coords)
        print(f"--- Window '{WINDOW_NAME}' is active ---")
        print("Click on the video feed to get pixel coordinates.")
        print("Press 'q' in the window to quit.")

        while True:
            frame = cam.capture_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            # Convert Bayer pattern to BGR color image for display
            if len(frame.shape) == 2:
                try:
                    # Using the same Bayer pattern as the main script
                    color_frame = cv2.cvtColor(frame, cv2.COLOR_BAYER_RG2BGR)
                except cv2.error:
                    # Fallback if debayering fails
                    color_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            else:
                color_frame = frame
            
            # The raw image is very large, let's resize it for display
            # while maintaining aspect ratio.
            display_width = 1280
            h, w = color_frame.shape[:2]
            if w > display_width:
                aspect_ratio = h / w
                display_height = int(display_width * aspect_ratio)
                display_frame = cv2.resize(color_frame, (display_width, display_height))
            else:
                display_frame = color_frame

            cv2.imshow(WINDOW_NAME, display_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except KeyboardInterrupt:
        print("\nCaught KeyboardInterrupt. Shutting down.")
    finally:
        print("--- Shutting down ---")
        if cam:
            cam.stop_grabbing()
            cam.close()
            print("Camera connection closed.")
        cv2.destroyAllWindows()
        print("OpenCV windows closed.")

if __name__ == "__main__":
    main()
