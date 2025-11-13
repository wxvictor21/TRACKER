import numpy as np
import os

# --- Paths ---
# Path to the trained YOLO model (not used in optical flow, but kept for other scripts)
MODEL_PATH = r"C:\Users\Victor Navarro\EPIs\runs\detect\train13\weights\best.pt"

# Output directory for training data
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


# --- Image Processing (for binarization, if needed elsewhere) ---
# HSV color range for yellow
LOWER_YELLOW = np.array([20, 100, 100])
UPPER_YELLOW = np.array([35, 255, 255])

# HSV color range for grayish/white (for reflective stripes, if needed elsewhere)
LOWER_GRAY = np.array([0, 0, 100])
UPPER_GRAY = np.array([180, 50, 255])

# Kernel for morphological operations
MORPH_KERNEL = np.ones((5, 5), np.uint8)


# --- Distance Calculation Tuning ---
# Correction factor to fine-tune the distance calculation.
CORRECTION_FACTOR = 1.0
KNOWN_VEST_HEIGHT_CM = 60.0
FOCAL_LENGTH_PIXELS = 900.0


# --- Training Configuration (from CFG.py) ---
# Number of classes to train
NUM_CLASSES_TO_TRAIN = 3

# Class names
CLASSES = ['NO-Safety Vest', 'Person', 'Safety Vest']