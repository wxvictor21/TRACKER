from ultralytics import YOLO
from config import Config
import os


def main():
    """
    Main function to run training.
    """
    print("Starting model training...")
    model = YOLO("runs/detect/train8/weights/last.pt")
    model.train(
        data=os.path.join(Config.OUTPUT_DIR, 'data.yaml'),
        epochs=50,
        imgsz=640,
        batch=16,
        conf=0.5,
    )
    print("Training finished.")


if __name__ == '__main__':
    main()