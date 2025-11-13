import yaml
import os
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
from tqdm import tqdm
import random
import pandas as pd
from skimage.feature import hog
from skimage.exposure import rescale_intensity

# --- Configuration ---
DATA_YAML_PATH = "data.yaml"
IMAGE_SAMPLE_SIZE_PER_CLASS = 75
OUTPUT_DIR = "analysis_results"

# --- Helper Functions ---

def parse_data_yaml(yaml_path):
    """Parses the data.yaml file to get dataset info."""
    try:
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        for k in ['train', 'val', 'test']:
            if not data.get(k) or not Path(data[k]).exists():
                print(f"Warning: Path for '{k}' not found or not specified: {data.get(k)}")
                data[k] = None
        if 'names' not in data:
            raise KeyError("'names' key not found in YAML file.")
        return data
    except FileNotFoundError:
        print(f"Error: The file {yaml_path} was not found.")
        return None
    except Exception as e:
        print(f"An error occurred while parsing the YAML file: {e}")
        return None

def get_all_files(path, extensions):
    """Recursively gets all files with given extensions from a directory."""
    if not path: return []
    files = []
    for ext in extensions:
        files.extend(Path(path).rglob(f'*{ext}'))
    return files

def plot_and_save_chart(data, title, xlabel, output_path, kind='bar', bins=30):
    """Plots and saves a chart (bar or hist)."""
    plt.figure(figsize=(12, 7))
    if kind == 'bar':
        labels = list(data.keys())
        values = list(data.values())
        splot = sns.barplot(x=labels, y=values, palette='viridis')
        for p in splot.patches:
            splot.annotate(f'{p.get_height():.0f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                           ha='center', va='center', xytext=(0, 9), textcoords='offset points')
        plt.ylabel("Number of Images")
    elif kind == 'hist':
        sns.histplot(data, kde=True, bins=bins)
        plt.ylabel("Frequency")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved plot to {output_path}")

# --- Analysis Functions ---

def analyze_dataset_size(image_paths):
    print("\n--- 1. Dataset Size ---")
    total_size = sum(p.stat().st_size for p in image_paths)
    total_size_mb = total_size / (1024 * 1024)
    print(f"Total number of images: {len(image_paths)}")
    print(f"Total size of all images: {total_size_mb:.2f} MB")

def analyze_image_resolutions(image_paths, output_path):
    print("\n--- 2. Image Resolution and Aspect Ratio Analysis ---")
    resolutions, aspect_ratios = [], []
    for img_path in tqdm(image_paths, desc="Analyzing resolutions"):
        try:
            img = cv2.imread(str(img_path))
            if img is not None:
                h, w, _ = img.shape
                resolutions.append(f"{w}x{h}")
                aspect_ratios.append(w / h)
        except Exception as e:
            print(f"Warning: Could not read {img_path}. Error: {e}")
    if not aspect_ratios: return
    print(f"Found {len(set(resolutions))} unique resolutions. Most common: {Counter(resolutions).most_common(1)[0][0]}")
    plot_and_save_chart(aspect_ratios, "Image Aspect Ratio Distribution", "Aspect Ratio (Width / Height)", output_path, kind='hist')

def analyze_object_density(image_paths, output_path):
    print("\n--- 3. Object Density Analysis ---")
    labels_per_image = []
    for img_path in tqdm(image_paths, desc="Analyzing object density"):
        label_path = img_path.parent.parent / 'labels' / f'{img_path.stem}.txt'
        if label_path.exists():
            with open(label_path, 'r') as f:
                labels_per_image.append(len(f.readlines()))
    plot_and_save_chart(labels_per_image, "Distribution of Objects Per Image", "Number of Objects", output_path, kind='hist')

def analyze_object_sizes(image_paths, class_names, output_path):
    print("\n--- 4. Relative Object Size Analysis ---")
    object_areas = defaultdict(list)
    for img_path in tqdm(image_paths, desc="Analyzing object sizes"):
        label_path = img_path.parent.parent / 'labels' / f'{img_path.stem}.txt'
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    try:
                        class_id, w, h = int(parts[0]), float(parts[3]), float(parts[4])
                        if 0 <= class_id < len(class_names):
                            object_areas[class_names[class_id]].append(w * h)
                    except (ValueError, IndexError): continue
    df = pd.DataFrame.from_dict(object_areas, orient='index').transpose()
    plt.figure(figsize=(14, 8)); sns.boxplot(data=df, palette="viridis")
    plt.title("Relative Object Area by Class"); plt.ylabel("Relative Area (width * height)"); plt.xlabel("Class")
    plt.xticks(rotation=45, ha='right'); plt.tight_layout(); plt.savefig(output_path); plt.close()
    print(f"Saved plot to {output_path}")

def visualize_hog_features(image_paths, output_path, num_samples=4):
    """Computes and visualizes HOG features for a sample of images."""
    print("\n--- 5. HOG Feature Visualization ---")
    sample_paths = random.sample(image_paths, min(len(image_paths), num_samples))
    fig, axes = plt.subplots(num_samples, 2, figsize=(10, 5 * num_samples))
    fig.suptitle('HOG Feature Visualization', fontsize=16)
    for i, img_path in enumerate(sample_paths):
        try:
            img = cv2.imread(str(img_path))
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            fd, hog_image = hog(gray_img, orientations=8, pixels_per_cell=(16, 16),
                                cells_per_block=(1, 1), visualize=True)
            hog_image_rescaled = rescale_intensity(hog_image, in_range=(0, 10))

            ax = axes[i] if num_samples == 1 else axes[i, 0]
            ax.imshow(img_rgb); ax.set_title("Original Image"); ax.axis('off')
            ax = axes[i] if num_samples == 1 else axes[i, 1]
            ax.imshow(hog_image_rescaled, cmap=plt.cm.gray); ax.set_title("HOG Visualization"); ax.axis('off')
        except Exception as e:
            print(f"Warning: Could not process HOG for {img_path}. Error: {e}")
    plt.tight_layout(rect=[0, 0.03, 1, 0.96]); plt.savefig(output_path); plt.close()
    print(f"Saved plot to {output_path}")

def analyze_and_save_hsv_plots(class_name, image_paths, output_path):
    print(f"\nAnalyzing HSV for class: '{class_name}'")
    hues, saturations, values = [], [], []
    sample_size = min(len(image_paths), IMAGE_SAMPLE_SIZE_PER_CLASS)
    if sample_size == 0: return
    sample_paths = random.sample(image_paths, sample_size)
    for img_path in tqdm(sample_paths, desc=f"Processing {class_name}"):
        try:
            img = cv2.imread(str(img_path))
            if img is not None:
                hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                h, s, v = cv2.split(hsv_img)
                hues.extend(h.flatten()); saturations.extend(s.flatten()); values.extend(v.flatten())
        except Exception as e: print(f"Warning: Could not process {img_path}. Error: {e}")
    if not hues: return
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(f'HSV Pixel Distribution for Class: "{class_name}" (Sample of {sample_size} images)', fontsize=16)
    sns.histplot(hues, bins=180, kde=False, color='red', ax=axes[0]); axes[0].set_title('Hue')
    sns.histplot(saturations, bins=256, kde=False, color='green', ax=axes[1]); axes[1].set_title('Saturation')
    sns.histplot(values, bins=256, kde=False, color='blue', ax=axes[2]); axes[2].set_title('Value')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.savefig(output_path); plt.close()
    print(f"Saved HSV analysis plot to {output_path}")

# --- Main Execution ---

def main():
    """Main function to run the class-based dataset analysis."""
    print("--- Starting Comprehensive Dataset Analysis ---")
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    
    dataset_info = parse_data_yaml(DATA_YAML_PATH)
    if not dataset_info: return

    image_paths = []
    for key in ['train', 'val', 'test']:
        if dataset_info.get(key):
            path = Path(dataset_info[key]) / 'images'
            if path.exists(): image_paths.extend(get_all_files(path, ['.jpg', '.jpeg', '.png']))
    
    if not image_paths: print("Fatal: No image files were found. Exiting."); return
    class_names = dataset_info.get('names', [])

    # --- Run All Analyses ---
    analyze_dataset_size(image_paths)
    analyze_image_resolutions(image_paths, Path(OUTPUT_DIR) / "aspect_ratio_distribution.png")
    analyze_object_density(image_paths, Path(OUTPUT_DIR) / "object_density_distribution.png")
    analyze_object_sizes(image_paths, class_names, Path(OUTPUT_DIR) / "object_size_distribution.png")
    visualize_hog_features(image_paths, Path(OUTPUT_DIR) / "hog_feature_visualization.png")

    images_per_class = defaultdict(set)
    for img_path in image_paths:
        label_path = img_path.parent.parent / 'labels' / f'{img_path.stem}.txt'
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    try:
                        class_id = int(line.strip().split()[0])
                        if 0 <= class_id < len(class_names):
                            images_per_class[class_names[class_id]].add(img_path)
                    except (ValueError, IndexError): continue

    print("\n--- Image Counts Per Class ---")
    image_counts = {name: len(paths) for name, paths in images_per_class.items()}
    plot_and_save_chart(image_counts, "Number of Images Containing Each Class", "Class Name", Path(OUTPUT_DIR) / "class_image_counts.png")

    print("\n--- Class-based HSV Analysis ---")
    for class_name, paths in images_per_class.items():
        safe_class_name = "".join(c for c in class_name if c.isalnum() or c in (' ', '_')).rstrip()
        hsv_plot_path = Path(OUTPUT_DIR) / f"hsv_distribution_{safe_class_name.replace(' ', '_')}.png"
        analyze_and_save_hsv_plots(class_name, list(paths), hsv_plot_path)
    
    print("\n--- Analysis Complete ---")
    print(f"All result plots have been saved in the '{OUTPUT_DIR}' directory.")

if __name__ == "__main__":
    main()
