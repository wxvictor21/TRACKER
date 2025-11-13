import kagglehub
import os
import yaml
from config import Config

def update_yolo_labels(directory):
    """
    Scans a directory for YOLO label files (.txt) and updates the class IDs.
    - Keeps only the specified class IDs.
    - Remaps the kept class IDs to new sequential values (0, 1, 2, ...).
    """
    # Original class indices to keep
    # 4: NO-Safety Vest, 5: Person, 7: Safety Vest
    remap_dict = {
        4: 0,  # NO-Safety Vest -> 0
        5: 1,  # Person -> 1
        7: 2   # Safety Vest -> 2
    }
    kept_classes = set(remap_dict.keys())
    
    labels_dir = os.path.join(directory, "labels")
    if not os.path.exists(labels_dir):
        print(f"Warning: Label directory not found, skipping update: {labels_dir}")
        return

    print(f"Updating labels in: {labels_dir}...")
    for filename in os.listdir(labels_dir):
        if filename.endswith(".txt"):
            filepath = os.path.join(labels_dir, filename)
            
            with open(filepath, "r") as f:
                lines = f.readlines()

            new_lines = []
            for line in lines:
                parts = line.strip().split()
                if not parts:
                    continue
                
                original_class_id = int(parts[0])
                
                if original_class_id in kept_classes:
                    new_class_id = remap_dict[original_class_id]
                    new_lines.append(f"{new_class_id} {' '.join(parts[1:])}\n")
            
            with open(filepath, "w") as f:
                f.writelines(new_lines)
    print("Update complete.")

def main():
    """
    Main function to download data, prepare yaml, and update labels.
    """
    # This logic handles path construction for older kagglehub library versions.
    print("Checking for dataset...")
    # Define the path to the specific version directory we expect.
    dataset_parent_path = os.path.join(os.path.expanduser("~"), ".cache", "kagglehub", "datasets", "snehilsanyal", "construction-site-safety-image-dataset-roboflow")
    version_path = os.path.join(dataset_parent_path, 'versions', '3')

    # If the specific version directory doesn't exist, download it.
    if not os.path.exists(version_path):
        print("Downloading dataset from Kaggle...")
        # kagglehub.dataset_download returns the path to the version directory.
        version_path = kagglehub.dataset_download("snehilsanyal/construction-site-safety-image-dataset-roboflow")
        print(f"Dataset downloaded to: {version_path}")
    else:
        print(f"Dataset version already exists at: {version_path}")

    # Construct the path to the actual data from the now-guaranteed-correct version path.
    data_root_path = os.path.join(version_path, 'css-data')

    # This is critical to match the classes we want to train.
    update_yolo_labels(os.path.join(data_root_path, 'train'))
    update_yolo_labels(os.path.join(data_root_path, 'valid'))
    update_yolo_labels(os.path.join(data_root_path, 'test'))

    # This creates the .yaml file with the correct, absolute paths for the current machine.
    print("Creating data.yaml file...")
    dict_file = {
        'train': os.path.join(data_root_path, 'train'),
        'val': os.path.join(data_root_path, 'valid'),
        'test': os.path.join(data_root_path, 'test'),
        'nc': Config.NUM_CLASSES_TO_TRAIN,
        'names': Config.CLASSES
    }

    data_yaml_path = os.path.join(Config.OUTPUT_DIR, 'data.yaml')
    with open(data_yaml_path, 'w') as file:
        yaml.dump(dict_file, file)
    print(f"data.yaml created at: {data_yaml_path}")

if __name__ == '__main__':
    main()
