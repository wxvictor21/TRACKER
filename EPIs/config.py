class Config:
    """Configuración general para el entrenamiento del modelo PPE Detection."""

    # --- Modo de ejecución ---
    DEBUG_MODE = True
    DATA_PORTION = 0.2 if DEBUG_MODE else 1.0
    RANDOM_SEED = 2025

    # --- Rutas y directorios ---
    DATASET_PATH = '/kaggle/input/construction-safety-dataset/data/'
    OUTPUT_PATH = './output_models/'
    
    # --- Información de clases ---
    TARGET_CLASSES = [
        'Person', 'Hardhat', 'NO-Hardhat', 'Mask',
        'NO-Mask', 'Safety Vest', 'NO-Safety Vest',
        'Safety Cone', 'Machinery', 'Vehicle'
    ]
    N_CLASSES = len(TARGET_CLASSES)

    # --- Parámetros del modelo ---
    MODEL_BACKBONE = 'yoloe-11n-seg'
    MODEL_WEIGHTS = f'{MODEL_BACKBONE}.pt'
    EXPERIMENT_NAME = f'ppe_exp_{N_CLASSES}_classes'

    # --- Entrenamiento ---
    TOTAL_EPOCHS = 5 if DEBUG_MODE else 100
    BATCH_SIZE = 12
    OPTIMIZER_TYPE = 'AdamW'     # Posibles: SGD, Adam, RMSProp, etc.
    BASE_LR = 0.001
    LR_DECAY = 0.02
    WEIGHT_REGULARIZATION = 0.0005
    DROPOUT_RATE = 0.03
    EARLY_STOP_PATIENCE = 20
    ENABLE_PROFILING = False
    LABEL_SMOOTH = 0.1