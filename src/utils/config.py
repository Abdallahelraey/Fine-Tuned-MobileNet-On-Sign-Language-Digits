from pydantic_settings import BaseSettings

class Config(BaseSettings):
    # Data
    DATA_BASE_PATH: str
    DATA_PROCESSED_PATH: str
    DATA_TRAIN_PATH: str
    DATA_VALIDATION_PATH: str
    DATA_TEST_PATH: str
    DATA_DATASET_BASE_PATH: str

    # Preprocessing
    PREPROCESSING_TARGET_SIZE: str
    PREPROCESSING_BATCH_SIZE: int
    PREPROCESSING_VALIDATION_SPLIT: float
    PREPROCESSING_TEST_SPLIT: float
    PREPROCESSING_AUGMENTATION_ROTATION_RANGE: int
    PREPROCESSING_AUGMENTATION_WIDTH_SHIFT_RANGE: float
    PREPROCESSING_AUGMENTATION_HEIGHT_SHIFT_RANGE: float
    PREPROCESSING_AUGMENTATION_SHEAR_RANGE: float
    PREPROCESSING_AUGMENTATION_ZOOM_RANGE: float
    PREPROCESSING_AUGMENTATION_HORIZONTAL_FLIP: bool
    PREPROCESSING_AUGMENTATION_FILL_MODE: str

    # Model
    INPUT_SHAPE_OF_MODEL: str
    NUM_CLASSES_OF_MODEL: int
    BASE_MODEL_OF_MODEL: str
    INCLUDE_TOP_OF_MODEL: bool
    WEIGHTS_OF_MODEL: str
    POOLING_OF_MODEL: str
    DROPOUT_RATE_OF_MODEL: float
    DENSE_UNITS_OF_MODEL: int
    ACTIVATION_OF_MODEL: str
    OUTPUT_ACTIVATION_OF_MODEL: str
    TRAINABLE_BASE_LAYERS_OF_MODEL: int
    TRAINING_REDUCE_LR_MONITOR: str

    # Training
    TRAINING_EPOCHS: int
    TRAINING_INITIAL_LEARNING_RATE: float
    TRAINING_BATCH_SIZE: int
    TRAINING_VALIDATION_SPLIT: float
    TRAINING_EARLY_STOPPING_PATIENCE: int
    TRAINING_REDUCE_LR_PATIENCE: int
    TRAINING_REDUCE_LR_FACTOR: float
    TRAINING_MIN_LR: float
    TRAINING_OPTIMIZER: str
    TRAINING_LOSS: str
    TRAINING_METRICS: str

    # Paths
    PATHS_MODEL_SAVE_PATH: str
    PATHS_LOG_DIR: str
    PATHS_CHECKPOINT_PATH: str

    # Visualization
    VISUALIZATION_PLOT_HISTORY: bool
    VISUALIZATION_PLOT_CONFUSION_MATRIX: bool
    VISUALIZATION_CLASS_NAMES: str
    VISUALIZATION_SAVE_PLOTS: bool
    VISUALIZATION_PLOTS_DIR: str

    # Testing
    TESTING_BATCH_SIZE: int
    TESTING_USE_BEST_MODEL: bool

    # Miscellaneous
    MISC_RANDOM_SEED: int
    MISC_VERBOSE: int

    # Experiment tracking
    EXPERIMENT_NAME: str
    EXPERIMENT_TAGS: str
    EXPERIMENT_LOG_MODEL: bool

    # Logging
    LOGGING_LEVEL: str
    LOGGING_FILE_PATH: str
    LOGGING_FORMAT: str

    # Class configuration for environment files
    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'

def load_config():
    return Config()

# Usage example
if __name__ == "__main__":
    config = load_config()
    print(config)
