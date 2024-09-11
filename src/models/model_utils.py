import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from src.utils.logging_config import logger 
from dotenv import load_dotenv
from src.utils.config import load_config
load_dotenv()

config = load_config()

def save_model(model, model_name):
    save_path = config.PATHS_MODEL_SAVE_PATH
    saved_model_path = os.path.join(save_path,model_name)
    model.save(saved_model_path)
    logger.info(f"Model saved at {saved_model_path}.")

def load_model(model_name):
    save_path = config.PATHS_MODEL_SAVE_PATH
    load_model_path = os.path.join(save_path,model_name)
    model = tf.keras.models.load_model(load_model_path)
    logger.info(f"Model loaded from {load_model_path}.")
    return model

def display_model_summary(model):
    model.summary()
    logger.info("Model summary displayed.")

def plot_training_history(history):
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    logger.info("Training history plotted.")

def plot_confusion_matrix(y_true, y_pred, class_names):

    predicted_classes = np.argmax(y_pred, axis=1)

    # Create the confusion matrix
    cm = confusion_matrix(y_true, predicted_classes)

    # Create a ConfusionMatrixDisplay object
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

    # Plot the confusion matrix
    plt.figure(figsize=(10, 10))
    disp.plot(cmap='viridis')
    plt.title('Confusion Matrix')
    plt.show()
    logger.info("Confusion matrix plotted.")

def print_classification_report(y_true, y_pred, class_names):

    unique_classes = np.unique(np.concatenate((y_true, y_pred)))
    
    filtered_class_names = [class_names[i] for i in unique_classes]
    
    report = classification_report(y_true, y_pred, target_names=filtered_class_names, labels=unique_classes)
    print(report)
    logger.info("Classification report printed.")


# Usage
def main():
    pass

if __name__ == "__main__":
    main()