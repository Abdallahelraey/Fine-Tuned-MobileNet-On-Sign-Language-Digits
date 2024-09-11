import os
import tensorflow as tf
from dotenv import load_dotenv
from src.data.make_dataset import DatasetOrganizer
from src.data.preprocess import DataProcessor
from src.models.model import SignLanguageModel
from src.utils.logging_config import logger  
import numpy as np

def main():
    logger.info("Main entry point of the application started.")  

    # Load environment variables and configuration
    load_dotenv()
    logger.info("Environment variables loaded.")  

    # Organize dataset if needed
    organizer = DatasetOrganizer()
    organizer.organize_dataset()
    logger.info("Dataset organized successfully.")  

    # Initialize data processor
    processor = DataProcessor()
    logger.info("DataProcessor initialized.")  

    # Load and preprocess data
    train_data, validation_data, test_data = processor.load_dataset()
    logger.info("Data loaded and preprocessed successfully.")  

    # # Create and compile the model
    model = SignLanguageModel()
    # model.build_model()
    # model.compile_model()
    # logger.info("Model compiled successfully.")  

    # # Train the model
    # history = model.train(train_data, validation_data)
    # logger.info("Model training completed.") 

    # Load the model
    model = model.load_model("finetuned_mobilenetv2_model.keras")
    logger.info("Model loaded successfully.") 
    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(test_data)
    logger.info(f"Test accuracy: {test_accuracy:.4f}")  

    # Make predictions on test data
    img_path = "D:/Projects/AI Tutorials/TensorFlow/Fine-Tuned-MobileNet-On-Sign-Language-Digits/data/Examples/example_0.JPG"
    image = processor.prepare_image_for_prediction(img_path)
    model_predictions = model.predict(image)
    Model_prediction = np.argmax(model.predict(image))
    logger.info("Predictions made on test data.")  

    print(f"Test accuracy: {test_accuracy}")
    print(model_predictions)
    print(Model_prediction)

    logger.info("Main entry point of the application completed successfully.")  

if __name__ == "__main__":
    main()
