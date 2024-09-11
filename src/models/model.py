import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
import os
import numpy as np
from src.models.model_utils import save_model, load_model
from src.utils.logging_config import logger  
from dotenv import load_dotenv
from src.utils.config import load_config
load_dotenv()




class SignLanguageModel:
    def __init__(self):
        self.config = load_config()
        self.save_path = self.config.PATHS_MODEL_SAVE_PATH
        self.input_shape = tuple(map(int, self.config.INPUT_SHAPE_OF_MODEL.split(',')))
        self.num_classes = self.config.NUM_CLASSES_OF_MODEL
        self.learning_rate = self.config.TRAINING_INITIAL_LEARNING_RATE
        self.model = self.build_model()
        
    def build_model(self):
        base_model = MobileNetV2(input_shape=self.input_shape, include_top=False, weights='imagenet')
        

        for layer in base_model.layers[-20:]:
            layer.trainable = True

        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.5)(x)
        output = Dense(self.num_classes, activation='softmax')(x)

        model = Model(inputs=base_model.input, outputs=output)
        return model

    def compile_model(self):
        self.model.compile(
            optimizer=self.config.TRAINING_OPTIMIZER,
            loss=self.config.TRAINING_LOSS,
            metrics=[self.config.TRAINING_METRICS]
            # metrics=[tf.keras.metrics.Accuracy()]
        )
        logger.info("Model compiled successfully.")  

    def train(self, train_batches, validation_batches,callbacks=None):
        lr_scheduler = ReduceLROnPlateau(
            monitor=self.config.TRAINING_REDUCE_LR_MONITOR,
            factor=self.config.TRAINING_REDUCE_LR_FACTOR,
            patience=self.config.TRAINING_REDUCE_LR_PATIENCE,
            min_lr=self.config.TRAINING_MIN_LR  
        )

        history = self.model.fit(
            train_batches,
            validation_data=validation_batches,
            epochs=self.config.TRAINING_EPOCHS,
            # batch_size=self.config.TRAINING_BATCH_SIZE,
            callbacks= [lr_scheduler]
        )
        logger.info("Model training completed.")  
        return history

    def evaluate(self, test_data):
        result = self.model.evaluate(test_data)
        logger.info("Model evaluation completed.")  
        return result
    

    def predict(self, image):
        predictions = self.model.predict(image)
        Model_prediction = np.argmax(self.model.predict(image))
        logger.info("Model prediction completed.")  
        return predictions ,Model_prediction

    def save_model(self, model_name):
        save_model(self.model, model_name)

    @staticmethod
    def load_model(model_name):
        model = load_model(model_name)
        return model

def main():
    from src.data.preprocess import DataProcessor

    processor = DataProcessor()
    train_data, validation_data, test_data = processor.process_and_load_data()

    model = SignLanguageModel()
    model.compile_model()
    history = model.train(train_data, validation_data)
    
    test_loss, test_accuracy = model.evaluate(test_data)
    logger.info(f"Test accuracy: {test_accuracy}")  
    print(f"Test accuracy: {test_accuracy}")

    model.save_model('finetuned_mobilenetv2_model.keras')

if __name__ == "__main__":
    main()
