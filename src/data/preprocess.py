import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from src.utils.logging_config import logger 
from keras.preprocessing import image
from dotenv import load_dotenv
from src.utils.config import load_config

load_dotenv()

class DataProcessor:
    def __init__(self, target_size=(224, 224), batch_size=10):
        self.config = load_config()
        self.base_path = self.config.DATA_BASE_PATH
        self.target_size = tuple(map(int,self.config.PREPROCESSING_TARGET_SIZE.split(','))) 
        self.batch_size = self.config.PREPROCESSING_BATCH_SIZE
        self.data_path = os.path.join(self.base_path, self.config.DATA_DATASET_BASE_PATH)
        self.processed_path = os.path.join(self.base_path, self.config.DATA_PROCESSED_PATH)
        self._is_processed = self._check_if_procecssed()
        logger.info("DataProcessor initialized.")  

    def _check_if_procecssed(self):
        if os.path.exists(self.processed_path) and os.listdir(self.processed_path):
            logger.info("Data appears to be already processed. Skipping processing process.")
            return True
        else:
            logger.info("Data needs to be processed.")
            return False

    def create_data_generator(self):
        preprocessing_function = tf.keras.applications.mobilenet_v2.preprocess_input
        return ImageDataGenerator(preprocessing_function=preprocessing_function)

    def create_data_flow(self, data_generator, data_path, class_mode, shuffle=True):
        return data_generator.flow_from_directory(
            data_path,
            target_size=self.target_size,
            batch_size=self.batch_size,
            class_mode=class_mode,
            shuffle=shuffle
        )

    def load_dataset(self):
        data_generator = self.create_data_generator()
        
        train_path = os.path.join(self.data_path, 'train')
        validation_path = os.path.join(self.data_path, 'validation')
        test_path = os.path.join(self.data_path, 'test')
        
        train_batches = self.create_data_flow(data_generator, train_path, 'categorical')
        validation_batches = self.create_data_flow(data_generator, validation_path, 'categorical')
        test_batches = self.create_data_flow(data_generator, test_path, 'categorical', shuffle=False)
        
        logger.info("Dataset loaded successfully.")  
        return train_batches, validation_batches, test_batches


    def save_batch(self, batches, filename):
        images, labels = next(batches)  
        np.savez(os.path.join(self.processed_path, filename), images=images, labels=labels)
        logger.info(f"Batch saved as {filename}.")  

    def load_batch(self, filename):
        data = np.load(os.path.join(self.processed_path, filename))
        return data['images'], data['labels']

    def save_sample_batches(self, train_batches, validation_batches, test_batches):
        self.save_batch(train_batches, 'train_batch_sample.npz')
        self.save_batch(validation_batches, 'validation_batch_sample.npz')
        self.save_batch(test_batches, 'test_batch_sample.npz')
        logger.info("Sample batches saved successfully.")  

    def load_sample_batches(self):
        train_data = self.load_batch('train_batch_sample.npz')
        validation_data = self.load_batch('validation_batch_sample.npz')
        test_data = self.load_batch('test_batch_sample.npz')
        logger.info("Sample batches loaded successfully.") 
        return train_data, validation_data, test_data

    def prepare_image_for_prediction(self, img_path):
        img = image.load_img(img_path, target_size= self.target_size)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  
        img_array /= 255.0  
        return img_array



# Usage
def main():
    processor = DataProcessor()
    train_data, validation_data, test_data = processor.load_dataset()

if __name__ == "__main__":
    main()
