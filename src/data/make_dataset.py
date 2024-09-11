import os
import shutil
import random
from dotenv import load_dotenv
from src.utils.config import load_config
from src.utils.logging_config import logger
load_dotenv()



class DatasetOrganizer:
    def __init__(self, num_classes=10, valid_size=30, test_size=5):
        self.config = load_config()
        self._base_path = self.config.DATA_BASE_PATH
        self._data_path = os.path.join(self._base_path, 'data', 'raw', 'Dataset')
        self._num_classes = num_classes
        self._valid_size = valid_size
        self._test_size = test_size
        self._is_organized = self._check_if_organized()

    def organize_dataset(self):
        """Main function to organize the dataset."""
        if self._is_organized:
            logger.info("Data appears to be already organized. Skipping organization process.")
            return False  

        if not self._is_organized:
            self._create_directories()
            self._create_class_directories()
            self._split_data()
            self._is_organized = True
            logger.info("Dataset organization completed.")
            return True  
        else:
            logger.info("Dataset directories already exist. Skipping organization process.")
            return False  

    def _check_if_organized(self):
        return all(os.path.exists(os.path.join(self._data_path, dir_name)) 
                   for dir_name in ['train', 'validation', 'test'])

    def _create_directories(self):
        """Create train, validation, and test directories."""
        for dir_name in ['train', 'validation', 'test']:
            os.makedirs(os.path.join(self._data_path, dir_name), exist_ok=True)

    def _create_class_directories(self):
        """Create class directories within train, validation, and test."""
        for dir_name in ['train', 'validation', 'test']:
            for i in range(self._num_classes):
                os.makedirs(os.path.join(self._data_path, dir_name, f'class_{i}'), exist_ok=True)

    def _move_files(self, src_dir, dest_dir):
        """Move files from source to destination directory."""
        for file in os.listdir(src_dir):
            shutil.move(os.path.join(src_dir, file), dest_dir)

    def _split_data(self):
        """Split data into train, validation, and test sets."""
        for i in range(self._num_classes):
            train_dir = os.path.join(self._data_path, 'train', f'class_{i}')
            valid_dir = os.path.join(self._data_path, 'validation', f'class_{i}')
            test_dir = os.path.join(self._data_path, 'test', f'class_{i}')


            self._move_files(os.path.join(self._data_path, str(i)), train_dir)


            valid_samples = random.sample(os.listdir(train_dir), self._valid_size)
            for sample in valid_samples:
                shutil.move(os.path.join(train_dir, sample), os.path.join(valid_dir, sample))


            test_samples = random.sample(os.listdir(train_dir), self._test_size)
            for sample in test_samples:
                shutil.move(os.path.join(train_dir, sample), os.path.join(test_dir, sample))

def organize_dataset(base_path, num_classes=10, valid_size=30, test_size=5):
    organizer = DatasetOrganizer(base_path, num_classes, valid_size, test_size)
    return organizer.organize_dataset()  

# Usage
if __name__ == "__main__":
    is_organized = organize_dataset()
    logger.info(f"Dataset organization status: {'Completed' if is_organized else 'Not needed'}")
