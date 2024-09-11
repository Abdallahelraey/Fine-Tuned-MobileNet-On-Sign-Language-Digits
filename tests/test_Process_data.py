import os
import numpy as np
import pytest
from src.data.preprocess import DataProcessor

@pytest.fixture
def data_processor():
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    return DataProcessor(base_path)

def test_create_data_generator(data_processor):
    generator = data_processor.create_data_generator()
    assert generator is not None
    assert hasattr(generator, 'flow_from_directory')

def test_create_processed_data_path(data_processor):
    path = data_processor.create_processed_data_path()
    assert os.path.exists(path)
    expected_end = os.path.join('data', 'processed')
    assert path.endswith(expected_end)

def test_save_and_load_batch(data_processor):
    # Create dummy data
    images = np.random.rand(10, 224, 224, 3)
    labels = np.random.rand(10, 10)
    
    # Save dummy data
    data_processor.save_batch(iter([(images, labels)]), 'test_batch.npz')
    
    # Load saved data
    loaded_images, loaded_labels = data_processor.load_batch('test_batch.npz')
    
    assert np.array_equal(images, loaded_images)
    assert np.array_equal(labels, loaded_labels)

@pytest.mark.skipif(not os.path.exists(os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'Dataset')),
                    reason="Raw dataset not found")
def test_load_dataset(data_processor):
    train_batches, validation_batches, test_batches = data_processor.load_dataset()
    
    assert train_batches is not None
    assert validation_batches is not None
    assert test_batches is not None
    
    assert train_batches.batch_size == data_processor.batch_size
    assert validation_batches.batch_size == data_processor.batch_size
    assert test_batches.batch_size == data_processor.batch_size

@pytest.mark.skipif(not os.path.exists(os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'Dataset')),
                    reason="Raw dataset not found")
def test_process_and_load_data(data_processor):
    train_data, validation_data, test_data = data_processor.process_and_load_data()
    
    assert isinstance(train_data, tuple) and len(train_data) == 2
    assert isinstance(validation_data, tuple) and len(validation_data) == 2
    assert isinstance(test_data, tuple) and len(test_data) == 2
    
    assert train_data[0].shape[1:] == data_processor.target_size + (3,)
    assert validation_data[0].shape[1:] == data_processor.target_size + (3,)
    assert test_data[0].shape[1:] == data_processor.target_size + (3,)