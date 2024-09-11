import os
import pytest
from src.data.make_dataset import DatasetOrganizer, organize_dataset

@pytest.fixture
def temp_dir(tmp_path):
    # Create a temporary directory structure
    for i in range(10):
        os.makedirs(tmp_path / str(i), exist_ok=True)
        for j in range(50):  # Create 50 dummy files in each class
            (tmp_path / str(i) / f"file_{j}.txt").touch()
    return tmp_path

def test_dataset_organizer_initialization(temp_dir):
    organizer = DatasetOrganizer(temp_dir)
    assert organizer._data_path == temp_dir
    assert organizer._num_classes == 10
    assert organizer._valid_size == 30
    assert organizer._test_size == 5
    assert not organizer._is_organized

def test_dataset_organization(temp_dir):
    organizer = DatasetOrganizer(temp_dir)
    organizer.organize_dataset()
    
    assert organizer._is_organized
    
    # Check if directories are created
    for dir_name in ['train', 'validation', 'test']:
        assert (temp_dir / dir_name).is_dir()
        for i in range(10):
            assert (temp_dir / dir_name / f"class_{i}").is_dir()

    # Check if files are distributed correctly
    for i in range(10):
        train_files = list((temp_dir / 'train' / f"class_{i}").glob('*'))
        valid_files = list((temp_dir / 'validation' / f"class_{i}").glob('*'))
        test_files = list((temp_dir / 'test' / f"class_{i}").glob('*'))
        
        assert len(train_files) == 15  # 50 - 30 - 5
        assert len(valid_files) == 30
        assert len(test_files) == 5

def test_organize_dataset_function(temp_dir):
    is_organized = organize_dataset(temp_dir)
    assert is_organized

    # Check if the function doesn't reorganize an already organized dataset
    is_reorganized = organize_dataset(temp_dir)
    assert not is_reorganized

def test_custom_parameters(temp_dir):
    organizer = DatasetOrganizer(temp_dir, num_classes=5, valid_size=20, test_size=10)
    organizer.organize_dataset()

    assert organizer._is_organized
    assert len(list((temp_dir / 'train').glob('class_*'))) == 5
    
    for i in range(5):
        valid_files = list((temp_dir / 'validation' / f"class_{i}").glob('*'))
        test_files = list((temp_dir / 'test' / f"class_{i}").glob('*'))
        
        assert len(valid_files) == 20
        assert len(test_files) == 10

# Add more tests as needed
