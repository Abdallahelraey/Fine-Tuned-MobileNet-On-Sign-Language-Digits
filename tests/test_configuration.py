import os
import pytest
from src.utils.config import Config, load_config

@pytest.fixture
def mock_env(monkeypatch):
    monkeypatch.setenv("DATA_BASE_PATH", "/mock/path")
    monkeypatch.setenv("PREPROCESSING_TARGET_SIZE", "224,224")
    monkeypatch.setenv("NUM_CLASSES_OF_MODEL", "10")
    monkeypatch.setenv("TRAINING_EPOCHS", "20")
    monkeypatch.setenv("VISUALIZATION_PLOT_HISTORY", "true")
    monkeypatch.setenv("EXPERIMENT_TAGS", "mock,test,config")

def test_config_loading(mock_env):
    config = load_config()
    
    # assert config.DATA_BASE_PATH == "/mock/path"
    assert config.PREPROCESSING_TARGET_SIZE == "224,224"
    assert config.NUM_CLASSES_OF_MODEL == 10
    assert config.TRAINING_EPOCHS == 20
    assert config.VISUALIZATION_PLOT_HISTORY == True
    assert config.EXPERIMENT_TAGS == "mock,test,config"

def test_config_types():
    config = Config()
    
    assert isinstance(config.DATA_BASE_PATH, str)
    assert isinstance(config.PREPROCESSING_TARGET_SIZE, str)
    assert isinstance(config.NUM_CLASSES_OF_MODEL, int)
    assert isinstance(config.TRAINING_EPOCHS, int)
    assert isinstance(config.VISUALIZATION_PLOT_HISTORY, bool)
    assert isinstance(config.EXPERIMENT_TAGS, str)

def test_parse_list_function():
    from main import parse_list
    
    assert parse_list("224,224") == [224, 224]
    assert parse_list("224,224,3") == [224, 224, 3]
    assert parse_list("0,1,2,3,4,5") == [0, 1, 2, 3, 4, 5]

def test_config_with_env_file(tmp_path):
    env_content = """
    PREPROCESSING_TARGET_SIZE=128,128
    NUM_CLASSES_OF_MODEL=5
    TRAINING_EPOCHS=10
    VISUALIZATION_PLOT_HISTORY=false
    EXPERIMENT_TAGS=test,env,file
    """
    env_file = tmp_path / ".env"
    env_file.write_text(env_content)
    
    os.environ["ENV_FILE"] = str(env_file)
    
    config = load_config()
    
    # assert config.DATA_BASE_PATH == "/mock/path"
    assert config.PREPROCESSING_TARGET_SIZE == "224,224"
    assert config.NUM_CLASSES_OF_MODEL == 10
    assert config.TRAINING_EPOCHS == 20
    assert config.VISUALIZATION_PLOT_HISTORY == True

if __name__ == "__main__":
    pytest.main([__file__])
