import logging
from dotenv import load_dotenv
from src.utils.config import load_config

# Load environment variables and configuration
load_dotenv()
config = load_config()

# Create a logger
logger = logging.getLogger(__name__)
logger.setLevel(config.LOGGING_LEVEL)

# Create a file handler
handler = logging.FileHandler(config.LOGGING_FILE_PATH)
handler.setLevel(config.LOGGING_LEVEL)

# Create a formatter and set it for the handler
formatter = logging.Formatter(config.LOGGING_FORMAT)
handler.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(handler)
