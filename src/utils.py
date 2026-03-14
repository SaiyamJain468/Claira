import logging
from pathlib import Path

def setup_logging(log_file: Path = Path("logs/claira.log")):
    """
    Sets up logging for the Claira project.
    
    Args:
        log_file (Path): Path to the log file.
    """
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def get_project_root() -> Path:
    """
    Returns the root directory of the project.
    
    Returns:
        Path: The project root directory.
    """
    return Path(__file__).parent.parent
