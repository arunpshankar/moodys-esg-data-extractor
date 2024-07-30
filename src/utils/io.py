from src.config.logging import logger 
from typing import Optional 
from typing import Any 
import json 
import os 


def load_file(file_path: str) -> Optional[str]:
    """
    Load text content from a file.

    Args:
        file_path (str): The path to the file to be loaded.

    Returns:
        Optional[str]: The content of the file as a string, or None if an error occurs.
    """
    try:
        logger.info(f"Attempting to load text file from {file_path}")
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            logger.info(f"Successfully loaded file: {file_path}")
            return content
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
    except IOError as e:
        logger.error(f"IO error occurred while reading file {file_path}: {e}")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
    return None


def load_binary_file(file_path: str) -> Optional[bytes]:
    """
    Load binary content from a file.

    Args:
        file_path (str): The path to the file to be loaded.

    Returns:
        Optional[bytes]: The binary content of the file, or None if an error occurs.
    """
    try:
        logger.info(f"Attempting to load binary file from {file_path}")
        with open(file_path, 'rb') as file:
            content = file.read()
            logger.info(f"Successfully loaded binary file: {file_path}")
            return content
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
    except IOError as e:
        logger.error(f"Error reading file {file_path}: {e}")
    return None


def save_json(data: Any, file_path: str) -> bool:
    """
    Save JSON data to a file.

    Args:
        data (Any): The JSON data to be saved.
        file_path (str): The path to the file where the data should be saved.

    Returns:
        bool: True if the file was saved successfully, False otherwise.
    """
    try:
        logger.info(f"Attempting to save JSON data to {file_path}")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, indent=4)
            logger.info(f"Successfully saved JSON data to {file_path}")
            return True
    except IOError as e:
        logger.error(f"Error saving JSON data to {file_path}: {e}")
    return False

