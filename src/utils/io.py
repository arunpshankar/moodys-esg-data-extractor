from src.config.logging import logger 
from typing import Generator
from typing import Optional 
from typing import List 
from typing import Dict 
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


def load_jsonl(file_path: str) -> List[Dict]:
    """
    Reads a JSONL (JSON Lines) file and returns a list of dictionaries.

    Each line in the file should be a valid JSON object.

    Parameters:
    file_path (str): The path to the JSONL file.

    Returns:
    List[Dict]: A list of dictionaries, where each dictionary represents a JSON object from the file.
    
    Raises:
    FileNotFoundError: If the file at the specified path does not exist.
    JSONDecodeError: If a line in the file is not a valid JSON object.
    """
    json_list = []
    
    try:
        with open(file_path, 'r') as file:
            for line in file:
                try:
                    json_obj = json.loads(line)
                    json_list.append(json_obj)
                except json.JSONDecodeError as e:
                    logger.error(f"Error decoding JSON from line: {line.strip()} - {e}")
    except FileNotFoundError as e:
        logger.error(f"File not found: {file_path} - {e}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred while reading the file: {file_path} - {e}")
        raise
    
    return json_list


def convert_json_to_jsonl(input_file: str, output_file: str, workflow: str) -> None:
    """
    Convert a JSON file to a JSONL file with branching based on the workflow.

    Args:
        input_file (str): The path to the input JSON file.
        output_file (str): The path to the output JSONL file.
        workflow (str): The workflow type, either 'single_step' or other.
    """
    try:
        logger.info(f"Reading the input JSON file: {input_file}")
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Ensure the output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        logger.info(f"Ensured the output directory exists: {os.path.dirname(output_file)}")

        logger.info(f"Writing data to the output JSONL file: {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            if workflow == 'single_step':
                for item in data["metrics"]:
                    json_line = json.dumps(item)
                    f.write(json_line + '\n')
            else:
                for item in data:
                    json_line = json.dumps(item)
                    f.write(json_line + '\n')
        logger.info(f"Successfully converted JSON to JSONL: {output_file}")

    except FileNotFoundError:
        logger.error(f"File not found: {input_file}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from {input_file}: {e}")
        raise
    except IOError as e:
        logger.error(f"Error reading or writing file: {e}")
        raise
    except KeyError as e:
        logger.error(f"Missing expected key in JSON data: {e}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        raise


def get_pdf_file_names(directory: str) -> Generator[str, None, None]:
    """
    Generator function that iterates through a list of PDF documents in a given directory
    and yields file names without the .pdf extension.

    :param directory: The directory containing PDF files.
    :return: An iterator of file names without the .pdf extension.
    :raises ValueError: If the provided directory is not valid.
    """
    try:
        if not os.path.isdir(directory):
            raise ValueError(f"Invalid directory: {directory}")

        for file_name in os.listdir(directory):
            if file_name.endswith('.pdf'):
                yield os.path.splitext(file_name)[0]
    except Exception as e:
        logger.error(f"An error occurred while processing the directory {directory}: {e}")
        raise