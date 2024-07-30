from vertexai.generative_models import HarmBlockThreshold
from vertexai.generative_models import GenerationConfig
from vertexai.generative_models import GenerativeModel
from vertexai.generative_models import HarmCategory
from vertexai.generative_models import Part
from src.config.logging import logger
from src.config.setup import config
from typing import Optional 
from typing import List
from typing import Dict 
from typing import Any 
import json
import os


DATA_DIR = './data'
OUTPUT_DIR = os.path.join(DATA_DIR, 'output')
VALIDATION_DIR = os.path.join(DATA_DIR, 'validation')


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


def create_generation_config(response_schema: Dict[str, Any]) -> GenerationConfig:
    """
    Create a GenerationConfig instance.

    Args:
        response_schema (Dict[str, Any]): The schema for the response.

    Returns:
        GenerationConfig: An instance of GenerationConfig with the specified parameters.
    """
    try:
        logger.info("Creating generation configuration")
        config = GenerationConfig(
            temperature=0.0, 
            top_p=0.0, 
            top_k=1, 
            candidate_count=1, 
            max_output_tokens=8192,
            response_mime_type="application/json",
            response_schema=response_schema
        )
        logger.info("Successfully created generation configuration")
        return config
    except Exception as e:
        logger.error(f"Error creating generation configuration: {e}")
        raise


def create_safety_settings() -> Dict[HarmCategory, HarmBlockThreshold]:
    """
    Create a safety settings dictionary.

    Returns:
        Dict[HarmCategory, HarmBlockThreshold]: A dictionary mapping harm categories to block thresholds.
    """
    try:
        logger.info("Creating safety settings")
        safety_settings = {
            HarmCategory.HARM_CATEGORY_UNSPECIFIED: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE
        }
        logger.info("Successfully created safety settings")
        return safety_settings
    except Exception as e:
        logger.error(f"Error creating safety settings: {e}")
        raise  # Re-raise the exception after logging


def generate_response(model: GenerativeModel, contents: List[Part], response_schema: Dict[str, Any]) -> Any:
    """
    Generate content using the generative model.

    Args:
        model (GenerativeModel): The generative model to use.
        contents (List[Part]): The contents to be processed by the model.
        response_schema (Dict[str, Any]): The schema for the response.

    Returns:
        Any: The generated response.
    """
    try:
        logger.info("Generating response using the generative model")
        response = model.generate_content(
            contents,
            generation_config=create_generation_config(response_schema)
            safety_settings=create_safety_settings()
        )
        output_json = json.loads(response.text.strip())
        logger.info(f"Response generated: {output_json}")
        logger.info(f"Finish reason: {response.candidates[0].finish_reason}")
        logger.info(f"Safety ratings: {response.candidates[0].safety_ratings}")
        return output_json
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON response: {e}")
        raise  # Re-raise the exception after logging
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        raise  # Re-raise the exception after logging

def run():
    pass


if __name__ == '__main__':
    run()
