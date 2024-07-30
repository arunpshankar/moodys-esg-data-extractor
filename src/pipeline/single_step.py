from vertexai.generative_models import HarmBlockThreshold
from vertexai.generative_models import GenerationConfig
from src.utils.template import load_system_instruction
from vertexai.generative_models import GenerativeModel
from src.utils.template import load_user_instruction
from src.utils.template import load_response_schema
from vertexai.generative_models import HarmCategory
from src.utils.io import convert_json_to_jsonl
from vertexai.generative_models import Part
from src.config.logging import logger
from src.config.setup import config
from src.utils.io import load_file
from typing import Optional 
from typing import List
from typing import Dict 
from typing import Any 
import json
import os


OUTPUT_DIR = os.path.join(config.DATA_DIR, 'output')
VALIDATION_DIR = os.path.join(config.DATA_DIR, 'validation')

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
            generation_config=create_generation_config(response_schema),
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


def run(model: GenerativeModel, pdf_parts: Part, output_path: str):
    try:
        logger.info("Starting processing ...")
        system_instruction = 
        model = GenerativeModel(config.TEXT_GEN_MODEL_NAME, system_instruction=system_instruction)
        user_prompt = "Analyze the following PDF and follow the rules."
        contents = [pdf_parts, user_prompt]
    except Exception as e:
        logger.error(e)


if __name__ == '__main__':
    run()
