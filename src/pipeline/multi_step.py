from vertexai.generative_models import HarmBlockThreshold
from vertexai.generative_models import GenerationConfig
from src.utils.template import load_system_instruction
from vertexai.generative_models import GenerativeModel
from src.utils.template import load_user_instruction
from src.utils.template import load_response_schema
from vertexai.generative_models import HarmCategory
from src.utils.io import convert_json_to_jsonl
from vertexai.generative_models import Part
from src.utils.io import load_binary_file
from src.config.logging import logger
from src.config.setup import config
from src.utils.io import save_json
from typing import List
from typing import Dict 
from typing import Any 
import json
import time
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


def step_0(model: GenerativeModel, pdf_parts: Part, output_path: str) -> None:
    """
    Extract the metadata fields from the provided PDF document using an LLM (Gemini).

    Args:
        model (GenerativeModel): The generative model instance configured for text generation.
        pdf_parts (Part): The parts of the PDF document to be processed.
        output_path (str): The file path where the output JSON will be saved.

    Raises:
        ValueError: If the model fails to generate a response.
        IOError: If saving the JSON to the output path fails.
    """
    try:
        # Load system and user instructions for the first step of the workflow
        system_instruction = load_system_instruction(workflow='multi_step', step=0)
        model = GenerativeModel(config.TEXT_GEN_MODEL_NAME, system_instruction=system_instruction)
        user_instruction = load_user_instruction(workflow='multi_step', step=0)
        
        # Prepare the contents for the model
        contents: List[Any] = [pdf_parts, user_instruction]
        
        # Load the response schema for the first step
        response_schema: Dict[str, Any] = load_response_schema(workflow='multi_step', step=0)
        
        # Generate the response using the model
        output_json: Dict[str, Any] = generate_response(model, contents, response_schema)
        
        if not output_json:
            raise ValueError("Failed to generate response from the model.")
        
        # Save the generated response as a JSON file
        save_json(output_json, output_path)
        logger.info(f"Output JSON successfully saved to {output_path}")
    
    except ValueError as ve:
        logger.error(f"ValueError occurred: {ve}")
        raise
    
    except IOError as ioe:
        logger.error(f"IOError occurred while saving JSON: {ioe}")
        raise
    
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        raise


def step_1(model: GenerativeModel, pdf_parts: Part, output_path: str) -> None:
    """
    Identify and extract all energy consumption metrics mentioned in the document.
    Return each metric with its code and item name using an LLM (Gemini).

    Args:
        model (GenerativeModel): The generative model instance configured for text generation.
        pdf_parts (Part): The parts of the PDF document to be processed.
        output_path (str): The file path where the output JSON will be saved.

    Raises:
        ValueError: If the model fails to generate a response.
        IOError: If saving the JSON to the output path fails.
    """
    try:
        # Load system and user instructions for the second step of the workflow
        system_instruction = load_system_instruction(workflow='multi_step', step=1)
        model = GenerativeModel(config.TEXT_GEN_MODEL_NAME, system_instruction=system_instruction)
        user_instruction = load_user_instruction(workflow='multi_step', step=1)
        
        # Prepare the contents for the model
        contents: List[Any] = [pdf_parts, user_instruction]
        
        # Load the response schema for the second step
        response_schema: Dict[str, Any] = load_response_schema(workflow='multi_step', step=1)
        
        # Generate the response using the model
        output_json: Dict[str, Any] = generate_response(model, contents, response_schema)
        
        if not output_json:
            raise ValueError("Failed to generate response from the model.")
        
        # Save the generated response as a JSON file
        save_json(output_json, output_path)
        logger.info(f"Output JSON successfully saved to {output_path}")
    
    except ValueError as ve:
        logger.error(f"ValueError occurred: {ve}")
        raise
    
    except IOError as ioe:
        logger.error(f"IOError occurred while saving JSON: {ioe}")
        raise
    
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        raise


def step_2(file_name: str, model: GenerativeModel, pdf_parts: Part, output_path: str) -> None:
    """
    Extracts information for each metric listed in the provided text file from the corresponding PDF.
    
    For each metric, the following information is extracted:
    - Raw numerical value
    - Unit of measurement
    - Page number
    - Relevant text snippet

    Args:
        file_name (str): The name of the file containing the metrics.
        model (GenerativeModel): The generative model instance configured for text generation.
        pdf_parts (Part): The parts of the PDF document to be processed.
        output_path (str): The file path where the output JSON will be saved.

    Raises:
        ValueError: If the model fails to generate a response.
        IOError: If saving the JSON to the output path fails.
    """
    try:
        # Load system and user instructions for the third step of the workflow
        system_instruction = load_system_instruction(workflow='multi_step', step=2)
        model = GenerativeModel(config.TEXT_GEN_MODEL_NAME, system_instruction=system_instruction)
        user_instruction = load_user_instruction(workflow='multi_step', step=2)
        
        # Load the output from step 1
        out_step_1_file = load_binary_file(os.path.join(OUTPUT_DIR, f'multi_step/{file_name}/out_step_1.txt'))
        out_step_1 = Part.from_data(data=out_step_1_file, mime_type='text/plain')
        
        # Prepare the contents for the model
        contents: List[Any] = [pdf_parts, out_step_1, user_instruction]
        
        # Load the response schema for the third step
        response_schema: Dict[str, Any] = load_response_schema(workflow='multi_step', step=2)
        
        # Generate the response using the model
        output_json: Dict[str, Any] = generate_response(model, contents, response_schema)
        
        if not output_json:
            raise ValueError("Failed to generate response from the model.")
        
        # Save the generated response as a JSON file
        save_json(output_json, output_path)
        logger.info(f"Output JSON successfully saved to {output_path}")
    
    except ValueError as ve:
        logger.error(f"ValueError occurred: {ve}")
        raise
    
    except IOError as ioe:
        logger.error(f"IOError occurred while saving JSON: {ioe}")
        raise
    
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        raise


def step_3(file_name: str, model: GenerativeModel, pdf_parts: Part, output_path: str) -> None:
    """
    For each extracted metric, extract additional information from the provided PDF.
    
    Specifically, the following information is determined:
    - Reporting year, focusing on the most recent if multiple years are present.
    - Scope (Global, Regional, or Country-Specific) and a flag (Full or Partial) for each value, with reasoning for the flag assignment.
    - Classification of each value as either 'Operational Consumption' or 'Supply Chain Consumption' based on the context in the document.

    Args:
        file_name (str): The name of the file containing the metrics.
        model (GenerativeModel): The generative model instance configured for text generation.
        pdf_parts (Part): The parts of the PDF document to be processed.
        output_path (str): The file path where the output JSON will be saved.

    Raises:
        ValueError: If the model fails to generate a response.
        IOError: If saving the JSON to the output path fails.
    """
    try:
        # Load system and user instructions for the fourth step of the workflow
        system_instruction = load_system_instruction(workflow='multi_step', step=3)
        model = GenerativeModel(config.TEXT_GEN_MODEL_NAME, system_instruction=system_instruction)
        user_instruction = load_user_instruction(workflow='multi_step', step=3)
        
        # Load the output from step 2
        out_step_2_file = load_binary_file(os.path.join(OUTPUT_DIR, f'multi_step/{file_name}/out_step_2.txt'))
        out_step_2 = Part.from_data(data=out_step_2_file, mime_type='text/plain')
        
        # Prepare the contents for the model
        contents: List[Any] = [pdf_parts, out_step_2, user_instruction]
        
        # Load the response schema for the fourth step
        response_schema: Dict[str, Any] = load_response_schema(workflow='multi_step', step=3)
        
        # Generate the response using the model
        output_json: Dict[str, Any] = generate_response(model, contents, response_schema)
        
        if not output_json:
            raise ValueError("Failed to generate response from the model.")
        
        # Save the generated response as a JSON file
        save_json(output_json, output_path)
        logger.info(f"Output JSON successfully saved to {output_path}")
    
    except ValueError as ve:
        logger.error(f"ValueError occurred: {ve}")
        raise
    
    except IOError as ioe:
        logger.error(f"IOError occurred while saving JSON: {ioe}")
        raise
    
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        raise


def run(file_name: str) -> None:
    """
    Run the entire extraction process for the given PDF file.

    The process includes:
    1. Extracting metadata fields.
    2. Identifying and extracting energy consumption metrics.
    3. Extracting detailed information for each metric.
    4. Extracting additional information and classifying each metric.

    Args:
        file_name (str): The name of the PDF file (without extension) to be processed.

    Raises:
        Exception: If any step in the process fails, the exception is logged and re-raised.
    """
    try:
        logger.info(f"Running extraction for file: {file_name}")
        file_path = os.path.join(config.DATA_DIR, f'docs/{file_name}.pdf')
        pdf_bytes = load_binary_file(file_path)
        pdf_parts = Part.from_data(data=pdf_bytes, mime_type='application/pdf')
        start_time = time.time()
        
        # Run each step in the extraction process
        step_0(config.TEXT_GEN_MODEL_NAME, pdf_parts, os.path.join(OUTPUT_DIR, f'multi_step/{file_name}/out_step_0.txt'))
        step_1(config.TEXT_GEN_MODEL_NAME, pdf_parts, os.path.join(OUTPUT_DIR, f'multi_step/{file_name}/out_step_1.txt'))
        step_2(file_name, config.TEXT_GEN_MODEL_NAME, pdf_parts, os.path.join(OUTPUT_DIR, f'multi_step/{file_name}/out_step_2.txt'))
        
        output_path = os.path.join(OUTPUT_DIR, f'multi_step/{file_name}/out_step_3.txt')
        step_3(file_name, config.TEXT_GEN_MODEL_NAME, pdf_parts, output_path)
        
        # Convert the final output to JSONL format
        convert_json_to_jsonl(output_path, os.path.join(VALIDATION_DIR, f'generated/multi_step/{file_name}.jsonl'), workflow='multi_step')
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(f"Extraction process completed successfully in {elapsed_time:.2f} seconds")
    
    except Exception as e:
        logger.error(f"Error in run process: {e}")
        raise  # Re-raise the exception after logging


if __name__ == '__main__':
    file_name = '100395060535523152'
    run(file_name)