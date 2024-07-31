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


def step_0(model: GenerativeModel, pdf_parts: Part, output_path: str):
    system_instruction = load_system_instruction(workflow='multi_step', step=0)
    model = GenerativeModel(config.TEXT_GEN_MODEL_NAME, system_instruction=system_instruction)
    user_instruction = load_user_instruction(workflow='multi_step', step=0)
    contents = [pdf_parts, user_instruction]
    response_schema = load_response_schema(workflow='multi_step', step=0)
    output_json = generate_response(model, contents, response_schema)
    save_json(output_json, output_path)


def step_1(model: GenerativeModel, pdf_parts: Part, output_path: str):
    system_instruction = load_system_instruction(workflow='multi_step', step=1)
    model = GenerativeModel(config.TEXT_GEN_MODEL_NAME, system_instruction=system_instruction)
    user_instruction = load_user_instruction(workflow='multi_step', step=1)
    contents = [pdf_parts, user_instruction]
    response_schema = load_response_schema(workflow='multi_step', step=1)
    output_json = generate_response(model, contents, response_schema)
    save_json(output_json, output_path)


def step_2(file_name: str, model: GenerativeModel, pdf_parts: Part, output_path: str):
    system_instruction = load_system_instruction(workflow='multi_step', step=2)
    model = GenerativeModel(config.TEXT_GEN_MODEL_NAME, system_instruction=system_instruction)
    user_instruction = load_user_instruction(workflow='multi_step', step=2)
    out_step_1_file = load_binary_file(os.path.join(OUTPUT_DIR, f'multi_step/{file_name}/out_step_1.txt'))
    out_step_1 = Part.from_data(data=out_step_1_file, mime_type='text/plain')
    contents = [pdf_parts, out_step_1, user_instruction]
    response_schema = load_response_schema(workflow='multi_step', step=2)
    output_json = generate_response(model, contents, response_schema)
    save_json(output_json, output_path)


def step_3(file_name: str, model: GenerativeModel, pdf_parts: Part, output_path: str):
    system_instruction = load_system_instruction(workflow='multi_step', step=3)
    model = GenerativeModel(config.TEXT_GEN_MODEL_NAME, system_instruction=system_instruction)
    user_instruction = load_user_instruction(workflow='multi_step', step=3)
    out_step_2_file = load_binary_file(os.path.join(OUTPUT_DIR, f'multi_step/{file_name}/out_step_2.txt'))
    out_step_2 = Part.from_data(data=out_step_2_file, mime_type='text/plain')
    contents = [pdf_parts, out_step_2, user_instruction]
    response_schema = load_response_schema(workflow='multi_step', step=2)
    output_json = generate_response(model, contents, response_schema)
    save_json(output_json, output_path)


def run(file_name: str):
    try:
        logger.info(f"Running extraction for file: {file_name}")
        file_path = os.path.join(config.DATA_DIR, f'docs/{file_name}.pdf')
        pdf_bytes = load_binary_file(file_path)
        pdf_parts = Part.from_data(data=pdf_bytes, mime_type='application/pdf')
        start_time = time.time()
        step_0(config.TEXT_GEN_MODEL_NAME, pdf_parts, os.path.join(OUTPUT_DIR, f'multi_step/{file_name}/out_step_0.txt'))
        step_1(config.TEXT_GEN_MODEL_NAME, pdf_parts, os.path.join(OUTPUT_DIR, f'multi_step/{file_name}/out_step_1.txt'))
        step_2(file_name, config.TEXT_GEN_MODEL_NAME, pdf_parts, os.path.join(OUTPUT_DIR, f'multi_step/{file_name}/out_step_2.txt'))
        output_path = os.path.join(OUTPUT_DIR, f'multi_step/{file_name}/out_step_3.txt')
        step_3(file_name, config.TEXT_GEN_MODEL_NAME, pdf_parts, output_path)
        convert_json_to_jsonl(output_path, os.path.join(VALIDATION_DIR, f'generated/multi_step/{file_name}.jsonl'), workflow='multi_step')
        end_time = time.time()
        elapsed_time = end_time - start_time
        logger.info(f"Extraction process completed successfully in {elapsed_time:.2f} seconds")
    except Exception as e:
        logger.error(f"Error in run process: {e}")
        raise  # Re-raise the exception after logging


if __name__ == '__main__':
    run('84535104943034784')
