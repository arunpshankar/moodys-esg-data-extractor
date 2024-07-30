from src.config.logging import logger 
from src.config.setup import config
from src.utils.io import load_file
from typing import Optional 
from typing import List 
from typing import Dict 
from typing import Any 
import json 
import os 


def load_system_instruction(workflow: str, step: Optional[int] = None) -> List[str]:
    """
    Load system instructions based on the workflow and step.

    Args:
        workflow (str): The workflow name, can be either 'single_step' or 'multi_step'.
        step (Optional[int]): The step number, can be 0, 1, 2, or 3 if applicable.

    Returns:
        List[str]: A list containing the system instruction(s).
    """
    try:
        if step is not None:
            logger.info(f"Loading multi-step system instruction for workflow: {workflow}, step: {step}")
            system_instruction = [load_file(os.path.join(config.DATA_DIR, f'templates/{workflow}/system_instruction/system_instruction_step_{step}.txt'))]
        else:
            logger.info(f"Loading single-step system instruction for workflow: {workflow}")
            system_instruction = [load_file(os.path.join(config.DATA_DIR, f'templates/{workflow}/system_instruction.txt'))]
        logger.info("System instruction loaded successfully")
        return system_instruction
    except Exception as e:
        logger.error(f"Error loading system instruction for workflow {workflow} with step {step}: {e}")
        raise


def load_user_instruction(workflow: str, step: Optional[int] = None) -> str:
    """
    Load user instructions based on the workflow and step.

    Args:
        workflow (str): The workflow name, can be either 'single_step' or 'multi_step'.
        step (Optional[int]): The step number, can be 0, 1, 2, or 3 if applicable.

    Returns:
        str: The user instruction.
    """
    try:
        if step is not None:
            logger.info(f"Loading multi-step user instruction for workflow: {workflow}, step: {step}")
            user_instruction = load_file(os.path.join(config.DATA_DIR, f'templates/{workflow}/user_instruction/user_instruction_step_{step}.txt'))
        else:
            logger.info(f"Loading single-step user instruction for workflow: {workflow}")
            user_instruction = load_file(os.path.join(config.DATA_DIR, f'templates/{workflow}/user_instruction.txt'))
        logger.info("User instruction loaded successfully")
        return user_instruction
    except Exception as e:
        logger.error(f"Error loading user instruction for workflow {workflow} with step {step}: {e}")
        raise


def load_response_schema(workflow: str, step: Optional[int] = None) -> Dict[str, Any]:
    """
    Load response schema based on the workflow and step.

    Args:
        workflow (str): The workflow name, can be either 'single_step' or 'multi_step'.
        step (Optional[int]): The step number, can be 0, 1, 2, or 3 if applicable.

    Returns:
        Dict[str, Any]: The response schema as a dictionary.
    """
    try:
        if step is not None:
            logger.info(f"Loading multi-step response schema for workflow: {workflow}, step: {step}")
            response_schema_path = os.path.join(config.DATA_DIR, f'templates/{workflow}/schema/step_{step}_response.json')
        else:
            logger.info(f"Loading single-step response schema for workflow: {workflow}")
            response_schema_path = os.path.join(config.DATA_DIR, f'templates/{workflow}/response_schema.json')

        response_schema_content = load_file(response_schema_path)
        response_schema = json.loads(response_schema_content)
        logger.info("Response schema loaded successfully")
        return response_schema
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from {response_schema_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading response schema for workflow {workflow} with step {step}: {e}")
        raise
