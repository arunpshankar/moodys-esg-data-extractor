from src.config.logging import logger 
from src.config.setup import config
from src.utils.io import load_file
from typing import Optional 
from typing import List 
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
            system_instruction = [load_file(os.path.join(config.DATA_DIR, f'templates/{workflow}/system_instruction/system_instructions_step_{step}.txt'))]
        else:
            logger.info(f"Loading single-step system instruction for workflow: {workflow}")
            system_instruction = [load_file(os.path.join(config.DATA_DIR, f'templates/{workflow}/system_instruction.txt'))]
        logger.info("System instruction loaded successfully")
        return system_instruction
    except Exception as e:
        logger.error(f"Error loading system instruction for workflow {workflow} with step {step}: {e}")
        raise


def load_user_instruction():
    pass


def load_response_schema():
    pass
