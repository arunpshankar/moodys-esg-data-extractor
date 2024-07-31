from src.utils.evaluate import compare_json_objects
from src.config.logging import logger
from src.config.setup import config
from src.utils.io import load_jsonl
from typing import Tuple
from typing import List 
from typing import Any 
import os


def compare_jsonl_files(expected_file_path: str, generated_file_path: str) -> Tuple[List[Tuple[Any, Any]], int]:
    """
    Compares two JSONL files and finds matching objects.

    Args:
        expected_file_path (str): Path to the expected JSONL file.
        generated_file_path (str): Path to the generated JSONL file.

    Returns:
        Tuple[List[Tuple[Any, Any]], int]: A list of matching object pairs and the count of objects in the generated file.
    """
    try:
        expected_json_list = load_jsonl(expected_file_path)
        generated_json_list = load_jsonl(generated_file_path)
    except Exception as e:
        logger.error(f"Error loading JSONL files: {e}")
        return [], 0

    matches = []
    for generated_obj in generated_json_list:
        for expected_obj in expected_json_list:
            try:
                is_code_matched, is_value_matched = compare_json_objects(expected_obj, generated_obj)
                if is_code_matched and is_value_matched:
                    matches.append((expected_obj, generated_obj))
            except Exception as e:
                logger.error(f"Error comparing JSON objects: {e}")

    return matches, len(generated_json_list)


if __name__ == '__main__':
    file_id = '100395060535523152'
    workflow_step = 'single_step'

    expected_file_path = os.path.join(config.DATA_DIR, f'validation/expected/{file_id}.jsonl')
    generated_file_path = os.path.join(config.DATA_DIR, f'validation/generated/{workflow_step}/{file_id}.jsonl')

    matches, total_generated_objects = compare_jsonl_files(expected_file_path, generated_file_path)

    if total_generated_objects > 0:
        match_percentage = len(matches) / total_generated_objects * 100
        logger.info(f"Matched {len(matches)} out of {total_generated_objects} objects. Match percentage: {match_percentage:.2f}%")
    else:
        logger.warning("No objects to compare in the generated file.")
