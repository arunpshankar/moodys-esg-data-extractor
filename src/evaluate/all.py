from src.evaluate.single import compare_jsonl_files
from src.config.logging import logger
from src.config.setup import config
from typing import TextIO
from typing import Tuple
from typing import List
import json
import os


def iterate_and_compare(dir1: str, dir2: str, workflow: str) -> None:
    """
    Compare JSONL files from two directories and log the results.

    Parameters:
    dir1 (str): The directory containing the generated JSONL files (extracted by LLM, Gemini).
    dir2 (str): The directory containing the expected JSONL files (ground truth by SME).
    workflow (str): The workflow name used to construct file paths.

    Returns:
    None
    """
    match_file_path = os.path.join(config.DATA_DIR, f'evaluation/{workflow}/matches.jsonl')
    accuracy_file_path = os.path.join(config.DATA_DIR, f'evaluation/{workflow}/coverage.txt')

    os.makedirs(os.path.dirname(match_file_path), exist_ok=True)
    os.makedirs(os.path.dirname(accuracy_file_path), exist_ok=True)

    try:
        with open(match_file_path, 'w') as match_file, open(accuracy_file_path, 'w') as accuracy_file:
            for filename in os.listdir(dir1):
                if filename.endswith(".jsonl"):
                    file1_path = os.path.join(dir1, filename)
                    file2_path = os.path.join(dir2, filename)

                    if os.path.exists(file2_path):
                        try:
                            matches, total_expected = compare_jsonl_files(file1_path, file2_path)
                            log_matches(match_file, filename, matches)
                            log_accuracy(accuracy_file, filename, matches, total_expected)
                        except Exception as e:
                            logger.error(f"Error comparing files {file1_path} and {file2_path}: {e}")
                    else:
                        logger.warning(f"File {filename} not found in {dir2}")
    except Exception as e:
        logger.error(f"Error opening output files: {e}")


def log_matches(match_file: TextIO, filename: str, matches: List[Tuple[dict, dict]]) -> None:
    """
    Log matches to the match file.

    Parameters:
    match_file (os.TextIO): The file object to write matches to.
    filename (str): The filename being processed.
    matches (List[Tuple[dict, dict]]): The list of matched generated and expected JSON objects.

    Returns:
    None
    """
    for generated, expected in matches:
        match_file.write(json.dumps({'filename': filename, 'generated': generated, 'expected': expected}) + '\n')


def log_accuracy(accuracy_file: TextIO, filename: str, matches: List[Tuple[dict, dict]], total_expected: int) -> None:
    """
    Log accuracy to the accuracy file.

    Parameters:
    accuracy_file (os.TextIO): The file object to write accuracy to.
    filename (str): The filename being processed.
    matches (List[Tuple[dict, dict]]): The list of matched generated and expected JSON objects.
    total_expected (int): The total number of expected matches.

    Returns:
    None
    """
    accuracy = len(matches) / total_expected * 100 if total_expected else 0
    accuracy_file.write(f"{filename}: {accuracy:.2f}%\n")


if __name__ == "__main__":
    workflow = 'multi_step'
    dir1 = os.path.join(config.DATA_DIR, f'validation/generated/{workflow}')
    dir2 = os.path.join(config.DATA_DIR, f'validation/expected')
    
    iterate_and_compare(dir1, dir2, workflow)
