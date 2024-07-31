from src.pipeline.single_step import run as single_step_run
from src.utils.io import get_pdf_file_names
from src.config.logging import logger
from src.config.setup import config
import os


def run(directory: str) -> None:
    """
    Run the single-step data extraction process on all PDF files in the specified directory.

    Args:
        directory (str): The directory path where PDF files are located.
    """
    try:
        for file_name in get_pdf_file_names(directory):
            try:
                logger.info(f"Processing file: {file_name}")
                single_step_run(file_name)
            except Exception as e:
                logger.error(f"Error processing file {file_name}: {e}")
    except Exception as e:
        logger.error(f"Error retrieving PDF file names from directory {directory}: {e}")


if __name__ == '__main__':
    try:
        directory = os.path.join(config.DATA_DIR, 'docs/')
        run(directory)
    except Exception as e:
        logger.critical(f"Critical failure in main execution: {e}")
