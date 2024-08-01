from src.pipeline.single_step import run as single_step_run
from src.utils.io import get_pdf_file_names
from src.config.logging import logger
from src.config.setup import config
import asyncio
import os


async def process_file(file_name: str) -> None:
    """
    Process a single PDF file asynchronously.

    Args:
        file_name (str): The name of the PDF file to process.
    """
    try:
        logger.info(f"Processing file: {file_name}")
        # Wrap the synchronous function in a coroutine
        await asyncio.to_thread(single_step_run, file_name)
        logger.info(f"Finished processing file: {file_name}")
    except Exception as e:
        logger.error(f"Error processing file {file_name}: {e}")


async def run(directory: str, concurrency: int = 5) -> None:
    """
    Run the single-step data extraction process on PDF files in the specified directory concurrently.

    Args:
        directory (str): The directory path where PDF files are located.
        concurrency (int): The number of files to process concurrently. Defaults to 5.
    """
    try:
        # Convert generator to list
        pdf_files = list(get_pdf_file_names(directory))
        logger.info(f"Found {len(pdf_files)} PDF files in the directory.")

        if not pdf_files:
            logger.warning("No PDF files found in the specified directory.")
            return

        # Create a semaphore to limit concurrency
        semaphore = asyncio.Semaphore(concurrency)

        async def process_with_semaphore(file_name: str) -> None:
            async with semaphore:
                await process_file(file_name)

        # Create tasks for each file
        tasks = [asyncio.create_task(process_with_semaphore(file)) for file in pdf_files]

        # Wait for all tasks to complete
        await asyncio.gather(*tasks)

    except Exception as e:
        logger.error(f"Error retrieving or processing PDF files from directory {directory}: {e}")


async def main() -> None:
    """
    Main entry point for the asynchronous PDF processing script.
    """
    try:
        directory = os.path.join(config.DATA_DIR, 'docs/')
        logger.info(f"Starting parallel PDF processing in directory: {directory}")
        await run(directory)
        logger.info("PDF processing completed successfully.")
    except Exception as e:
        logger.critical(f"Critical failure in main execution: {e}")


if __name__ == '__main__':
    asyncio.run(main())