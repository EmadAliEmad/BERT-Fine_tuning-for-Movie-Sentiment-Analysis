# %%writefile logger.py
import logging # Standard Python logging module (used by RichHandler internally).
import sys     # System-specific parameters and functions (for stderr).
from loguru import logger # Simplified and powerful logging library.
from rich.console import Console # Rich console for beautiful terminal output.
from rich.logging import RichHandler # Loguru handler to integrate with Rich.

# Import project_config to access log directory path defined in config.py.
from config import project_config 

def setup_logging(level: str = "INFO") -> None:
    """
    Sets up a professional logging configuration using Loguru and Rich.
    Logs messages to both console (formatted by Rich) and a file.
    
    Args:
        level (str): Minimum logging level to display and save (e.g., "INFO", "DEBUG", "ERROR").
                     Messages below this level will be ignored.
    """
    
    # Remove any default Loguru handlers to take full control over logging destinations.
    logger.remove()
    
    # Add a RichHandler to direct log messages to the console (stderr).
    # `Console(stderr=True)` directs output to the standard error stream, which is common in Jupyter environments.
    # `rich_tracebacks=True` enhances traceback formatting for better debugging.
    logger.add(
        RichHandler(console=Console(stderr=True), rich_tracebacks=True), 
        # Define the format for log messages: timestamp, level, source (file:function:line), and message.
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level=level # Set the minimum logging level for console output.
    )
    
    # Add a file handler to persist log messages to a file.
    logger.add(
        project_config.logs_dir / "app.log", # Construct the full log file path using ProjectConfig.
        rotation="10 MB", # Configure log file rotation: a new file is created when the current one reaches 10 MB.
        retention="7 days", # Configure log file retention: log files older than 7 days are automatically deleted.
        level=level, # Set the minimum logging level for file output.
        # Define the format for log messages saved to the file.
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}" 
    )
    
    # Log a confirmation message that the logging setup has been completed successfully.
    logger.info("Logging setup completed")