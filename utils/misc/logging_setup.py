import logging
import sys
import os
import platform

# Define custom log levels
PROCESSING_LEVEL = 21  # Between INFO (20) and WARNING (30)
SUCCESS_LEVEL = 22     # Between INFO (20) and WARNING (30)

# Add custom level names
logging.addLevelName(PROCESSING_LEVEL, "PROCESSING")
logging.addLevelName(SUCCESS_LEVEL, "SUCCESS")


class ColoredFormatter(logging.Formatter):
    """Formatter to add colors to log levels"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Color support detection
        self.supports_color = (
            hasattr(sys.stdout, "isatty")
            and sys.stdout.isatty()
            and (
                platform.system() != "Windows"
                or "ANSICON" in os.environ
                or "WT_SESSION" in os.environ
                or os.environ.get("TERM", "").startswith("xterm")
            )
        )

        if self.supports_color:
            self.COLORS = {
                "DEBUG": "\033[36m",        # Cyan
                "INFO": "\033[37m",         # White
                "PROCESSING": "\033[34m",   # Blue
                "SUCCESS": "\033[32m",      # Green
                "WARNING": "\033[33m",      # Yellow
                "ERROR": "\033[31m",        # Red
                "CRITICAL": "\033[35m",     # Magenta
            }
            self.RESET = "\033[0m"
            self.BOLD = "\033[1m"
        else:
            self.COLORS = {}
            self.RESET = ""
            self.BOLD = ""

    def format(self, record):
        if self.supports_color and record.levelname in self.COLORS:
            # Check if this is a status message (has status symbols)
            message = record.getMessage()
            is_status_message = any(
                message.startswith(symbol) for symbol in ["✓", "✗", "⚠", "⧗"]
            )

            # Create a copy to avoid modifying original
            record_copy = logging.makeLogRecord(record.__dict__)

            if is_status_message:
                # For status messages: color both levelname AND message
                colored_levelname = (
                    f"{self.COLORS[record.levelname]}{self.BOLD}"
                    f"{record.levelname}{self.RESET}"
                )
                colored_message = (
                    f"{self.COLORS[record.levelname]}{message}{self.RESET}"
                )
                record_copy.levelname = colored_levelname
                record_copy.msg = colored_message
                record_copy.args = ()
            else:
                # For regular messages: just color levelname
                colored_levelname = (
                    f"{self.COLORS[record.levelname]}{self.BOLD}"
                    f"{record.levelname}{self.RESET}"
                )
                record_copy.levelname = colored_levelname

            return super().format(record_copy)
        else:
            return super().format(record)


def add_status_methods():
    """Add status methods to logger class"""

    def success(self, message, *args, **kwargs):
        if self.isEnabledFor(SUCCESS_LEVEL):
            self._log(SUCCESS_LEVEL, f"✓ {message}", args, **kwargs)

    def error_status(self, message, *args, **kwargs):
        if self.isEnabledFor(logging.ERROR):
            self._log(logging.ERROR, f"✗ {message}", args, **kwargs)

    def warning_status(self, message, *args, **kwargs):
        if self.isEnabledFor(logging.WARNING):
            self._log(logging.WARNING, f"⚠ {message}", args, **kwargs)

    def processing(self, message, *args, **kwargs):
        if self.isEnabledFor(PROCESSING_LEVEL):
            self._log(PROCESSING_LEVEL, f"⧗ {message}", args, **kwargs)

    # Add methods to logger class
    logging.Logger.success = success
    logging.Logger.error_status = error_status
    logging.Logger.warning_status = warning_status
    logging.Logger.processing = processing


def setup_logging():
    """Configure logging with colors and status methods"""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Clear existing handlers
    if logger.handlers:
        for handler in logger.handlers:
            logger.removeHandler(handler)

    # File handler (no colors) - overwrite log file each run
    file_handler = logging.FileHandler(
        "llm_prompting.log", mode="w", encoding="utf-8"
    )
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(file_handler)

    # Console handler (with colors)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(
        ColoredFormatter("%(asctime)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(console_handler)

    # Add custom status methods
    add_status_methods()

    return logger


# Initialize logger
logger = setup_logging()
