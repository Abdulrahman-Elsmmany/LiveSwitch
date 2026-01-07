"""
Colored Logger for Multi-Assistant LiveKit Worker.

This module provides a specialized logging system with colored output
for different event types, making console output easier to scan.
"""

import logging
import os
from typing import Any

from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme

# Color scheme for different event types
COLORS = {
    "ERROR": "bright_red",
    "WARNING": "bright_yellow",
    "INFO": "bright_green",
    "DEBUG": "bright_blue",
    "HANDOFF": "bright_magenta",  # Agent-to-agent transfers
    "AGENT": "bright_cyan",  # Agent entering/exiting
    "SESSION": "bright_green",  # Session lifecycle
    "TOOL": "bright_yellow",  # Tool execution
    "CONFIG": "bright_blue",  # Configuration loading
    "MEMORY": "bright_white",  # Session memory injection
    "TRANSCRIPT": "green",  # Real-time conversation transcript
}

# Get log level from environment variable
ENV_LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

_LEVEL_MAP: dict[str, int] = {
    "CRITICAL": logging.CRITICAL,
    "ERROR": logging.ERROR,
    "WARNING": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
    "NOTSET": logging.NOTSET,
}


class ColoredLogger(logging.Logger):
    """Logger with colored output for different event types."""

    # Custom log levels
    HANDOFF_LEVEL = 25  # Between INFO (20) and WARNING (30)
    AGENT_LEVEL = 24
    SESSION_LEVEL = 23
    TOOL_LEVEL = 22
    CONFIG_LEVEL = 21
    MEMORY_LEVEL = 26  # Session memory injection events
    TRANSCRIPT_LEVEL = 27  # Real-time conversation transcript (high visibility)

    def __init__(self, name: str, level: int | None = None):
        if level is None:
            level = _LEVEL_MAP.get(ENV_LOG_LEVEL, logging.INFO)

        super().__init__(name, level)

        # Create theme for rich console
        log_style_theme = {
            f"logging.level.{level_name.lower()}": style
            for level_name, style in COLORS.items()
        }
        custom_theme = Theme(log_style_theme)
        self.console = Console(theme=custom_theme)

        # Clear existing handlers
        for handler in self.handlers[:]:
            self.removeHandler(handler)

        # Add handler based on environment
        if self._is_testing():
            self.addHandler(logging.StreamHandler())
        else:
            self.addHandler(self._create_rich_handler())

        self.propagate = False

    def _is_testing(self) -> bool:
        """Check if running under pytest."""
        return "PYTEST_CURRENT_TEST" in os.environ

    def _create_rich_handler(self) -> RichHandler:
        """Create RichHandler with optimized settings."""
        return RichHandler(
            console=self.console,
            show_time=True,
            show_level=True,
            show_path=False,
            enable_link_path=False,
            rich_tracebacks=True,
            tracebacks_show_locals=False,
            markup=True,
        )

    def handoff(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Log handoff events (purple)."""
        self.log(self.HANDOFF_LEVEL, f"[bright_magenta]{message}[/]", *args, **kwargs)

    def agent(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Log agent events (cyan)."""
        self.log(self.AGENT_LEVEL, f"[bright_cyan]{message}[/]", *args, **kwargs)

    def session(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Log session events (green)."""
        self.log(self.SESSION_LEVEL, f"[bright_green]{message}[/]", *args, **kwargs)

    def tool(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Log tool events (yellow)."""
        self.log(self.TOOL_LEVEL, f"[bright_yellow]{message}[/]", *args, **kwargs)

    def config(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Log config events (blue)."""
        self.log(self.CONFIG_LEVEL, f"[bright_blue]{message}[/]", *args, **kwargs)

    def memory(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Log memory injection events (white)."""
        self.log(self.MEMORY_LEVEL, f"[bright_white]{message}[/]", *args, **kwargs)

    def transcript(self, message: str, *args: Any, **kwargs: Any) -> None:
        """Log real-time transcript events (green, high visibility)."""
        self.log(self.TRANSCRIPT_LEVEL, f"[green]{message}[/]", *args, **kwargs)


# Register custom log levels
logging.addLevelName(ColoredLogger.HANDOFF_LEVEL, "HANDOFF")
logging.addLevelName(ColoredLogger.AGENT_LEVEL, "AGENT")
logging.addLevelName(ColoredLogger.SESSION_LEVEL, "SESSION")
logging.addLevelName(ColoredLogger.TOOL_LEVEL, "TOOL")
logging.addLevelName(ColoredLogger.CONFIG_LEVEL, "CONFIG")
logging.addLevelName(ColoredLogger.MEMORY_LEVEL, "MEMORY")
logging.addLevelName(ColoredLogger.TRANSCRIPT_LEVEL, "TRANSCRIPT")

# Set as default logger class
logging.setLoggerClass(ColoredLogger)

# Cache for logger instances
_loggers: dict[str, ColoredLogger] = {}


def get_logger(name: str, level: int | None = None) -> ColoredLogger:
    """
    Get a colored logger instance.

    Args:
        name: Logger name (typically __name__)
        level: Optional log level override

    Returns:
        Configured ColoredLogger instance
    """
    if name not in _loggers:
        logger = ColoredLogger(name, level)
        _loggers[name] = logger
    return _loggers[name]


if __name__ == "__main__":
    # Test the logger
    logger = get_logger(__name__)

    logger.debug("Debug message")
    logger.info("Info message")
    logger.config("Loading configuration from config/medical_triage.json")
    logger.session("Session abc123 started with entry agent: receptionist")
    logger.agent("Agent 'receptionist' entering conversation")
    logger.tool("Executing tool: verify_patient")
    logger.handoff("Handoff: receptionist -> nurse_triage")
    logger.transcript("User: Hi, I need to schedule an appointment")
    logger.transcript("receptionist: Hello! I'd be happy to help you schedule...")
    logger.warning("Warning message")
    logger.error("Error message")
