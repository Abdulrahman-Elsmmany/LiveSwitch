"""Configuration loader for multi-assistant system.

This module handles loading and parsing JSON configuration files
with proper error handling and validation.
"""

import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from src.config.schemas import MultiAssistantConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ConfigurationError(Exception):
    """Base exception for configuration errors."""

    pass


class ConfigFileNotFoundError(ConfigurationError):
    """Raised when configuration file is not found."""

    pass


class ConfigParseError(ConfigurationError):
    """Raised when configuration file cannot be parsed as JSON."""

    pass


class ConfigValidationError(ConfigurationError):
    """Raised when configuration fails schema validation."""

    def __init__(self, message: str, errors: list[dict[str, Any]]) -> None:
        """Initialize with validation errors.

        Args:
            message: Error message.
            errors: List of validation error details.
        """
        super().__init__(message)
        self.errors = errors


def load_config(config_path: str | Path) -> MultiAssistantConfig:
    """Load and validate a multi-assistant configuration file.

    This function performs three stages of validation:
    1. File existence check
    2. JSON parsing
    3. Pydantic schema validation

    Args:
        config_path: Path to the JSON configuration file.

    Returns:
        Validated MultiAssistantConfig object.

    Raises:
        ConfigFileNotFoundError: If the config file doesn't exist.
        ConfigParseError: If the file is not valid JSON.
        ConfigValidationError: If the configuration fails schema validation.

    Example:
        >>> config = load_config("config/medical_triage.json")
        >>> print(config.metadata.name)
        Medical Office Triage System
    """
    path = Path(config_path)

    # Stage 1: File existence check
    if not path.exists():
        raise ConfigFileNotFoundError(
            f"Configuration file not found: {path.absolute()}"
        )

    if not path.is_file():
        raise ConfigFileNotFoundError(
            f"Configuration path is not a file: {path.absolute()}"
        )

    # Stage 2: JSON parsing
    try:
        with open(path, encoding="utf-8") as f:
            raw_config = json.load(f)
    except json.JSONDecodeError as e:
        raise ConfigParseError(
            f"Failed to parse configuration file as JSON: {e.msg} "
            f"at line {e.lineno}, column {e.colno}"
        ) from e
    except UnicodeDecodeError as e:
        raise ConfigParseError(
            f"Configuration file encoding error: {e}. Ensure file is UTF-8 encoded."
        ) from e

    # Stage 3: Pydantic schema validation
    try:
        config = MultiAssistantConfig.model_validate(raw_config)
    except ValidationError as e:
        errors = e.errors()
        error_messages = []
        for err in errors:
            loc = " -> ".join(str(loc) for loc in err["loc"])
            error_messages.append(f"  - {loc}: {err['msg']}")

        raise ConfigValidationError(
            f"Configuration validation failed with {len(errors)} error(s):\n"
            + "\n".join(error_messages),
            errors=[
                {"location": err["loc"], "message": err["msg"], "type": err["type"]}
                for err in errors
            ],
        ) from e

    logger.config(
        f"Successfully loaded configuration: {config.metadata.name} "
        f"(v{config.metadata.version}) with {len(config.assistants)} assistants"
    )

    return config


def load_config_from_dict(config_dict: dict[str, Any]) -> MultiAssistantConfig:
    """Load and validate configuration from a dictionary.

    Useful for testing or programmatic configuration.

    Args:
        config_dict: Configuration dictionary.

    Returns:
        Validated MultiAssistantConfig object.

    Raises:
        ConfigValidationError: If the configuration fails schema validation.
    """
    try:
        config = MultiAssistantConfig.model_validate(config_dict)
    except ValidationError as e:
        errors = e.errors()
        error_messages = []
        for err in errors:
            loc = " -> ".join(str(loc) for loc in err["loc"])
            error_messages.append(f"  - {loc}: {err['msg']}")

        raise ConfigValidationError(
            f"Configuration validation failed with {len(errors)} error(s):\n"
            + "\n".join(error_messages),
            errors=[
                {"location": err["loc"], "message": err["msg"], "type": err["type"]}
                for err in errors
            ],
        ) from e

    return config
