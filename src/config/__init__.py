"""Configuration module for multi-assistant system.

This module handles:
- Pydantic schema definitions for configuration validation
- JSON configuration loading and parsing
- Semantic validation of configuration integrity
"""

from src.config.loader import load_config
from src.config.schemas import (
    AssistantConfig,
    GlobalSettings,
    HandoffTarget,
    MultiAssistantConfig,
    OrchestrationConfig,
)
from src.config.validator import validate_config

__all__ = [
    "load_config",
    "validate_config",
    "MultiAssistantConfig",
    "AssistantConfig",
    "HandoffTarget",
    "OrchestrationConfig",
    "GlobalSettings",
]
