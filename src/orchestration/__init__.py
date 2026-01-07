"""Orchestration module for multi-assistant system.

This module handles:
- Dynamic agent class generation from configuration
- Agent registry and factory patterns
- Handoff coordination and context transfer
- Orchestration management
"""

from src.orchestration.factory import AgentFactory
from src.orchestration.handoff import HandoffCoordinator
from src.orchestration.manager import OrchestrationManager

__all__ = [
    "AgentFactory",
    "HandoffCoordinator",
    "OrchestrationManager",
]
