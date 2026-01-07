"""Semantic validator for multi-assistant configuration.

This module performs business logic validation that goes beyond
schema validation. It ensures referential integrity and catches
configuration errors that would cause runtime failures.
"""

from dataclasses import dataclass, field

from src.config.schemas import MultiAssistantConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ValidationWarning:
    """A non-fatal validation warning."""

    code: str
    message: str
    location: str | None = None


@dataclass
class ValidationError:
    """A fatal validation error."""

    code: str
    message: str
    location: str | None = None


@dataclass
class ValidationResult:
    """Result of semantic validation.

    Attributes:
        is_valid: Whether the configuration is valid.
        errors: List of fatal validation errors.
        warnings: List of non-fatal warnings.
    """

    is_valid: bool = True
    errors: list[ValidationError] = field(default_factory=list)
    warnings: list[ValidationWarning] = field(default_factory=list)

    def add_error(self, code: str, message: str, location: str | None = None) -> None:
        """Add a validation error.

        Args:
            code: Error code for programmatic handling.
            message: Human-readable error message.
            location: Optional location in config (e.g., "assistants[0].handoff_targets[1]").
        """
        self.errors.append(ValidationError(code=code, message=message, location=location))
        self.is_valid = False

    def add_warning(
        self, code: str, message: str, location: str | None = None
    ) -> None:
        """Add a validation warning.

        Args:
            code: Warning code for programmatic handling.
            message: Human-readable warning message.
            location: Optional location in config.
        """
        self.warnings.append(
            ValidationWarning(code=code, message=message, location=location)
        )


class SemanticValidationError(Exception):
    """Raised when semantic validation fails."""

    def __init__(self, result: ValidationResult) -> None:
        """Initialize with validation result.

        Args:
            result: The validation result containing errors.
        """
        error_messages = [
            f"  - [{e.code}] {e.message}" + (f" at {e.location}" if e.location else "")
            for e in result.errors
        ]
        super().__init__(
            f"Semantic validation failed with {len(result.errors)} error(s):\n"
            + "\n".join(error_messages)
        )
        self.result = result


def validate_config(config: MultiAssistantConfig) -> ValidationResult:
    """Perform semantic validation on a configuration.

    This validates business logic and referential integrity:
    - Entry point exists in assistants list
    - All handoff targets reference existing assistants
    - Fallback assistant exists if specified
    - No orphaned assistants (warning only)
    - Handoff rules reference valid assistants

    Args:
        config: The configuration to validate.

    Returns:
        ValidationResult with any errors and warnings.

    Raises:
        SemanticValidationError: If validation fails (when raise_on_error=True).

    Example:
        >>> result = validate_config(config)
        >>> if not result.is_valid:
        ...     for error in result.errors:
        ...         print(f"Error: {error.message}")
    """
    result = ValidationResult()
    all_assistant_ids = config.get_all_assistant_ids()

    # Validate entry point exists
    _validate_entry_point(config, all_assistant_ids, result)

    # Validate fallback assistant exists
    _validate_fallback_assistant(config, all_assistant_ids, result)

    # Validate all handoff targets reference existing assistants
    _validate_handoff_targets(config, all_assistant_ids, result)

    # Validate handoff rules (for rule-based handoffs)
    _validate_handoff_rules(config, all_assistant_ids, result)

    # Check for orphaned assistants (warning only)
    _check_orphaned_assistants(config, result)

    # Log validation result
    if result.is_valid:
        if result.warnings:
            logger.warning(
                f"Configuration validation passed with {len(result.warnings)} warning(s)"
            )
        else:
            logger.config("Configuration validation passed")
    else:
        logger.error(
            f"Configuration validation failed with {len(result.errors)} error(s)"
        )

    return result


def validate_config_strict(config: MultiAssistantConfig) -> MultiAssistantConfig:
    """Validate configuration and raise on errors.

    Convenience function that performs validation and raises
    SemanticValidationError if any errors are found.

    Args:
        config: The configuration to validate.

    Returns:
        The same configuration if valid.

    Raises:
        SemanticValidationError: If validation fails.
    """
    result = validate_config(config)
    if not result.is_valid:
        raise SemanticValidationError(result)
    return config


def _validate_entry_point(
    config: MultiAssistantConfig,
    all_ids: set[str],
    result: ValidationResult,
) -> None:
    """Validate that entry_point references an existing assistant."""
    entry_point = config.orchestration.entry_point
    if entry_point not in all_ids:
        result.add_error(
            code="INVALID_ENTRY_POINT",
            message=f"Entry point '{entry_point}' does not reference an existing assistant. "
            f"Available assistants: {sorted(all_ids)}",
            location="orchestration.entry_point",
        )


def _validate_fallback_assistant(
    config: MultiAssistantConfig,
    all_ids: set[str],
    result: ValidationResult,
) -> None:
    """Validate that fallback_assistant references an existing assistant if set."""
    fallback = config.orchestration.fallback_assistant
    if fallback is not None and fallback not in all_ids:
        result.add_error(
            code="INVALID_FALLBACK_ASSISTANT",
            message=f"Fallback assistant '{fallback}' does not reference an existing assistant. "
            f"Available assistants: {sorted(all_ids)}",
            location="orchestration.fallback_assistant",
        )


def _validate_handoff_targets(
    config: MultiAssistantConfig,
    all_ids: set[str],
    result: ValidationResult,
) -> None:
    """Validate that all handoff targets reference existing assistants."""
    for i, assistant in enumerate(config.assistants):
        for j, target in enumerate(assistant.handoff_targets):
            if target.assistant_id not in all_ids:
                result.add_error(
                    code="INVALID_HANDOFF_TARGET",
                    message=f"Handoff target '{target.assistant_id}' from assistant "
                    f"'{assistant.id}' does not reference an existing assistant. "
                    f"Available assistants: {sorted(all_ids)}",
                    location=f"assistants[{i}].handoff_targets[{j}]",
                )


def _validate_handoff_rules(
    config: MultiAssistantConfig,
    all_ids: set[str],
    result: ValidationResult,
) -> None:
    """Validate that handoff rules reference existing assistants."""
    for i, rule in enumerate(config.orchestration.handoff_rules):
        if rule.source not in all_ids:
            result.add_error(
                code="INVALID_RULE_SOURCE",
                message=f"Handoff rule source '{rule.source}' does not reference "
                f"an existing assistant. Available assistants: {sorted(all_ids)}",
                location=f"orchestration.handoff_rules[{i}].source",
            )
        if rule.target not in all_ids:
            result.add_error(
                code="INVALID_RULE_TARGET",
                message=f"Handoff rule target '{rule.target}' does not reference "
                f"an existing assistant. Available assistants: {sorted(all_ids)}",
                location=f"orchestration.handoff_rules[{i}].target",
            )


def _check_orphaned_assistants(
    config: MultiAssistantConfig,
    result: ValidationResult,
) -> None:
    """Check for assistants that cannot be reached via handoffs.

    An orphaned assistant is one that:
    - Is not the entry point
    - Is not referenced by any handoff target
    - Is not the fallback assistant

    This is a warning, not an error, as orphaned assistants might be
    intentionally included for future use or testing.
    """
    entry_point = config.orchestration.entry_point
    fallback = config.orchestration.fallback_assistant

    # Collect all referenced assistant IDs
    referenced_ids: set[str] = {entry_point}
    if fallback:
        referenced_ids.add(fallback)

    for assistant in config.assistants:
        for target in assistant.handoff_targets:
            referenced_ids.add(target.assistant_id)

    # Find orphaned assistants
    all_ids = config.get_all_assistant_ids()
    orphaned = all_ids - referenced_ids

    for orphan_id in orphaned:
        result.add_warning(
            code="ORPHANED_ASSISTANT",
            message=f"Assistant '{orphan_id}' is not reachable from entry point "
            f"and has no inbound handoff targets",
            location=f"assistants[id={orphan_id}]",
        )
