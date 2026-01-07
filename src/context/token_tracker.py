"""Token tracking using actual LLM metrics.

This module tracks real token usage from LLMMetrics.prompt_tokens,
not estimates. This is the actual token count from the model.

Inspired by Claude Code's auto-compact approach which monitors
context usage and compacts at strategic thresholds.
"""

import os
from dataclasses import dataclass, field
from typing import Any

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Default context limits for common models
MODEL_CONTEXT_LIMITS: dict[str, int] = {
    "google/gemini-2.0-flash-exp:free": 1_048_576,
    "google/gemini-2.0-flash-lite-001": 1_048_576,
    "google/gemini-2.5-flash-preview-05-20": 1_048_576,
    "mistralai/mistral-small-3.1-24b-instruct:free": 32_000,
    "deepseek/deepseek-chat": 65_536,
    "openai/gpt-4o-mini": 128_000,
    "openai/gpt-4o": 128_000,
    "anthropic/claude-sonnet-4": 200_000,
    "default": 32_000,
}


def get_compaction_threshold() -> float:
    """Get compaction threshold from environment variable.

    The threshold determines when context compaction is triggered,
    expressed as a percentage of the model's context window.

    Returns:
        Threshold as decimal (0.0-1.0). Default is 0.40 (40%).
    """
    threshold_str = os.environ.get("CONTEXT_COMPACTION_THRESHOLD", "0.40")
    try:
        threshold = float(threshold_str)
        # Clamp to valid range
        return max(0.1, min(0.9, threshold))
    except ValueError:
        logger.warning(
            f"Invalid CONTEXT_COMPACTION_THRESHOLD: {threshold_str}, using 0.40"
        )
        return 0.40


def get_compact_target() -> float:
    """Get target percentage to compact DOWN TO.

    After compaction is triggered, this determines what percentage
    of the context window the compacted result should target.

    Returns:
        Target as decimal (0.0-1.0). Default is 0.20 (20%).
    """
    target_str = os.environ.get("CONTEXT_COMPACT_TARGET", "0.20")
    try:
        target = float(target_str)
        # Clamp to valid range (5% to 50%)
        return max(0.05, min(0.50, target))
    except ValueError:
        logger.warning(
            f"Invalid CONTEXT_COMPACT_TARGET: {target_str}, using 0.20"
        )
        return 0.20


def get_context_limit(model_id: str | None = None) -> int:
    """Get context window limit for model.

    Args:
        model_id: OpenRouter model ID (e.g., 'google/gemini-2.0-flash-exp:free')

    Returns:
        Context window size in tokens.
    """
    if model_id and model_id in MODEL_CONTEXT_LIMITS:
        return MODEL_CONTEXT_LIMITS[model_id]

    # Try partial match (without :free suffix)
    if model_id:
        base_model = model_id.split(":")[0]
        for key, value in MODEL_CONTEXT_LIMITS.items():
            if key.startswith(base_model):
                return value

    return MODEL_CONTEXT_LIMITS["default"]


@dataclass
class TokenTracker:
    """Tracks actual token usage from LLM responses.

    Uses LLMMetrics.prompt_tokens to know the real context size,
    which is the most accurate way to track token usage.

    Attributes:
        model_id: The OpenRouter model ID being used.
        last_prompt_tokens: Token count from the most recent LLM call.
        last_completion_tokens: Completion tokens from the most recent call.
        total_prompt_tokens: Cumulative prompt tokens for the session.
        total_completion_tokens: Cumulative completion tokens for the session.
        request_count: Number of LLM requests made in this session.
    """

    model_id: str | None = None
    last_prompt_tokens: int = 0
    last_completion_tokens: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    request_count: int = 0
    _context_limit: int = field(init=False)
    _threshold: float = field(init=False)
    _compact_target: float = field(init=False)

    def __post_init__(self) -> None:
        """Initialize computed fields after dataclass init."""
        self._context_limit = get_context_limit(self.model_id)
        self._threshold = get_compaction_threshold()
        self._compact_target = get_compact_target()
        logger.config(
            f"TokenTracker initialized: model={self.model_id}, "
            f"limit={self._context_limit:,}, threshold={self._threshold:.0%}, "
            f"compact_target={self._compact_target:.0%}"
        )

    def update_from_metrics(self, metrics: Any) -> None:
        """Update token counts from LLMMetrics.

        This should be called from the LLM metrics_collected event handler.

        Args:
            metrics: LLMMetrics object with prompt_tokens, completion_tokens
        """
        self.last_prompt_tokens = getattr(metrics, "prompt_tokens", 0)
        self.last_completion_tokens = getattr(metrics, "completion_tokens", 0)
        self.total_prompt_tokens += self.last_prompt_tokens
        self.total_completion_tokens += self.last_completion_tokens
        self.request_count += 1

        logger.debug(
            f"Token update: prompt={self.last_prompt_tokens:,}, "
            f"completion={self.last_completion_tokens:,}, "
            f"usage={self.usage_percent:.1f}%"
        )

    @property
    def context_limit(self) -> int:
        """Get model's context window limit."""
        return self._context_limit

    @property
    def threshold(self) -> float:
        """Get compaction threshold (0.0-1.0)."""
        return self._threshold

    @property
    def threshold_tokens(self) -> int:
        """Get token count that triggers compaction."""
        return int(self._context_limit * self._threshold)

    @property
    def compact_target(self) -> float:
        """Get target percentage for compacted context (0.0-1.0)."""
        return self._compact_target

    @property
    def target_tokens(self) -> int:
        """Get target token count after compaction."""
        return int(self._context_limit * self._compact_target)

    @property
    def usage_percent(self) -> float:
        """Get current context usage as percentage."""
        if self._context_limit == 0:
            return 0.0
        return (self.last_prompt_tokens / self._context_limit) * 100

    @property
    def should_compact(self) -> bool:
        """Check if context should be compacted based on last prompt_tokens.

        Returns:
            True if the last prompt exceeded the compaction threshold.
        """
        should = self.last_prompt_tokens > self.threshold_tokens
        if should:
            logger.config(
                f"Compaction needed: {self.last_prompt_tokens:,} tokens "
                f"({self.usage_percent:.1f}%) > {self._threshold:.0%} threshold"
            )
        return should

    def get_stats(self) -> dict[str, Any]:
        """Get token usage statistics.

        Returns:
            Dictionary with all token tracking statistics.
        """
        return {
            "model_id": self.model_id,
            "context_limit": self._context_limit,
            "threshold": self._threshold,
            "threshold_tokens": self.threshold_tokens,
            "compact_target": self._compact_target,
            "target_tokens": self.target_tokens,
            "last_prompt_tokens": self.last_prompt_tokens,
            "last_completion_tokens": self.last_completion_tokens,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "usage_percent": self.usage_percent,
            "should_compact": self.should_compact,
            "request_count": self.request_count,
        }


# Global token tracker instance (per session)
_token_tracker: TokenTracker | None = None


def get_token_tracker(model_id: str | None = None) -> TokenTracker:
    """Get or create global token tracker.

    Args:
        model_id: OpenRouter model ID (only used on first call to create tracker).

    Returns:
        The global TokenTracker instance.
    """
    global _token_tracker
    if _token_tracker is None:
        _token_tracker = TokenTracker(model_id=model_id)
    return _token_tracker


def reset_token_tracker() -> None:
    """Reset token tracker for a new session.

    Call this at the start of each new session to ensure
    fresh token tracking.
    """
    global _token_tracker
    _token_tracker = None
    logger.debug("Token tracker reset")
