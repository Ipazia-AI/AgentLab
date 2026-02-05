"""
AgentLab LLM module.

This module provides chat model implementations and utilities for LLM interactions.
"""

from .base_api import AbstractChatModel, BaseModelArgs
from .rlm_chat_model import (
    MaxDepthError,
    MaxIterationsError,
    RLMChatModel,
    RLMError,
    RLMModelArgs,
)
from .rlm_parser import check_for_final_answer, find_final_answer, is_final
from .rlm_repl import REPLError, REPLExecutor

__all__ = [
    # Base classes
    "AbstractChatModel",
    "BaseModelArgs",
    # RLM decorator
    "RLMChatModel",
    "RLMModelArgs",
    # RLM errors
    "RLMError",
    "MaxIterationsError",
    "MaxDepthError",
    # RLM utilities
    "REPLExecutor",
    "REPLError",
    "is_final",
    "find_final_answer",
    "check_for_final_answer",
]
