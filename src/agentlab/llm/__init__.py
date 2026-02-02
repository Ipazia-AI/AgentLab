"""
AgentLab LLM module.

This module provides chat model implementations and utilities for LLM interactions.
"""

from .base_api import AbstractChatModel, BaseModelArgs
from .rlm_chat_model import RLMChatModel, RLMModelArgs, RLMError, MaxIterationsError, MaxDepthError
from .rlm_repl import REPLExecutor, REPLError
from .rlm_parser import is_final, parse_response, extract_final, extract_final_var
from .rlm_prompt_parser import (
    ParsedPrompt,
    parse_agent_prompt,
    get_context_info,
    build_context_dict,
)

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
    "parse_response",
    "extract_final",
    "extract_final_var",
    # Prompt parser
    "ParsedPrompt",
    "parse_agent_prompt",
    "get_context_info",
    "build_context_dict",
]
