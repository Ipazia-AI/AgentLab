"""
RLM (Recursive Language Model) Agent for AgentLab.

This module implements an agent that uses the RLM paradigm: instead of stuffing
the full DOM/AXTree into the prompt, it externalizes the observation to a REPL
environment where the model can programmatically explore it.

Based on the RLM paper: https://arxiv.org/abs/2512.24601
"""

from .rlm_agent import RLMGenericAgent, RLMGenericAgentArgs
from .rlm_prompt_flags import RLMPromptFlags
from .rlm_repl import REPLExecutor, REPLError
from .rlm_parser import is_final, parse_response, extract_final, extract_final_var

__all__ = [
    "RLMGenericAgent",
    "RLMGenericAgentArgs",
    "RLMPromptFlags",
    "REPLExecutor",
    "REPLError",
    "is_final",
    "parse_response",
    "extract_final",
    "extract_final_var",
]
