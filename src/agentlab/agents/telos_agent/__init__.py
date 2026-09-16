"""Telos host agent for AgentLab / BrowserGym.

Telos is a planner: this package is the host that turns WorkArena observations
into Telos inputs and grounds Telos NL actions into BrowserGym action strings.
"""

from .telos_agent import TelosAgentArgs

__all__ = ["TelosAgentArgs"]
