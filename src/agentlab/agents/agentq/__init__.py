"""
AgentQ: MCTS-based web agent with tournament critic ranking.

Implements the AgentQ paper approach with:
- Tournament-style action ranking (iterative best selection)
- Fast reward for efficient rollout estimation
- In-context DPO learning from preference pairs
"""

from agentlab.agents.agentq.agentq import AgentQ, AgentQArgs
from agentlab.agents.agentq.mcts import MCTS, MCTSNode
from agentlab.agents.agentq.evaluators import AbsoluteCritic, TournamentCritic, BaseCritic
from agentlab.agents.agentq.selectors import MaxVisitSelector, AheadKSelector, ActionSelector

__all__ = [
    "AgentQ",
    "AgentQArgs",
    "MCTS",
    "MCTSNode",
    "AbsoluteCritic",
    "TournamentCritic",
    "BaseCritic",
    "MaxVisitSelector",
    "AheadKSelector",
    "ActionSelector",
]
