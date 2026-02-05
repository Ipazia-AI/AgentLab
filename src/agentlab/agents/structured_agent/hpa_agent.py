from dataclasses import dataclass
from typing import Any

from browsergym.core.action.highlevel import HighLevelActionSet

from agentlab.agents.agent_args import AgentArgs
from agentlab.agents.dynamic_prompting import ObsFlags, make_obs_preprocessor
from agentlab.agents.generic_agent.generic_agent import GenericAgent, GenericAgentArgs
from agentlab.llm.base_api import BaseModelArgs
from agentlab.llm.tracking import cost_tracker_decorator

from .andor_tree import Node, NodeType
from .hpa import HPA


@dataclass
class HPAAgentArgs(GenericAgentArgs):
    chat_model_args: BaseModelArgs | None = None
    budget: int = 1000
    max_revision_count: int = 3
    action_subsets: tuple[str, ...] = ("workarena",)
    multiaction: bool = False

    def __post_init__(self):
        if self.chat_model_args is not None:
            self.agent_name = f"HPAAgent-{self.chat_model_args.model_name}".replace("/", "_")
        else:
            self.agent_name = "HPAAgent"

    def make_agent(self) -> GenericAgent:
        return HPAAgent(
            chat_model_args=self.chat_model_args,
            budget=self.budget,
            max_revision_count=self.max_revision_count,
            action_subsets=self.action_subsets,
            multiaction=self.multiaction,
        )

    def prepare(self):
        if self.chat_model_args is not None:
            return self.chat_model_args.prepare_server()
        return None

    def close(self):
        if self.chat_model_args is not None:
            return self.chat_model_args.close_server()
        return None


class HPAAgent(GenericAgent):
    def __init__(
        self,
        chat_model_args: BaseModelArgs | None,
        budget: int,
        max_revision_count: int,
        action_subsets: tuple[str, ...],
        multiaction: bool,
    ):
        self.chat_llm = chat_model_args.make_model() if chat_model_args is not None else None
        self.action_set = HighLevelActionSet(action_subsets, multiaction=multiaction)

        self._obs_preprocessor = make_obs_preprocessor(ObsFlags())

        self.hpa = HPA(
            chat_llm=self.chat_llm,
            action_set=self.action_set,
            budget=budget,
            max_revision_count=max_revision_count,
        )
        self._local_reset()

    def reset(self, seed=None):
        super().reset(seed)
        self._local_reset()

    def _local_reset(self):
        self.hpa.reset()
        self.pending_action_node = None

    @cost_tracker_decorator
    def get_action(self, obs: Any) -> tuple[str | None, dict]:

        obs = self.obs_preprocessor(obs)
        goal = self._extract_goal(obs)

        if self.pending_action_node is not None and isinstance(obs, dict):
            # To be adjusted to the right observation key depending on how we manage the success result
            success = "axtree_txt" in obs and obs.get("axtree_txt") is not None
            self.hpa.finalize_action(self.pending_action_node, obs, success)
            self.pending_action_node = None

        action_node = self.hpa.run_until_action(
            goal=goal,
            obs=obs,
        )

        action = None
        if action_node is not None:
            action = action_node.text
            self.pending_action_node = action_node

        agent_info = {
            "stats": self.chat_llm.get_stats(),
            "hpa": {
                "pending_node_id": (
                    str(self.pending_action_node.id) if self.pending_action_node else None
                ),
                "pending_node_type": (
                    self.pending_action_node.type.name if self.pending_action_node else None
                ),
                "stack_depth": len(self.hpa.stack),
            },
        }
        return action, agent_info

    def _extract_goal(self, obs: dict) -> str:
        """Extract goal string from observation."""
        if "goal" in obs and obs["goal"]:
            return obs["goal"]

        if "goal_object" in obs:
            g_obj = obs["goal_object"]
            if isinstance(g_obj, tuple):
                g_obj = g_obj[0]
            if isinstance(g_obj, dict):
                return g_obj.get("text", "Complete the task.")
            return str(g_obj)

        return "Complete the task."
