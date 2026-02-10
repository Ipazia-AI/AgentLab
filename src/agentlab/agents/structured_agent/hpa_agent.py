from dataclasses import dataclass
from typing import Any

from browsergym.core.action.highlevel import HighLevelActionSet
from browsergym.experiments import AgentInfo

from agentlab.agents.agent_args import AgentArgs
from agentlab.agents.dynamic_prompting import ObsFlags, make_obs_preprocessor
from agentlab.agents.generic_agent.generic_agent import GenericAgent, GenericAgentArgs
from agentlab.agents.generic_agent.generic_agent_prompt import GenericPromptFlags
from agentlab.llm.base_api import BaseModelArgs
from agentlab.llm.llm_utils import AIMessage, Discussion, SystemMessage
from agentlab.llm.tracking import cost_tracker_decorator

from .andor_tree import Node, NodeType
from .hpa import HPA


@dataclass
class HPAAgentArgs(GenericAgentArgs):
    chat_model_args: BaseModelArgs | None = None
    budget: int = 1000
    max_revision_count: int = 3
    multiaction: bool = False

    def __post_init__(self):
        if self.chat_model_args is not None:
            self.agent_name = f"HPAAgent-{self.chat_model_args.model_name}".replace("/", "_")
        else:
            self.agent_name = "HPAAgent"

    def make_agent(self) -> GenericAgent:
        return HPAAgent(
            chat_model_args=self.chat_model_args,
            flags=self.flags,
            budget=self.budget,
            max_revision_count=self.max_revision_count,
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
        flags: GenericPromptFlags,
        budget: int,
        max_revision_count: int,
        multiaction: bool,
    ):
        self.chat_llm = chat_model_args.make_model() if chat_model_args is not None else None
        self.flags = flags
        self.action_set = self.flags.action.action_set.make_action_set()
        self._obs_preprocessor = make_obs_preprocessor(self.flags.obs)

        self.hpa = HPA(
            chat_llm=self.chat_llm,
            action_flags=self.flags.action,
            action_set=self.action_set,
            budget=budget,
            max_revision_count=max_revision_count,
            max_prompt_tokens=self.flags.max_prompt_tokens,
            max_trunc_itr=self.flags.max_trunc_itr,
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
            self.hpa.finalize_action(self.pending_action_node, obs)
            self.pending_action_node = None

        action_node = self.hpa.run_until_action(
            goal=goal,
            obs=obs,
        )

        action = None
        if action_node is not None:
            action = action_node.action
            self.pending_action_node = action_node

        # Build chat messages for browsergym chat interface
        # Include goal and action history for user visibility
        chat_messages = Discussion()

        # Add system message with goal
        system_content = f"Goal: {goal}\n\nYou are using AND/OR tree to select actions."
        chat_messages.add_message(SystemMessage(system_content))

        # Add user message with current observation context (if available)
        if obs.get("chat_messages"):
            # Browsergym chat_messages use format: {'role': str, 'message': str, 'timestamp': float}
            # Convert to standard format
            for msg in obs["chat_messages"]:
                if isinstance(msg, dict):
                    role = msg.get("role", "user")
                    # Handle browsergym format (uses 'message') vs standard format (uses 'content')
                    content = msg.get("content") or msg.get("message", "")
                    if content:
                        chat_messages.add_message({"role": role, "content": content})
        else:
            # Otherwise, create a user message with goal
            chat_messages.add_message({"role": "user", "content": f"Task: {goal}"})

        # Add assistant message with the selected action and reasoning
        assistant_content = f"Action: {action}"
        # if len(self.actions) > 1:
        #     assistant_content += (
        #         f"\n\nPrevious actions: {', '.join(str(a) for a in self.actions[:-1])}"
        #     )
        chat_messages.add_message(AIMessage(assistant_content))

        # Return format expected by BrowserGym/AgentLab
        agent_info = AgentInfo(
            chat_messages=chat_messages,
            stats=self.chat_llm.get_stats(),
            extra_info={
                "hpa": {
                    "pending_node_id": (
                        str(self.pending_action_node.id) if self.pending_action_node else None
                    ),
                    "pending_node_type": (
                        self.pending_action_node.type.name if self.pending_action_node else None
                    ),
                    "stack_depth": len(self.hpa.stack),
                },
            },
        )

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
