from dataclasses import asdict, dataclass
from typing import Any

from browsergym.experiments import AgentInfo

from agentlab.agents import dynamic_prompting as dp
from agentlab.agents.generic_agent.generic_agent import GenericAgent, GenericAgentArgs
from agentlab.agents.structured_agent.hpa_prompt import (
    HPAPromptFlags,
    InsightPrompt,
    PlanningPrompt,
    SystemInsightPrompt,
    SystemPlanningPrompt,
)
from agentlab.agents.structured_agent.structured_agent_prompt import (
    GenericPromptFlags,
    MainPrompt,
)
from agentlab.llm.base_api import BaseModelArgs
from agentlab.llm.llm_utils import (
    BaseMessage,
    Discussion,
    ParseError,
    SystemMessage,
    retry,
)
from agentlab.llm.tracking import cost_tracker_decorator

from .hpa import HPA, Node


@dataclass
class HPAAgentArgs(GenericAgentArgs):
    budget: int = 1000
    flags: HPAPromptFlags = None

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
            max_retry=self.max_retry,
        )


class HPAAgent(GenericAgent):
    def __init__(
        self,
        chat_model_args: BaseModelArgs | None,
        flags: HPAPromptFlags,
        budget: int,
        max_retry: int,
    ):
        self.budget = budget
        super().__init__(chat_model_args, flags, max_retry)
        self.constraints: str | None = None
        self.progress: str | None = None
        self.suggestion: str | None = None

    def reset(self, seed=None):
        super().reset(seed)

        self.pending_action_node = None
        self.constraints = None
        self.progress = None
        self.suggestion = None
        self.max_depth = 4
        self.hpa = HPA(
            chat_llm=self.chat_llm,
            action_set=self.action_set,
            flags=self.flags,
            budget=self.budget,
            max_depth=self.max_depth,
        )

    @cost_tracker_decorator
    def get_action(self, obs: Any):

        self.obs_history.append(obs)

        if len(self.obs_history) == 1:
            self.hpa.set_goal(self.obs_history[0])

        if self.pending_action_node is not None and isinstance(obs, dict):
            # To be adjusted to the right observation key depending on how we manage the success result
            self.hpa.complete_pending(
                self.chat_model_args.model_name,
                self.constraints,
                self.progress,
                self.suggestion,
                self.obs_history,
            )
            self.pending_action_node = None

        self._infer_insight()
        self.pending_action_node = self.hpa.get_action_node(self._infer_plan)
        if self.pending_action_node is not None:
            chat_messages, stats = self._infer_action()

            agent_info = AgentInfo(
                think=self.thoughts[-1],
                chat_messages=chat_messages,
                stats=stats,
                extra_info={"chat_model_args": asdict(self.chat_model_args)},
            )

            return self.actions[-1], agent_info
        else:
            return None, AgentInfo(
                think=None,
                chat_messages=Discussion(),
                stats=self.chat_llm.get_stats(),
                extra_info={"chat_model_args": asdict(self.chat_model_args)},
            )

    def _infer_insight(self):
        ans_dict, chat_messages, stats = self._infer(
            InsightPrompt(
                obs_history=self.obs_history,
                actions=self.actions,
                memories=self.memories,
                thoughts=self.thoughts,
                flags=self.flags,
            ),
            SystemMessage(SystemInsightPrompt().prompt),
        )

        self.constraints = ans_dict.get("constraint", None)
        self.progress = ans_dict.get("progress", None)
        self.suggestion = ans_dict.get("suggestion", None)

    def _infer_plan(self, node: Node, tree_context: str):
        _, chat_messages, stats = self._infer(
            PlanningPrompt(
                node=node,
                max_depth=self.max_depth,
                obs_history=self.obs_history,
                actions=self.actions,
                memories=self.memories,
                thoughts=self.thoughts,
                constraints=self.constraints,
                progress=self.progress,
                suggestion=self.suggestion,
                tree_context=tree_context,
                flags=self.flags,
            ),
            SystemMessage(SystemPlanningPrompt().prompt),
        )

    def _infer_action(self) -> tuple[Discussion, dict]:

        main_prompt = MainPrompt(
            action_set=self.action_set,
            obs_history=self.obs_history,
            actions=self.actions,
            memories=self.memories,
            thoughts=self.thoughts,
            current_plan_step=self.pending_action_node.description,
            completed_plan_steps=self.hpa.get_plan()[0],
            future_plan_steps=self.hpa.get_plan()[1],
            flags=self.flags,
        )

        ans_dict, chat_messages, stats = self._infer(
            main_prompt,
            SystemMessage(dp.SystemPrompt().prompt),
        )

        self.actions.append(ans_dict.get("action", None))
        self.memories.append(ans_dict.get("memory", None))
        self.thoughts.append(ans_dict.get("think", None))

        print(f"\n{'='*60}")
        print(f"Action for node {self.pending_action_node.id}: {self.pending_action_node.description}")
        print(f"{'='*60}")
        # print(f"Prompt:\n{chat_messages}")
        # print(f"{'='*60}")
        print(f"Action taken: {self.actions[-1]}")
        print(f"Think: {self.thoughts[-1]}")
        print(f"{'='*60}\n")

        return chat_messages, stats

    def _infer(
        self, main_prompt: dp.Shrinkable, system_prompt: BaseMessage
    ) -> tuple[dict, Discussion, dict]:
        max_prompt_tokens, max_trunc_itr = self._get_maxes()

        system_prompt = SystemMessage(dp.SystemPrompt().prompt)

        human_prompt = dp.fit_tokens(
            shrinkable=main_prompt,
            max_prompt_tokens=max_prompt_tokens,
            model_name=self.chat_model_args.model_name,
            max_iterations=max_trunc_itr,
            additional_prompts=system_prompt,
        )
        try:
            chat_messages = Discussion([system_prompt, human_prompt])
            ans_dict = retry(
                self.chat_llm,
                chat_messages,
                n_retry=self.max_retry,
                parser=main_prompt._parse_answer,
            )
            ans_dict["busted_retry"] = 0
            # inferring the number of retries, TODO: make this less hacky
            ans_dict["n_retry"] = (len(chat_messages) - 3) / 2
        except ParseError as e:
            ans_dict = dict(
                action=None,
                n_retry=self.max_retry + 1,
                busted_retry=1,
            )

        stats = self.chat_llm.get_stats()
        stats["n_retry"] = ans_dict["n_retry"]
        stats["busted_retry"] = ans_dict["busted_retry"]

        return ans_dict, chat_messages, stats
