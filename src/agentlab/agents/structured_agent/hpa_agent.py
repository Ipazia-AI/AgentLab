from dataclasses import asdict, dataclass
from typing import Any

from browsergym.experiments import AgentInfo

from agentlab.agents import dynamic_prompting as dp
from agentlab.agents.generic_agent.generic_agent import GenericAgent, GenericAgentArgs
from agentlab.agents.structured_agent.analyze import format_hpa_plan_markdown
from agentlab.agents.structured_agent.andor_tree import NodeType
from agentlab.agents.structured_agent.hpa_prompt import (
    ActionVerificationPrompt,
    AndRecoveryPrompt,
    HPAPromptFlags,
    InsightPrompt,
    OrRecoveryPrompt,
    PlanningPrompt,
    SystemActionVerificationPrompt,
    SystemInsightPrompt,
    SystemPlanningPrompt,
)
from agentlab.agents.structured_agent.structured_agent_prompt import MainPrompt
from agentlab.llm.base_api import BaseModelArgs
from agentlab.llm.llm_utils import (
    BaseMessage,
    Discussion,
    ParseError,
    SystemMessage,
    retry,
)
from agentlab.llm.tracking import LLMTracker, cost_tracker_decorator, set_tracker

from .hpa import HPA, Node


@dataclass
class HPAAgentArgs(GenericAgentArgs):
    action_model_args: BaseModelArgs = None
    budget: int = 1000
    flags: HPAPromptFlags = None

    def __post_init__(self):
        planner_name = getattr(self.chat_model_args, "model_name", None)
        action_name = getattr(self.action_model_args, "model_name", None)
        if planner_name and action_name:
            self.agent_name = (
                f"HPAAgent-planner_{planner_name}-actor_{action_name}".replace("/", "_")
            )
        elif planner_name:
            self.agent_name = f"HPAAgent-{planner_name}".replace("/", "_")
        else:
            self.agent_name = "HPAAgent"

    def set_reproducibility_mode(self):
        self.chat_model_args.temperature = 0
        self.action_model_args.temperature = 0

    def prepare(self):
        self.chat_model_args.prepare_server()
        self.action_model_args.prepare_server()

    def close(self):
        self.chat_model_args.close_server()
        self.action_model_args.close_server()

    def make_agent(self) -> GenericAgent:
        return HPAAgent(
            chat_model_args=self.chat_model_args,
            action_model_args=self.action_model_args,
            flags=self.flags,
            budget=self.budget,
            max_retry=self.max_retry,
        )


class HPAAgent(GenericAgent):
    def __init__(
        self,
        chat_model_args: BaseModelArgs | None,
        action_model_args: BaseModelArgs,
        flags: HPAPromptFlags,
        budget: int,
        max_retry: int,
    ):
        self.budget = budget
        self.action_model_args = action_model_args
        self.action_llm = action_model_args.make_model()
        super().__init__(chat_model_args, flags, max_retry)
        self.constraints: str | None = None
        self.progress: str | None = None
        self.suggestion: str | None = None

    @staticmethod
    def _zero_tracker_stats(suffix: str) -> dict:
        return LLMTracker(suffix).stats

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

        with set_tracker(suffix="planner") as planner_tracker:
            if self.pending_action_node is not None and isinstance(obs, dict):
                self.hpa.complete_action_node(
                    self.chat_model_args.model_name,
                    self.constraints,
                    self.progress,
                    self.suggestion,
                    self.obs_history,
                    verification_function=self._infer_action_verification,
                )
                self.pending_action_node = None

            self._infer_insight()
            self.pending_action_node = self.hpa.get_action_node(
                self._infer_plan, self._infer_recovery
            )
        plan_info = self.hpa.get_telemetry(step_index=len(self.actions))
        markdown_page = format_hpa_plan_markdown(plan_info)

        model_extra = {
            "planner_model_args": asdict(self.chat_model_args),
            "action_model_args": asdict(self.action_model_args),
        }

        if self.pending_action_node is not None:

            chat_messages, stats = self._infer_action()
            stats.update(planner_tracker.stats)

            agent_info = AgentInfo(
                think=self.thoughts[-1],
                chat_messages=chat_messages,
                stats=stats,
                markdown_page=markdown_page,
                extra_info={**model_extra, "hpa_plan": plan_info},
            )
            return self.actions[-1], agent_info
        else:
            stats = {**self.chat_llm.get_stats(), **self.action_llm.get_stats()}
            stats.update(planner_tracker.stats)
            stats.update(self._zero_tracker_stats("actor"))
            agent_info = AgentInfo(
                think=None,
                chat_messages=Discussion(),
                stats=stats,
                markdown_page=markdown_page,
                extra_info={**model_extra, "hpa_plan": plan_info},
            )
            return None, agent_info

    def _infer_insight(self):
        ans_dict, _, _ = self._infer(
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
        self._infer(
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

    def _infer_recovery(self, node: Node, tree_context: str):
        if node.type == NodeType.AND:
            prompt_cls = AndRecoveryPrompt
        elif node.type == NodeType.OR:
            prompt_cls = OrRecoveryPrompt
        else:
            raise ValueError(f"Recovery not supported for node type: {node.type}")

        self._infer(
            prompt_cls(
                node=node,
                obs_history=self.obs_history,
                flags=self.flags,
            ),
            SystemMessage(SystemPlanningPrompt().prompt),
        )

    def _infer_action_verification(self, node: Node) -> tuple[bool, str]:
        previous_action = self.actions[-1] if self.actions else None
        previous_thought = self.thoughts[-1] if self.thoughts else None

        ans_dict, _, _ = self._infer(
            ActionVerificationPrompt(
                action_description=node.description,
                obs_history=self.obs_history,
                previous_action=previous_action,
                previous_thought=previous_thought,
                flags=self.flags,
            ),
            SystemMessage(SystemActionVerificationPrompt().prompt),
        )
        result = ans_dict.get("verification_result", "FAILURE")
        explanation = ans_dict.get("verification_explanation", "")
        return result == "SUCCESS", explanation

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

        with set_tracker(suffix="actor") as action_tracker:
            ans_dict, chat_messages, stats = self._infer(
                main_prompt,
                SystemMessage(dp.SystemPrompt().prompt),
                chat_llm=self.action_llm,
                model_args=self.action_model_args,
            )
        stats.update(action_tracker.stats)

        self.actions.append(ans_dict.get("action", None))
        self.memories.append(ans_dict.get("memory", None))
        self.thoughts.append(ans_dict.get("think", None))

        action = self.actions[-1]
        self.pending_action_node.action = action

        print(f"\n{'='*60}")
        print(f"Action Node {self.pending_action_node.id}: {self.pending_action_node.description}")
        print(f"{'='*60}")
        print(f"Inferred Action: {action}")
        print(f"Think: {self.thoughts[-1]}")
        print(f"{'='*60}\n")

        return chat_messages, stats

    def _get_maxes_for(self, model_args: BaseModelArgs):
        maxes = (
            self.flags.max_prompt_tokens,
            model_args.max_total_tokens,
            model_args.max_input_tokens,
        )
        maxes = [m for m in maxes if m is not None]
        max_prompt_tokens = min(maxes) if maxes else None
        max_trunc_itr = (
            self.flags.max_trunc_itr
            if self.flags.max_trunc_itr
            else 20
        )
        return max_prompt_tokens, max_trunc_itr

    def _infer(
        self,
        main_prompt: dp.Shrinkable,
        system_prompt: BaseMessage,
        chat_llm=None,
        model_args: BaseModelArgs | None = None,
    ) -> tuple[dict, Discussion, dict]:
        llm = chat_llm or self.chat_llm
        args = model_args or self.chat_model_args
        max_prompt_tokens, max_trunc_itr = self._get_maxes_for(args)

        human_prompt = dp.fit_tokens(
            shrinkable=main_prompt,
            max_prompt_tokens=max_prompt_tokens,
            model_name=args.model_name,
            max_iterations=max_trunc_itr,
            additional_prompts=system_prompt,
        )
        try:
            chat_messages = Discussion([system_prompt, human_prompt])
            ans_dict = retry(
                llm,
                chat_messages,
                n_retry=self.max_retry,
                parser=main_prompt._parse_answer,
            )
            ans_dict["busted_retry"] = 0
            ans_dict["n_retry"] = (len(chat_messages) - 3) / 2
        except ParseError:
            ans_dict = dict(
                action=None,
                n_retry=self.max_retry + 1,
                busted_retry=1,
            )

        stats = llm.get_stats()
        stats["n_retry"] = ans_dict["n_retry"]
        stats["busted_retry"] = ans_dict["busted_retry"]

        return ans_dict, chat_messages, stats
