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

        hpa_trace = self.hpa.clear_step_trace()
        cumulative_trace = self.hpa.get_cumulative_trace()

        if self.pending_action_node is not None:
            chat_messages, stats = self._infer_action()

            agent_info = AgentInfo(
                think=self.thoughts[-1],
                chat_messages=chat_messages,
                stats=stats,
                extra_info={
                    "chat_model_args": asdict(self.chat_model_args),
                    "hpa_trace": hpa_trace,
                },
                markdown_page=_render_hpa_trace_markdown(hpa_trace),
            )
            agent_info.agent_log = cumulative_trace

            return self.actions[-1], agent_info
        else:
            agent_info = AgentInfo(
                think=None,
                chat_messages=Discussion(),
                stats=self.chat_llm.get_stats(),
                extra_info={
                    "chat_model_args": asdict(self.chat_model_args),
                    "hpa_trace": hpa_trace,
                },
                markdown_page=_render_hpa_trace_markdown(hpa_trace),
            )
            agent_info.agent_log = cumulative_trace
            return None, agent_info

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

    def _infer_plan(self, node: Node, tree_context: str) -> tuple[dict, Discussion]:
        ans_dict, chat_messages, stats = self._infer(
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
        return ans_dict, chat_messages

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


def _render_hpa_trace_markdown(trace: list[dict]) -> str:
    """Build a human-readable markdown page from HPA step trace records."""
    if not trace:
        return "No HPA trace recorded for this step."

    sections: list[str] = []

    expansion_idx = 0
    for record in trace:
        rtype = record.get("type")

        if rtype == "node_expansion":
            expansion_idx += 1
            children = record.get("children", [])
            children_lines = "\n".join(
                f"  - `{c['id']}`: {c['description']}" for c in children
            ) or "  _(none)_"
            sections.append(
                f"## Node Expansion {expansion_idx}\n\n"
                f"**Node:** `{record.get('node_id')}` — {record.get('node_description_before')}\n\n"
                f"**Result type:** `{record.get('node_type_after')}`\n\n"
                f"**Description after:** {record.get('node_description_after')}\n\n"
                f"**Children:**\n{children_lines}\n\n"
                f"<details><summary>Tree context (input to LLM)</summary>\n\n"
                f"```\n{record.get('tree_context', '')}\n```\n\n</details>\n\n"
                f"<details><summary>Tree snapshot before expansion</summary>\n\n"
                f"```\n{_format_tree_snapshot(record.get('tree_snapshot_before', []))}\n```\n\n</details>"
            )

        elif rtype == "stack_snapshot":
            stack_entries = record.get("stack", [])
            stack_lines = "\n".join(
                f"  - `{e['node_id']}` ({e['state']}): {e['description']}"
                for e in stack_entries
            ) or "  _(empty)_"
            sections.append(
                f"## Stack at Action Selection\n\n"
                f"**Action node:** `{record.get('action_node_id')}` — "
                f"{record.get('action_node_description')}\n\n"
                f"**Stack ({len(stack_entries)} entries):**\n{stack_lines}\n\n"
                f"<details><summary>Full tree snapshot</summary>\n\n"
                f"```\n{_format_tree_snapshot(record.get('tree_snapshot', []))}\n```\n\n</details>"
            )

        elif rtype == "global_tree_update":
            result = record.get("result", {})
            pruned = result.get("prune", [])
            updated = result.get("update", {})
            pruned_str = ", ".join(f"`{p}`" for p in pruned) if pruned else "_(none)_"
            updated_str = "\n".join(
                f"  - `{nid}`: {desc}" for nid, desc in updated.items()
            ) if updated else "  _(none)_"
            sections.append(
                f"## Global Tree Update\n\n"
                f"**Pruned nodes:** {pruned_str}\n\n"
                f"**Updated nodes:**\n{updated_str}\n\n"
                f"<details><summary>Tree before update</summary>\n\n"
                f"```\n{_format_tree_snapshot(record.get('tree_snapshot_before', []))}\n```\n\n</details>\n\n"
                f"<details><summary>Tree after update</summary>\n\n"
                f"```\n{_format_tree_snapshot(record.get('tree_snapshot_after', []))}\n```\n\n</details>"
            )

    return "# HPA Step Trace\n\n" + "\n\n---\n\n".join(sections)


def _format_tree_snapshot(nodes: list[dict]) -> str:
    """Format a serialized tree snapshot as indented text."""
    if not nodes:
        return "(empty tree)"
    lines = []
    for n in nodes:
        depth = n["id"].count(".")
        indent = "  " * depth
        lines.append(f"{indent}[{n['status']}] {n['id']} ({n['type']}): {n['description']}")
    return "\n".join(lines)
