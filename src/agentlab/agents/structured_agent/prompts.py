from __future__ import annotations

from dataclasses import dataclass

from bgym import AbstractActionSet

from agentlab.agents import dynamic_prompting as dp


class _TextTrunkater(dp.Trunkater):
    def __init__(
        self,
        value: str | list[str] | None,
        visible: bool = True,
        start_trunkate_iteration: int = 3,
        shrink_speed: float = 0.3,
    ) -> None:
        super().__init__(
            visible=visible,
            start_trunkate_iteration=start_trunkate_iteration,
            shrink_speed=shrink_speed,
        )
        self._prompt = _normalize_block(value)


class _ShrinkableUserMessage(dp.Shrinkable):
    def __init__(self, blocks: list[tuple[str, str | dp.PromptElement | None]]) -> None:
        super().__init__()
        self._blocks = blocks

    def shrink(self) -> None:
        for _, value in self._blocks:
            if isinstance(value, dp.Shrinkable):
                value.shrink()

    @property
    def _prompt(self) -> str:
        message = ""
        for label, value in self._blocks:
            if isinstance(value, dp.PromptElement):
                value = value.prompt
            message += _optional_block(label, value)
        return message


def _normalize_block(value) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        value = "\n".join(str(v) for v in value if v is not None)
    return str(value).strip()


def _optional_block(label: str, value) -> str:
    content = _normalize_block(value)
    if not content:
        return ""
    return f"{label}:\n{content}\n"


@dataclass(init=False)
class GlobalTreeUpdatePrompt:
    system_message: str

    def __init__(self, action_set: AbstractActionSet, action_flags: dp.ActionFlags) -> None:
        self.action_prompt = dp.ActionPrompt(
            action_set,
            action_flags=action_flags,
        ).prompt

    def system_message(self) -> str:
        return f"""\
You are a global logical (AND/OR) tree update agent revising an existing logical planning tree for a web-browsing task based on current available information so that the task is executed more efficiently. 
Do not change the ordering of the sub-plans.

Possible Node Types:
- AND Node: Represents an ordered list of logical subgoals required to achieve the node’s objective.
- OR Node: Represents alternative sub-strategies (which can be other AND/OR nodes).
- ACTION Node: Single executable action strictly matching one element of the list of browser actions described below.

Node status indicators:
- VISITED: for visited nodes
- UNVISITED: for unvisited nodes
- PRUNED: for pruned nodes
- SUCCESS: for completed nodes 
- FAIL: for temporarily failed nodes

{self.action_prompt}

You are provided: 
- The root-level task description 
- Current AND/OR tree description 
- The accessibility tree structure of the current webpage as the observation 
- Notes summary: Summary of notes taken by the agent during the task

Your task is to carefully analyze all the information provided and determine which nodes to prune and which nodes to update.
Apply changes in this strict order:
1. PRUNE nodes that are no longer relevant or are duplicates. 
2. UPDATE node descriptions if intent is unchanged but content needs minor revision.

Important Rules: 
- You are only allowed to PRUNE or UPDATE nodes that have not been deleted, or pruned, or marked succesful. 
- Do not add status of the node while updating the description. 
- Only use existing node IDs from the current tree; do not create new node IDs or subtrees. 
- Do NOT PRUNE children that are necessary for satisfying the parent node’s objective. 
- Do not change node types. 
- Do not change the ordering of the sub-plans.
- Updating or pruning is not always necessary. Just fill the fields with empty lists or dictionaries if no changes are needed.

First reason about the update to the tree based on the information given to you (task description, task constraints, task progress summary, notes summary, current AND/OR tree description, observation) and then give your answer in the following format. 
Use the AND/OR tree description to ensure that you are not repeating any subgoals that have already been considered. 
Format your output as JSON.

Formatting Instructions:
{{
  "prune": [
    "node_id of the first node to prune",
	"node_id of the second node to prune",
	"node_id of the third node to prune"
  ],

  "update": {
	"node_id of the first node to update": "Describe here the node’s new objective (subgoal/strategy/description of action)",
	"node_id of the second node to update": "Describe here the node’s new objective (subgoal/strategy/description of action)",
	"node_id of the third node to update": "Describe here the node’s new objective (subgoal/strategy/description of action)"
  }
}}

Example:
{{
  "prune": [
	"1.2",
	"1.3"
  ],

  "update": {
	"0.1.2": "type eggless cake in search bar",
	"0.2": "Find an eggless cake recipe with over 60 votes and at least 4.5 star rating by examining search results"
  }
}}
"""

    @staticmethod
    def user_prompt(
        task_description: str,
        task_constraints: str | list[str] | None,
        notes_summary: str | None,
        observation: str,
        global_tree_info: str,
    ) -> dp.Shrinkable:
        return _ShrinkableUserMessage(
            [
                ("ROOT-LEVEL TASK DESCRIPTION", task_description),
                ("TASK CONSTRAINTS", task_constraints),
                ("NOTES SUMMARY", _TextTrunkater(notes_summary, start_trunkate_iteration=4)),
                ("OBSERVATION", _TextTrunkater(observation, start_trunkate_iteration=2)),
                (
                    "GLOBAL TREE INFORMATION",
                    _TextTrunkater(global_tree_info, start_trunkate_iteration=4),
                ),
            ]
        )

    @staticmethod
    def user_message(
        task_description: str,
        task_constraints: str | list[str] | None,
        notes_summary: str | None,
        observation: str,
        global_tree_info: str,
    ) -> str:
        return GlobalTreeUpdatePrompt.user_prompt(
            task_description=task_description,
            task_constraints=task_constraints,
            notes_summary=notes_summary,
            observation=observation,
            global_tree_info=global_tree_info,
        ).prompt
