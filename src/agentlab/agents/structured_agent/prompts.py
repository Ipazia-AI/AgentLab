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
You are an agent that operates as the global manager of a logical (AND/OR) tree that represents the execution plan for a web-browsing task.
Your job is to manage the tree by changing the nodes based on the current available information so that the task is executed more efficiently.

Possible Node Types:
- AND Node: Represents an ordered list of logical subgoals required to achieve the node’s objective. The children of an AND node are executed in the order they are listed. To be completed, all children must be completed successfully.
- OR Node: Represents alternative sub-strategies (which can be other AND/OR nodes). The children of an OR node represent different ways to achieve the node's objective. To be completed, at least one child must be completed successfully.
- ACTION Node: Single executable action strictly matching one element of the list of browser actions described below. The action node is a leaf node that represents a single action to be executed on the webpage.

Node status indicators:
- VISITED: for visited nodes
- UNVISITED: for unvisited nodes
- PRUNED: for pruned nodes
- SUCCESS: for completed nodes 
- FAIL: for temporarily failed nodes

Each node has a NODE ACTION field that shows the exact browser command executed for
ACTION nodes (e.g. click('a1201'), hover('a456')). Use this information to detect when multiple
nodes attempted the same action — this is a sign of a stuck loop that should be pruned
or updated. Note: the element bids in these actions are historical and may no longer
be valid on the current page.

This is the list of browser actions that can be performed on the webpage:
{self.action_prompt}

You are provided: 
- The overall goal of the task
- The Current tree status. For each node, you are provided with the node id, node type, node description, node status and the node action.
- The id of the node which action was executed as last on the webpage. 
- The accessibility tree structure of the current webpage as the observation after the action was executed.
- The overall task progress summary so far.
- A suggestion for the next action to take.

----------------------------------
Your task is to carefully analyze all the information provided and manipulate the tree to determine which nodes to prune and how to adapt the nodes with updated goal descriptions.
This refinement operation serves to reduce the tree complexity by removing unpromising branches, avoid redundancies in the tasks and improve the overall task execution efficiency.
PRUNE nodes that are no longer relevant or are duplicates. A node is considered duplicate if it has the same objective as another node in the tree. A node is irrelevant if it is no more necessary to achieve the overall task.
UPDATE the description of the nodes to set the new goal of the node based on the current observation and the progress summary. The update should serve to refine the sub-goals of the plan in order to better achieve the overall task.

## IMPORTANT RULES:
    ### PRUNING RULES:
        - You are only allowed to PRUNE nodes which status is not DELETED, PRUNED, or SUCCESS. 
        - NEVER prune UNVISITED or VISITED children of OR nodes — they are untried alternatives that
        must be preserved as fallback strategies. The tree traversal logic will try them if needed.
        - NEVER prune nodes with SUCCESS status.
        - Only use existing node IDs from the current tree; do not create new node IDs or subtrees.
        - Pruning is not always necessary. Just fill the fields with empty lists or dictionaries if no changes are needed. Be precise in the changes you make, delete only nodes that do not lead to any further progress in the task.
    ### UPDATING RULES:
        - You are only allowed to UPDATE nodes which hasn't been processed yet, i.e. which status is UNVISITED.
        - The update should serve to refine the sub-goals of the plan in order to better achieve the overall task.
        - Do not include in the description precise information about the current state of the observation. Write instead a general description of the node's objective.
        - Be aware of the structure of the tree and the relationships between the nodes before updating, especially when changing the description of a parent node.
        - The update should be based on the current state of information available.
        - The update should be precise in the changes you make, update only nodes that need to be updated.
        - The update is not always necessary. Just fill the fields with empty lists or dictionaries if no changes are needed.
        - NEVER add a particular action to the description of the node, just the general description of the node's objective.
    ### GENERAL RULES:
        - NEVER PRUNE or UPDATE the same nodes, you can only prune or update a node once.

Provide your answer in the following JSON format:

{{
  "prune": [
    "node_id of the first node to prune",
	"node_id of the second node to prune",
	"node_id of the third node to prune"
  ],
  "update": {{
    "node_id of the first node to update": "new goal description for the node",
    "node_id of the second node to update": "new goal description for the node",
    "node_id of the third node to update": "new goal description for the node",
  }}
}}

Example:
{{
  "prune": [
	"1.2",
	"1.3"
  ],
  "update": {{
    "2.2": "new goal description for the node",
    "2.3": "new goal description for the node",
  }}
}}
"""

    @staticmethod
    def user_prompt(
        task_description: str,
        node_id: str,
        constraints: str,
        progress: str,
        suggestion: str,
        observation: str,
        global_tree_info: str,
    ) -> dp.Shrinkable:
        return _ShrinkableUserMessage(
            [
                ("OVERALL GOAL OF THE TASK", task_description),
                ("CURRENT NODE ID", node_id),
                ("TASK CONSTRAINTS", constraints),
                ("TASK PROGRESS SUMMARY", _TextTrunkater(progress, start_trunkate_iteration=4)),
                ("NEXT STEP SUGGESTION", _TextTrunkater(suggestion, start_trunkate_iteration=4)),
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
