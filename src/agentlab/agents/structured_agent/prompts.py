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


@dataclass(frozen=True)
class TaskConstraintsPrompt:
    system_message: str = """\
You are a web-browsing assistant designed to extract structured constraint data from user queries. Given a natural language query, identify and return a list of task constraints that apply to the overall task.
A task constraint is an explicitly stated condition that affects the overall search or task execution.

Instructions:
- Extract only the task constraints explicitly stated in the query.
- Do not infer, assume, or add implicit constraints.
- Return valid JSON only, using the exact key "task_constraints".
- Each constraint must appear only once.
- If no constraints are present, return an empty list.
- Do not include any extra text or Markdown.

Response format (valid JSON):
{
  "task_constraints": [
    "explicit_task_constraint_1",
    "explicit_task_constraint_2"
  ]
}

Example query: "Find a black laptop bag under 40 and a wireless mouse under 25. Both should be delivered within 2 days. URL: www.amazon.com"
Example output:
{
  "task_constraints": [
    "Laptop bag color: black",
    "Laptop price: under $40",
    "Mouse type: wireless",
    "Wireless mouse price: under $25",
    "Delivery time: within 2 days"
  ]
}
"""

    @staticmethod
    def user_prompt(task_objective: str, current_observation: str | None = None) -> dp.Shrinkable:
        return _ShrinkableUserMessage(
            [
                ("QUERY", task_objective),
                (
                    "WEB PAGE CONTENT",
                    _TextTrunkater(current_observation, start_trunkate_iteration=2),
                ),
            ]
        )

    @staticmethod
    def user_message(task_objective: str, current_observation: str | None = None) -> str:
        return TaskConstraintsPrompt.user_prompt(
            task_objective=task_objective,
            current_observation=current_observation,
        ).prompt


@dataclass(frozen=True)
class ObservationSummaryPrompt:
    system_message: str = """\
You are a Context Summarization and Critiquing Agent for web-browsing tasks.
Your job is to maintain an accurate, up-to-date understanding of task progress by analyzing:
- Task description
- Task constraints (if any)
- Task progress summary (if provided)
- Observation history (if provided)
- Action history (if provided)
- Notes summary (if provided)
- Current observation (web page accessibility tree)

Instructions:
1. Do not infer any details or make assumptions.
2. Analyze all available inputs, especially the current accessibility tree.
3. Keep summaries aligned with the main task objective.
4. Do not omit information that may influence decisions or navigation.
5. Make detailed, well-structured, and actionable summaries.
6. Output valid JSON only, using the exact keys below.

Response format (valid JSON):
{
  "observation_summary": "Describe the information from the CURRENT OBSERVATION. Emphasize elements and features relevant for fulfilling the task objective. Include all important detail.",
  "observation_highlights": "List of relevant element IDs from the CURRENT OBSERVATION that are useful for the task.",
  "task_progress": "Summarize actions actually taken and assess each explicitly stated task requirement or constraint. Diagnose why the task is not complete, then give key takeaways.",
  "task_feedback": "Outline what the agent should focus on next to complete the task. (2 sentences)"
}

Example:
{
  "observation_summary": "The page shows search results for 'iPhone 12 Pro Blue 128GB' on Amazon. The first relevant listing is an Apple iPhone 12 Pro, 128GB in Pacific Blue (Renewed) for $314.39. Multiple other iPhone models are also shown including iPhone 12, 13, and 14 in various colors and storage configurations.",
  "observation_highlights": [6028, 6033, 6042, 7204, 9277, 9280, 7242],
  "task_progress": "The agent navigated to an Amazon search results page and identified a relevant listing matching the model, color, storage, and condition constraints. All explicitly defined constraints appear satisfied on the listing. However, the task is not complete because the agent has not clicked the product, verified details on the product page, or added the item to the cart. The key takeaway is that discovery is done and verification plus checkout steps remain.",
  "task_feedback": "Click the product listing to verify details on the product page, then add it to the cart. Confirm the exact model and specs before completing the task."
}
"""

    @staticmethod
    def user_prompt(
        task_description: str,
        observation: str,
        task_constraints: str | list[str] | None = None,
        task_progress_summary: str | None = None,
        observation_history: str | list[str] | None = None,
        action_history: str | list[str] | None = None,
        notes_summary: str | None = None,
    ) -> dp.Shrinkable:
        return _ShrinkableUserMessage(
            [
                ("TASK DESCRIPTION", task_description),
                ("TASK CONSTRAINTS", task_constraints),
                ("TASK PROGRESS SUMMARY", task_progress_summary),
                (
                    "OBSERVATION HISTORY",
                    _TextTrunkater(observation_history, start_trunkate_iteration=4),
                ),
                ("ACTION HISTORY", _TextTrunkater(action_history, start_trunkate_iteration=4)),
                ("NOTES SUMMARY", _TextTrunkater(notes_summary, start_trunkate_iteration=4)),
                ("CURRENT OBSERVATION", _TextTrunkater(observation, start_trunkate_iteration=2)),
            ]
        )

    @staticmethod
    def user_message(
        task_description: str,
        observation: str,
        task_constraints: str | list[str] | None = None,
        task_progress_summary: str | None = None,
        observation_history: str | list[str] | None = None,
        action_history: str | list[str] | None = None,
        notes_summary: str | None = None,
    ) -> str:
        return ObservationSummaryPrompt.user_prompt(
            task_description=task_description,
            observation=observation,
            task_constraints=task_constraints,
            task_progress_summary=task_progress_summary,
            observation_history=observation_history,
            action_history=action_history,
            notes_summary=notes_summary,
        ).prompt


@dataclass(frozen=True)
class NotesSummaryPrompt:
    system_message: str = """\
You are an advanced web-browsing agent that generates notes from the current observation and forms a response to the task using those notes.

You are given:
- Task description
- Task constraints (if any)
- Task progress summary (if provided)
- Action history (if provided)
- Previous notes (if provided)
- Current observation (web page accessibility tree)

Instructions:
1. Do not infer any details or make assumptions.
2. Analyze all available inputs, especially the current web page's accessibility tree.
3. Do not omit information that may influence decisions or navigation.
4. Output valid JSON only, using the exact keys below.

Response format (valid JSON):
{
  "new_notes": "Identify all new information grounded in observation that can help complete the task.",
  "task_response": "Provide the best possible response to the task using only the notes and action history. Do not add new information."
}

Example:
{
  "new_notes": "Found relevant iPhone 12 Pro listing matching requirements: model iPhone 12 Pro, 128GB, Pacific Blue, price $314.39, condition renewed, fully unlocked, rating 4.1/5 from 12,669 reviews.",
  "task_response": "Based on the task requirements, I found an Apple iPhone 12 Pro with 128GB storage in Pacific Blue, fully unlocked, in renewed condition, priced at $314.39, with a 4.1/5 rating from 12,669 reviews."
}
"""

    @staticmethod
    def user_prompt(
        task_description: str,
        observation: str,
        task_constraints: str | list[str] | None = None,
        task_progress_summary: str | None = None,
        action_history: str | list[str] | None = None,
        notes: str | None = None,
    ) -> dp.Shrinkable:
        return _ShrinkableUserMessage(
            [
                ("TASK DESCRIPTION", task_description),
                ("TASK CONSTRAINTS", task_constraints),
                ("TASK PROGRESS SUMMARY", task_progress_summary),
                ("ACTION HISTORY", _TextTrunkater(action_history, start_trunkate_iteration=4)),
                ("PREVIOUS NOTES", _TextTrunkater(notes, start_trunkate_iteration=4)),
                ("CURRENT OBSERVATION", _TextTrunkater(observation, start_trunkate_iteration=2)),
            ]
        )

    @staticmethod
    def user_message(
        task_description: str,
        observation: str,
        task_constraints: str | list[str] | None = None,
        task_progress_summary: str | None = None,
        action_history: str | list[str] | None = None,
        notes: str | None = None,
    ) -> str:
        return NotesSummaryPrompt.user_prompt(
            task_description=task_description,
            observation=observation,
            task_constraints=task_constraints,
            task_progress_summary=task_progress_summary,
            action_history=action_history,
            notes=notes,
        ).prompt


@dataclass(init=False)
class NodeExpansionPrompt:
    action_prompt: str

    def __init__(self, action_set: AbstractActionSet, action_flags: dp.ActionFlags) -> None:
        self.action_prompt = dp.ActionPrompt(
            action_set,
            action_flags=action_flags,
        ).prompt

    def system_message(self) -> str:
        return f"""\
You are an efficient logical (AND/OR) tree constructing agent specialized in web-browsing tasks. You solve complex problems using AND/OR planning trees.
You dynamically construct AND/OR planning trees from observations of the webpage’s accessibility tree structure for EFFICIENT and robust task execution.

You are provided:
- TASK DESCRIPTION: a textual description of the task to complete
- TASK CONSTRAINTS: a list of constraints that must be satisfied to complete the task
- TASK PROGRESS SUMMARY: a summary of the current task progress
- NOTES SUMMARY: a summary of the notes taken so far
- OBSERVATION: the current observation of the webpage's accessibility tree
- NODE INFORMATIONS:
    - node_id: the ID of the node to analyze
    - node_description: the description of the node to analyze
    - local_tree_info: information about the node’s siblings and the node parent’s siblings


There are three possible node types that can be used to construct the tree:
- AND Node: Represents an ordered list of logical subgoals required to achieve the node’s objective.
- OR Node: Represents alternative sub-strategies (which can be other AND/OR nodes).
- ACTION Node: Single executable action strictly matching one element of the list of browser actions below.


{self.action_prompt}

------------------------------------------------------------------------------------------------

Your task for the given node:
1. Determine whether the node is an AND node, an OR node, or an ACTION node.
2. Choose ONE of the following options:
   A. Mark node as ACTION if the goal can be achieved using a single atomic action from the list above.
   B. Expand the node if the goal cannot be solved by performing a single atomic action.
      - For AND nodes, provide the ordered list of logical subgoals.
      - For OR nodes, provide a list of alternative strategies ordered by likelihood of success, including a (0–1) score in each string (e.g., “Strategy here (score: 0.85)”).
      - Do not add speculative or redundant subgoals.

IMPORTANT RULES:
- If you think the task can be completed through an action, mark the node as an action.
- If you need to expand the task in subtasks, focus on expansions that will complete the task faster with high probability.
    - For AND nodes, ensure temporal order of children is correct and efficient.
- Do not output anything outside the specified JSON format.

Output must be valid JSON using one of the following formats:

Format 1 (node type ACTION):
{{
  "node_id": "ID of the node here",
  "node_description": "Describe what the atomic action does",
  "node_type": "ACTION",
  "expansion": "Exact atomic action from the list of browser actions",
  "reasoning": "Brief justification explaining why the node is classified as ACTION"
}}

Format 2 (node type AND):
{{
  "node_id": "ID of the node here",
  "node_description": "Description of the node here",
  "node_type": "AND",
  "expansion": [
    "Provide a textual description of the first subgoal. If the subgoal is an action, provide only a general description of the action, not the actual element to perform the action on",
    "Provide a textual description of the second subgoal. If the subgoal is an action, provide only a general description of the action, not the actual element to perform the action on"
  ],
  "reasoning": "Brief justification explaining why the node is classified as AND"
}}

Format 3 (node type OR):
{{
  "node_id": "ID of the node here",
  "node_description": "Description of the node here",
  "node_type": "OR",
  "expansion": [
    "Provide a textual description of the first scored alternative strategy. If the strategy is an action, provide only a general description of the action, not the actual element to perform the action on",
    "Provide a textual description of the second scored alternative strategy. If the strategy is an action, provide only a general description of the action, not the actual element to perform the action on"
  ],
  "reasoning": "Brief justification explaining why the node is classified as OR"
}}
"""

    @staticmethod
    def user_prompt(
        task_description: str,
        task_constraints: str | list[str] | None,
        task_progress_summary: str | None,
        notes_summary: str | None,
        observation: str,
        node_id: str,
        node_description: str,
        local_tree_info: str,
    ) -> dp.Shrinkable:
        return _ShrinkableUserMessage(
            [
                ("TASK DESCRIPTION", task_description),
                ("TASK CONSTRAINTS", task_constraints),
                (
                    "TASK PROGRESS SUMMARY",
                    _TextTrunkater(task_progress_summary, start_trunkate_iteration=4),
                ),
                ("NOTES SUMMARY", _TextTrunkater(notes_summary, start_trunkate_iteration=4)),
                ("OBSERVATION", _TextTrunkater(observation, start_trunkate_iteration=2)),
                ("node_id", node_id),
                ("node_description", node_description),
                (
                    "local_tree_info",
                    _TextTrunkater(local_tree_info, start_trunkate_iteration=4),
                ),
            ]
        )

    @staticmethod
    def user_message(
        task_description: str,
        task_constraints: str | list[str] | None,
        task_progress_summary: str | None,
        notes_summary: str | None,
        observation: str,
        node_id: str,
        node_description: str,
        local_tree_info: str,
    ) -> str:
        return NodeExpansionPrompt.user_prompt(
            task_description=task_description,
            task_constraints=task_constraints,
            task_progress_summary=task_progress_summary,
            notes_summary=notes_summary,
            observation=observation,
            node_id=node_id,
            node_description=node_description,
            local_tree_info=local_tree_info,
        ).prompt


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
- Task progress summary 
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
        task_progress_summary: str | None,
        notes_summary: str | None,
        observation: str,
        global_tree_info: str,
    ) -> dp.Shrinkable:
        return _ShrinkableUserMessage(
            [
                ("ROOT-LEVEL TASK DESCRIPTION", task_description),
                ("TASK CONSTRAINTS", task_constraints),
                (
                    "TASK PROGRESS SUMMARY",
                    _TextTrunkater(task_progress_summary, start_trunkate_iteration=4),
                ),
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
        task_progress_summary: str | None,
        notes_summary: str | None,
        observation: str,
        global_tree_info: str,
    ) -> str:
        return GlobalTreeUpdatePrompt.user_prompt(
            task_description=task_description,
            task_constraints=task_constraints,
            task_progress_summary=task_progress_summary,
            notes_summary=notes_summary,
            observation=observation,
            global_tree_info=global_tree_info,
        ).prompt
