import abc
import logging
from dataclasses import dataclass

from browsergym.core.action.highlevel import AbstractActionSet

from agentlab.agents import dynamic_prompting as dp
from agentlab.agents.generic_agent.generic_agent_prompt import GenericPromptFlags
from agentlab.llm.llm_utils import (
    HumanMessage,
    ParseError,
    parse_html_tags,
    parse_html_tags_raise,
)


@dataclass
class HPAPromptFlags(dp.Flags):
    action: dp.ActionFlags
    obs: dp.ObsFlags
    use_constraints: bool = True
    use_progress: bool = True
    use_suggestion: bool = True
    use_planning: bool = True
    use_concrete_example: bool = True
    use_abstract_example: bool = False
    use_hints: bool = False
    use_plan: bool = False  #
    be_cautious: bool = False
    use_criticise: bool = False  #
    use_thinking: bool = False
    use_memory: bool = False  #
    max_prompt_tokens: int = None
    max_trunc_itr: int = 20
    extra_instructions: str | None = None


class Constraint(dp.PromptElement):
    _prompt = ""
    _abstract_ex = """
<constraints>Identify a list of constraints that apply to the overall goal.
A constraint is an explicitly stated condition that affects the goal execution.</constraints>
"""

    _concrete_ex = """
<constraints><constraint>Laptop bag color: black</constraint>
<constraint>Laptop price: under $40</constraint>
<constraint>Mouse type: wireless</constraint>
<constraint>Wireless mouse price: under $25</constraint>
<constraint>Delivery time: within 2 days</constraint></constraints>
"""

    def _parse_answer(self, text_answer):
        try:
            return parse_html_tags_raise(text_answer, keys=["constraint"], merge_multiple=True)
        except ParseError as e:
            return {"constraint": text_answer, "parse_error": str(e)}


class Progress(dp.PromptElement):
    _prompt = ""
    _abstract_ex = """
<progress>Summarize actions actually taken and assess each explicitly stated task requirement or constraint.
Diagnose why the task is not complete.</progress>
"""

    _concrete_ex = """
<progress>The agent navigated to an Amazon search results page and identified a relevant listing matching the model, color, storage, and condition constraints.
All explicitly defined constraints appear satisfied on the listing.
However, the task is not complete because the agent has not clicked the product, verified details on the product page, or added the item to the cart.</progress>
"""

    def _parse_answer(self, text_answer):
        try:
            return parse_html_tags_raise(text_answer, keys=["progress"], merge_multiple=True)
        except ParseError as e:
            return {"progress": text_answer, "parse_error": str(e)}


class Suggestion(dp.PromptElement):

    _prompt = ""
    _abstract_ex = """
<suggestion>Identify all new information grounded in observation that can help complete the task.</suggestion>
"""

    _concrete_ex = """
<suggestion>Found relevant iPhone 12 Pro listing matching requirements: model iPhone 12 Pro, 128GB, 
Pacific Blue, price $314.39, condition renewed, fully unlocked, rating 4.1/5 from 12,669 reviews.</suggestion>
"""

    def _parse_answer(self, text_answer):
        try:
            return parse_html_tags_raise(text_answer, keys=["suggestion"], merge_multiple=True)
        except ParseError as e:
            return {"suggestion": text_answer, "parse_error": str(e)}


class InsightInstructions(dp.PromptElement):
    def __init__(self, goal_object, visible: bool = True, extra_instructions=None) -> None:
        super().__init__(visible)

        self._prompt = [
            dict(
                type="text",
                text="""
# Instructions

You are an excellent observer, very skilled at understanding how to interact with web pages, your goal is 
to discover insight related to the task that is being performed by a user.

Review the instructions from the user, the current state of the page and all other information
to find useful insights that can help the user accomplish their task.

## Goal:
""",
            )
        ]

        self._prompt += goal_object

        if extra_instructions:
            self._prompt += [
                dict(
                    type="text",
                    text=f"""

## Extra instructions:

{extra_instructions}
""",
                )
            ]


class SystemInsightPrompt(dp.PromptElement):
    _prompt = """\
You are an agent helping another agent to plan how to solve a web task based on the content of the page and
user instructions. Each time you provide insights those will be used as input for a planning algorithm 
to propose a plan to accomplish the task. You should be very concise and to the point, you should not provide
any additional information that is not related to the task."""


class InsightPrompt(dp.Shrinkable):
    def __init__(
        self,
        obs_history: list[dict],
        actions: list[str],
        memories: list[str],
        thoughts: list[str],
        flags: HPAPromptFlags,
    ):
        super().__init__()
        self.flags = flags
        self.history = dp.History(obs_history, actions, memories, thoughts, flags.obs)
        self.instructions = InsightInstructions(
            obs_history[-1]["goal_object"], extra_instructions=flags.extra_instructions
        )
        self.constraints = Constraint(visible=lambda: flags.use_constraints)
        self.progress = Progress(visible=lambda: flags.use_progress)
        self.suggestion = Suggestion(visible=lambda: flags.use_suggestion)

        self.obs = dp.Observation(
            obs_history[-1],
            self.flags.obs,
        )

    @property
    def _prompt(self) -> HumanMessage:
        prompt = HumanMessage(self.instructions.prompt)
        prompt.add_text(
            f"""\
{self.obs.prompt}\
{self.history.prompt}\
{self.constraints.prompt}\
{self.progress.prompt}\
{self.suggestion.prompt}\
"""
        )

        if self.flags.use_abstract_example:
            prompt.add_text(
                f"""
# Abstract Example

Here is an abstract version of the answer with description of the content of
each tag. Make sure you follow this structure, but replace the content with your
answer:
{self.constraints.abstract_ex}\
{self.progress.abstract_ex}\
{self.suggestion.abstract_ex}\
"""
            )

        if self.flags.use_concrete_example:
            prompt.add_text(
                f"""
# Concrete Example

Here is a concrete example of how to format your answer.
Make sure to follow the template with proper tags:
{self.constraints.concrete_ex}\
{self.progress.concrete_ex}\
{self.suggestion.concrete_ex}\
"""
            )
        return self.obs.add_screenshot(prompt)

    def shrink(self):
        self.history.shrink()
        self.obs.shrink()

    def _parse_answer(self, text_answer):
        ans_dict = {}
        ans_dict.update(self.constraints.parse_answer(text_answer))
        ans_dict.update(self.progress.parse_answer(text_answer))
        ans_dict.update(self.suggestion.parse_answer(text_answer))
        return ans_dict


class PlanningInstructions(dp.PromptElement):

    def __init__(self, goal_object, extra_instructions=None) -> None:
        super().__init__(visible=True)

        self._prompt = [
            dict(
                type="text",
                text="""
# Instructions

You are an excellent planner, very skilled at understanding how to interact with web pages.
Your goal is to define a node belonging to a tree of nodes structure, that represent the plan to accomplish the task that is being performed by a user, 
starting from a natural language description of the node.

There are three possible node types that can be used to construct the tree:
- AND Node: Represents an ordered list of logical subgoals required to achieve the node’s objective.
- OR Node: Represents alternative sub-strategies (which can be other AND/OR nodes).
- ACTION Node: Single executable action that can be performed on the webpage.


Your task for the given node:
1. Determine whether the node is an AND node, an OR node, or an ACTION node.
2. Choose ONE of the following options:
   A. Mark node as ACTION if the goal can be achieved using a single atomic action.
   B. Expand the node if the goal cannot be solved by performing a single atomic action.

## Important Rules:
- If you think the task can be completed through an action, mark the node as an action.
- If you need to expand the task in subtasks, focus on expansions that will complete the task faster with high probability.
- Before expanding the node, make sure to analyze the "Progress" and the "Suggestion" provided in the Hints section.
- DO NOT EXPAND THE NODE with subgoals already listed as "Completed Nodes" or "Not Yet Explored Nodes".
- For AND and OR nodes, provide the ordered list of logical subgoals.
- For AND and OR nodes, ensure temporal order of children is correct and efficient.
- Do not add speculative or redundant subgoals.

## Goal:
""",
            )
        ]

        self._prompt += goal_object

        if extra_instructions:
            self._prompt += [
                dict(
                    type="text",
                    text=f"""

## Extra instructions:

{extra_instructions}
""",
                )
            ]

    def _parse_answer(self, text_answer):
        try:
            return parse_html_tags_raise(text_answer, keys=["suggestion"], merge_multiple=True)
        except ParseError as e:
            return {"suggestion": text_answer, "parse_error": str(e)}


class PlanningHints(dp.PromptElement):

    def __init__(
        self,
        constraints: str,
        progress: str,
        suggestion: str,
        completed_nodes_description: str,
        not_yet_explored_nodes_description: str,
        visible: bool = True,
    ):
        super().__init__(visible=visible)
        completed_nodes_description = "\n".join(completed_nodes_description)
        not_yet_explored_nodes_description = "\n".join(not_yet_explored_nodes_description)
        self._prompt = f"""
# Hints

Here follows some information that can help you plan the task:
## Constraints:
{constraints}

## Progress: {progress}

## Suggestion: {suggestion}

## Completed Nodes:
{completed_nodes_description}

## Not Yet Explored Nodes:
{not_yet_explored_nodes_description}
"""


class NodeType(dp.PromptElement):
    _abstract_ex = """<node_type>Determine whether the node is an AND node, an OR node, or an ACTION node.</node_type>"""

    def __init__(self, node_type: str, visible: bool = True):
        super().__init__(visible=visible)
        self.node_type = node_type

        self._concrete_ex = f"""<node_type>{self.node_type}</node_type>"""

    def parse_answer(self, text_answer):
        try:
            return parse_html_tags_raise(text_answer, keys=["node_type"])
        except ParseError as e:
            return {"node_type": text_answer, "parse_error": str(e)}


class NodeDescription(dp.PromptElement):

    _abstract_ex = (
        """<node_description>Provide a textual description of the node.</node_description>"""
    )

    def __init__(self, description: str, visible: bool = True):
        super().__init__(visible=visible)
        self.description = description

    _concrete_ex = """<node_description>Click on the IPhone Pro 12 128GB Pacific Blue listing</node_description>"""

    def parse_answer(self, text_answer):
        try:
            return parse_html_tags_raise(text_answer, keys=["node_description"])
        except ParseError as e:
            return {"node_description": text_answer, "parse_error": str(e)}


class NodeExpansion(dp.PromptElement):
    _prompt = ""
    _abstract_ex = """<node_expansion>If it's necessary to expand the node, provide a list of subgoals or alternative strategies. Otherwise, provide an empty list.</node_expansion>"""

    def __init__(self, expansions: list[str], visible: bool = True):
        super().__init__(visible=visible)
        self.expansion = expansions

        self._concrete_ex = "\n".join(
            f"""<node_expansion>{item}</node_expansion>""" for item in expansions
        )

    def parse_answer(self, text_answer):
        try:
            content_dict, valid, retry_message = parse_html_tags(
                text_answer, keys=["node_expansion"], merge_multiple=True
            )
            if not valid:
                return {"node_expansion": [], "parse_error": retry_message}
            else:
                return {"node_expansion": content_dict["node_expansion"].split("\n")}
        except ParseError as e:
            return {"node_expansion": text_answer, "parse_error": str(e)}


class NodeReasoning(dp.PromptElement):
    _prompt = ""
    _abstract_ex = """<node_reasoning>Brief justification explaining the reasoning behind the node type and node expansion choices.</node_reasoning>"""

    def __init__(self, reasoning: str, visible: bool = True):
        super().__init__(visible=visible)
        self.reasoning = reasoning

        self._concrete_ex = f"""<node_reasoning>{self.reasoning}</node_reasoning>"""

    def parse_answer(self, text_answer):
        try:
            return parse_html_tags_raise(text_answer, keys=["node_reasoning"])
        except ParseError as e:
            return {"node_reasoning": text_answer, "parse_error": str(e)}


class Node(dp.PromptElement, abc.ABC):

    def __init__(
        self,
        type: str,
        description: str,
        expansion: list[str],
        reasoning: str,
        visible: bool = True,
    ):
        super().__init__(visible=visible)
        self.type = NodeType(type)
        self.description = NodeDescription(description)
        self.expansion = NodeExpansion(expansion)
        self.reasoning = NodeReasoning(reasoning)

        self._abstract_ex = f"""
{self.type._abstract_ex}
{self.description._abstract_ex}
{self.expansion._abstract_ex}
{self.reasoning._abstract_ex}
"""

        self._concrete_ex = (
            f"""
# Concrete {self.type.node_type} Example

Here is a concrete examples of how to format your answer if the node type is {self.type.node_type}.
Make sure to follow the template with proper tags:
{self.type._concrete_ex}
{self.description._concrete_ex}"""
            + (
                f"""
{self.expansion._concrete_ex}"""
                if self.expansion.expansion
                else ""
            )
            + f"""
{self.reasoning._concrete_ex}"""
        )

    def parse_answer(self, text_answer):
        try:
            ans_dict = {}
            ans_dict.update(self.type.parse_answer(text_answer))
            ans_dict.update(self.description.parse_answer(text_answer))
            ans_dict.update(self.expansion.parse_answer(text_answer))
            ans_dict.update(self.reasoning.parse_answer(text_answer))
            return ans_dict
        except ParseError as e:
            return {"node": text_answer, "parse_error": str(e)}


class ActionNode(Node):
    def __init__(self, description: str, reasoning: str, visible: bool = True):
        super().__init__(
            type="ACTION",
            description=description,
            expansion=[],
            reasoning=reasoning,
            visible=visible,
        )

    def parse_answer(self, text_answer):
        ans_dict = super().parse_answer(text_answer)

        if "node_type" in ans_dict and ans_dict["node_type"] != "ACTION":
            raise ParseError("Action node must be classified as ACTION")

        return ans_dict


class AndNode(Node):
    def __init__(
        self, description: str, expansion: list[str], reasoning: str, visible: bool = True
    ):
        super().__init__(
            type="AND",
            description=description,
            expansion=expansion,
            reasoning=reasoning,
            visible=visible,
        )

    def parse_answer(self, text_answer):
        ans_dict = super().parse_answer(text_answer)

        if "node_type" in ans_dict and ans_dict["node_type"] != "AND":
            raise ParseError("AND node must be classified as AND")

        if "node_expansion" in ans_dict and len(ans_dict["node_expansion"]) < 2:
            raise ParseError("AND node must have at least one expansion")

        return ans_dict


class OrNode(Node):
    def __init__(
        self, description: str, expansion: list[str], reasoning: str, visible: bool = True
    ):
        super().__init__(
            type="OR",
            description=description,
            expansion=expansion,
            reasoning=reasoning,
            visible=visible,
        )

    def parse_answer(self, text_answer):
        ans_dict = super().parse_answer(text_answer)

        if "node_type" in ans_dict and ans_dict["node_type"] != "OR":
            raise ParseError("OR node must be classified as OR")

        if "node_expansion" in ans_dict and len(ans_dict["node_expansion"]) < 2:
            raise ParseError("OR node must have at least one expansion")

        return ans_dict


class SystemPlanningPrompt(dp.PromptElement):
    _prompt = """\
You are an efficient logical (AND/OR) tree constructing agent helping another agent to plan how to solve a web task based on the content of the page and
user instructions. You dynamically construct AND/OR planning trees from the webpage’s accessibility tree structure for EFFICIENT and robust task execution.
Each time you provide a plan it will be used as a guide for planning the next action to take to accomplish the task.
You should be very concise and to the point, you should not provide any additional information that is not related to the task."""


class PlanningPrompt(dp.Shrinkable):
    def __init__(
        self,
        obs_history: list[dict],
        actions: list[str],
        memories: list[str],
        thoughts: list[str],
        constraints: str,
        progress: str,
        suggestion: str,
        node_info: str,
        actual_plan: (list[str], list[str]),
        flags: HPAPromptFlags,
    ):
        super().__init__()
        self.flags = flags
        self.node_info = node_info
        self.history = dp.History(obs_history, actions, memories, thoughts, flags.obs)
        self.instructions = PlanningInstructions(
            obs_history[-1]["goal_object"],
            extra_instructions=flags.extra_instructions,
        )
        self.hints = PlanningHints(
            constraints=constraints,
            progress=progress,
            suggestion=suggestion,
            completed_nodes_description=actual_plan[0],
            not_yet_explored_nodes_description=actual_plan[1],
        )
        self.action_node = ActionNode(
            description="Click on the IPhone Pro 12 128GB Pacific Blue listing",
            reasoning="The item is the best match for the requirements and the price is within the budget.",
        )
        self.and_node = AndNode(
            description="Navigate to the IPhone listings page and apply filters for Pacific Blue color, 128GB storage, and price under $400 before clicking the submit button",
            expansion=[
                "Click on the IPhone listings page",
                "Apply the filter for the Pacific Blue color",
                "Apply the filter for the 128GB storage",
                "Apply the filter for the price under $400",
                "Click submit button",
            ],
            reasoning="To reach the desired listing efficiently, start by accessing the main IPhone listings page, then narrow the results by applying filters for color, storage, and price, ensuring only relevant items are displayed before submitting the search.",
        )
        self.or_node = OrNode(
            description="Navigate to the IPhone listings page",
            expansion=[
                "Access the Apple sub-category",
                "Access the Phone sub-category",
            ],
            reasoning="There are multiple ways to reach the IPhone listings page: one is to go through the Apple sub-category, which directly relates to Apple devices including iPhones, and another is to use the Phone sub-category that may also lead to the product listings. Exploring either alternative enables reaching the target page and provides flexibility in navigation.",
        )

        self.obs = dp.Observation(
            obs_history[-1],
            self.flags.obs,
        )

    @property
    def _prompt(self) -> HumanMessage:
        prompt = HumanMessage(self.instructions.prompt)

        # {self.history.prompt}\

        prompt.add_text(
            f"""\
{self.obs.prompt}\
{self.hints.prompt}\

# NODE TO EXPAND:
{self.node_info}
"""
        )

        if self.flags.use_abstract_example:
            prompt.add_text(
                f"""
# Abstract Example

Here is an abstract version of the answer with description of the content of
each tag. Make sure you follow this structure, but replace the content with your
answer:
{self.action_node._abstract_ex}\
"""
            )

        if self.flags.use_concrete_example:
            prompt.add_text(
                f"""
{self.action_node._concrete_ex}
{self.and_node._concrete_ex}
{self.or_node._concrete_ex}
"""
            )

        return self.obs.add_screenshot(prompt)

    def shrink(self):
        self.history.shrink()
        self.obs.shrink()

    def _parse_answer(self, text_answer):
        ans_dict = {}

        tmp = parse_html_tags_raise(text_answer, keys=["node_type"])
        if "node_type" in tmp and tmp["node_type"] == "ACTION":
            ans_dict = self.action_node.parse_answer(text_answer)
        elif "node_type" in tmp and tmp["node_type"] == "AND":
            ans_dict = self.and_node.parse_answer(text_answer)
        elif "node_type" in tmp and tmp["node_type"] == "OR":
            ans_dict = self.or_node.parse_answer(text_answer)
        else:
            raise ParseError("Invalid node type")

        return ans_dict
