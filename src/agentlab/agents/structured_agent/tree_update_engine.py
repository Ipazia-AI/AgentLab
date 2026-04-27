import json

from bgym import AbstractActionSet

from agentlab.agents import dynamic_prompting as dp
from agentlab.agents.structured_agent.hpa_prompt import HPAPromptFlags
from agentlab.llm.llm_utils import (
    Discussion,
    HumanMessage,
    ParseError,
    SystemMessage,
    retry,
)

from .andor_tree import Node, NodeStatus, NodeType
from .prompts import GlobalTreeUpdatePrompt


class TreeUpdateEngine:
    def __init__(self, chat_llm, action_set: AbstractActionSet, flags: HPAPromptFlags):
        self.chat_llm = chat_llm
        self.action_set = action_set
        self.flags = flags

    def apply(
        self,
        *,
        model_name: str,
        task_description: str,
        constraints: str,
        progress: str,
        suggestion: str,
        obs_history: list[dict],
        pending_node: Node,
        stack: list[tuple[Node, object]],
    ) -> dict:
        """
        Updates the global tree based on the task description, task constraints, observation and other information.

        Args:
            task_description: string containing the task description.
            task_constraints: list of strings of the task constraints.
            observation: axtree of the current webpage.

        Returns:
            None
        """
        global_tree = self._get_global_tree(stack)
        global_tree_info = "\n\n".join(
            [
                f"NODE ID: {node.id}\nNODE TYPE: {node.type.name}\nNODE STATUS: {node.status.name}\nNODE DESCRIPTION: {node.description}\nNODE ACTION: {node.action}"
                for node in global_tree
            ]
        )
        system_message = GlobalTreeUpdatePrompt(self.action_set, self.flags.action).system_message()
        user_message = GlobalTreeUpdatePrompt.user_prompt(
            task_description=task_description,
            node_id=pending_node.id,
            constraints=constraints,
            progress=progress,
            suggestion=suggestion,
            observation=obs_history[-1].get("axtree_txt", ""),
            global_tree_info=global_tree_info,
        )
        result = self._call_json_prompt(model_name, system_message, user_message)
        pruned_node_ids = result.get("prune", [])
        # updated_node_ids = result.get("update", {})
        print(f"\n{'='*60}\nTree Update\n{'='*60}\n")
        self._prune_nodes_from_global_tree(pruned_node_ids=pruned_node_ids, global_tree=global_tree)
        # self._update_nodes_in_global_tree(
        #     updated_node_ids=updated_node_ids, global_tree=global_tree
        # )
        print(f"\n{'='*60}")
        return result

    def _get_global_tree(self, stack_items: list[tuple[Node, object]]) -> list[Node]:
        """
        Returns the global tree as a list of nodes.

        Args:
            stack_items: list of tuples containing the node and the state of the node.

        Returns:
            list[Node]: List of nodes in the global tree.
        """

        def walk_tree(node: Node, nodes_list: list[Node]):
            if node.status == NodeStatus.DELETED:
                return
            nodes_list.append(node)
            for child in node.children:
                walk_tree(child, nodes_list)
            return nodes_list

        if not stack_items:
            return []
        return walk_tree(stack_items[0][0], [])

    def _get_nodes_from_ids(self, node_ids: list[str], global_tree: list[Node]) -> list[Node]:
        """
        Returns the nodes from the global tree that have specified ids.

        Args:
            node_ids: list of node ids to get from the global tree.
            global_tree: list of nodes in the global tree.

        Returns:
            list[Node]: List of nodes from the global tree that have specified ids.
        """
        return [node for node in global_tree if node.id in node_ids]

    def _is_protected_from_pruning(self, node: Node) -> bool:
        """A node is protected if it already succeeded or is an untried
        alternative under an OR parent (the tree logic should decide its fate,
        not the global-update LLM)."""
        if node.status == NodeStatus.SUCCESS:
            return True
        if node.status == NodeStatus.VISITED:
            return True
        if (
            node.parent is not None
            and node.parent.type == NodeType.OR
            and node.status in {NodeStatus.UNVISITED, NodeStatus.VISITED}
        ):
            return True
        return False

    def _prune_nodes_from_global_tree(
        self, pruned_node_ids: list[str], global_tree: list[Node]
    ) -> None:
        """
        Sets the status of the nodes with specified ids to DELETED,
        unless the node is protected (SUCCESS or untried OR alternative).
        """
        nodes_to_prune = self._get_nodes_from_ids(node_ids=pruned_node_ids, global_tree=global_tree)
        for node in nodes_to_prune:
            if self._is_protected_from_pruning(node):
                continue
            node.status = NodeStatus.DELETED
            print(f"[PRUNED] Pruned node {node.id}")
        return

    def _update_nodes_in_global_tree(
        self, updated_node_ids: dict[str, str], global_tree: list[Node]
    ) -> None:
        """
        Updates the description of the nodes with specified ids.

        Args:
            updated_node_ids: dictionary of node ids to update and their new descriptions.
            global_tree: list of nodes in the global tree.

        Returns:
            None
        """
        nodes_to_update = self._get_nodes_from_ids(
            node_ids=list(updated_node_ids.keys()), global_tree=global_tree
        )
        for node in nodes_to_update:
            if node.status != NodeStatus.UNVISITED:
                print(
                    f"[UPDATE BLOCKED] Refusing to update node {node.id} (status={node.status.name})"
                )
                continue
            node.description = updated_node_ids[node.id]
            print(f"[UPDATED] Updated node {node.id} with description: {node.description}")
        return

    def _call_json_prompt(
        self, model_name: str, system_message: str, user_message: str | dp.Shrinkable
    ) -> dict:
        if isinstance(user_message, dp.Shrinkable):
            if self.flags.max_prompt_tokens is not None:
                user_message = dp.fit_tokens(
                    shrinkable=user_message,
                    max_prompt_tokens=self.flags.max_prompt_tokens,
                    model_name=model_name,
                    max_iterations=self.flags.max_trunc_itr,
                    additional_prompts=system_message,
                )
            else:
                user_message = user_message.prompt
        messages = Discussion([SystemMessage(system_message), HumanMessage(user_message)])

        def parser(text_answer: str) -> dict:
            candidate = text_answer.strip()
            start = candidate.find("{")
            end = candidate.rfind("}")
            if start != -1 and end != -1 and end > start:
                candidate = candidate[start : end + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError as exc:
                raise ParseError(f"Return valid JSON only. Error: {exc}") from exc

        return retry(self.chat_llm, messages, n_retry=2, parser=parser)
