import json
import logging
from typing import List, Tuple

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

from .andor_tree import Node, NodeState, NodeStatus, NodeType
from .prompts import GlobalTreeUpdatePrompt


class HPA:
    def __init__(
        self,
        chat_llm,
        action_set: AbstractActionSet,
        flags: HPAPromptFlags,
        budget: int = 1000,
        max_revision_count: int = 3,
        max_depth: int = 3,
    ):
        self.chat_llm = chat_llm
        self.action_set = action_set
        self.flags = flags
        self.budget = budget  # remove
        self.max_revision_count = max_revision_count
        self.max_depth = max_depth
        self.stack: List[Tuple[Node, NodeState]] = []
        self.task_constraints: list[str] = []
        self.task_progress_summary: str | None = None
        self.previous_notes: str | None = None
        self.pending_node: Node | None = None
        self.completed_nodes: list[Node] = []
        self._counter: int = 0

    def get_plan(self) -> (list[str], list[str]):
        completed_plan = [node.description for node in self.completed_nodes]
        pending_plan = [
            node.description
            for node, state in reversed(self.stack)
            if node.id != "0"
            and state != NodeState.EXITING
            and (node.status == NodeStatus.UNVISITED or node.type == NodeType.UNKNOWN)
        ]

        return completed_plan, pending_plan

    def set_goal(self, obs_first: dict):
        self.goal = obs_first["goal"]
        self.stack = [(Node(type=NodeType.UNKNOWN, description=self.goal), NodeState.ENTERING)]

    def get_action_node(self, expansion_function) -> Node | None:
        while self.stack:
            node, state = self.stack.pop()

            if node.status == NodeStatus.PRUNED:
                self._propagate_failure(node)
                continue

            if node.status == NodeStatus.DELETED:
                continue

            if state == NodeState.ENTERING:
                self.pending_node = self._process_node_entering(node, expansion_function)
                if self.pending_node.type == NodeType.ACTION:
                    return self.pending_node

            elif state == NodeState.EXITING:
                self._process_node_exiting(node)

            elif state == NodeState.FAILED:
                self._process_node_failed(node)

            self._counter += 1
            if self._counter >= self.budget:
                break

        return None

    def complete_pending(self, model_name: str, obs_history: list[dict]):
        action_error = obs_history[-1].get("last_action_error", "")
        success = True if action_error == "" else False
        if success:
            self.pending_node.status = NodeStatus.SUCCESS
        else:
            self.pending_node.status = NodeStatus.FAIL
            self.pending_node.action_error = action_error

        self._global_tree_update(
            model_name=model_name,
            task_description=self.goal,
            task_constraints=self.task_constraints,
            obs_history=obs_history,
        )

    # Algo 2 from the HPA paper
    def _process_node_entering(self, node: Node, expansion_function) -> Node:
        if node.parent and node.parent.type == NodeType.OR:
            # TODO: The role of this function is not clear, let's check it later what it is supposed to do.
            self._rollback_context(node.parent)

        # TODO: The role of this function is not clear, let's check it later what it is supposed to do.
        self._set_context(node)
        node.execution_count += 1

        if node.type == NodeType.UNKNOWN:
            expansion_function(
                node.description, self.get_plan(), lambda x: self._apply_expansion(node, x)
            )
            self.stack.append((node, NodeState.EXITING))

        if node.type == NodeType.ACTION:
            return node

        if node.type == NodeType.AND:
            if self._has_successful_children_and(node):
                return node
            if self._has_valid_children_and(node):
                for child in reversed(node.children):
                    # If node is in state UNVISITED, VISITED or FAIL
                    if child.status not in self._closed_statuses():
                        self.stack.append((child, NodeState.ENTERING))
            else:
                node.status = NodeStatus.FAIL

        if node.type == NodeType.OR:
            if self._has_successful_children_or(node):
                return node
            if self._has_valid_children_or(node):
                child = self._select_promising_child(node)
                self.stack.append((child, NodeState.ENTERING))
            else:
                node.status = NodeStatus.FAIL
        # return None
        return node

    # Algo 3 from the HPA paper
    def _process_node_exiting(self, node: Node):
        if node.type == NodeType.ACTION:
            self.completed_nodes.append(node)
            if node.status in {NodeStatus.FAIL, NodeStatus.PRUNED}:
                self.stack.append((node, NodeState.FAILED))
                return
        else:
            node.status = NodeStatus.SUCCESS

        if node.type == NodeType.AND:
            if self._has_successful_children_and(node):
                if self._check_and_complete(node):
                    node.status = NodeStatus.SUCCESS
                    return

            node.status = NodeStatus.FAIL
            self.stack.append((node, NodeState.FAILED))

        if node.type == NodeType.OR:
            if self._has_successful_children_or(node):
                node.status = NodeStatus.SUCCESS
                return

            node.status = NodeStatus.FAIL
            self.stack.append((node, NodeState.FAILED))

    # Algo 4 from the HPA paper
    def _process_node_failed(self, node: Node):
        if node.type == NodeType.ACTION:
            node.status = NodeStatus.PRUNED
            # If a failed ACTION node belongs to an
            # AND node, the agent deletes all remaining unexecuted siblings (This is what the _propagate_failure function does)
            self._propagate_failure(node)
            return

        elif node.type == NodeType.AND:
            if self._has_successful_children_and(node):
                is_complete = self._check_and_complete(node)
                if is_complete:
                    node.status = NodeStatus.SUCCESS
            is_valid = self._has_valid_children_and(node)
            if not is_valid:
                if node.revision_count < self.max_revision_count:
                    revised = self._revise_and(node)
                    if node.status == NodeStatus.SUCCESS:
                        return
                    elif revised or is_valid:
                        node.status = NodeStatus.VISITED
                        self.stack.append((node, NodeState.ENTERING))
                    else:
                        node.status = NodeStatus.PRUNED
                        self._propagate_failure(node)

        elif node.type == NodeType.OR:
            if self._has_valid_children_or(node):
                node.status = NodeStatus.VISITED
                self.stack.append((node, NodeState.ENTERING))
                return

            if node.revision_count < self.max_revision_count:
                revised = self._revise_or(node)
                if revised:
                    node.status = NodeStatus.VISITED
                    self.stack.append((node, NodeState.ENTERING))
                else:
                    node.status = NodeStatus.PRUNED
                    self._propagate_failure(node)

    def _apply_expansion(self, node: Node, expansion: dict) -> NodeType:
        node_type = str(expansion.get("node_type", "")).upper().strip()

        # Safety net: if the node is at max depth, override any AND/OR to ACTION.
        if node.depth >= self.max_depth and node_type in {"AND", "OR"}:
            logging.warning(
                "Node %s at depth %d reached max_depth=%d but LLM returned %s. "
                "Forcing ACTION from the first child description.",
                node.id,
                node.depth,
                self.max_depth,
                node_type,
            )
            children = expansion.get("expansion", [])
            # Best-effort: use the first child description as the action text.
            fallback = children[0] if isinstance(children, list) and children else "noop"
            raise ParseError(
                f"Maximum tree depth ({self.max_depth}) reached. "
                f"You MUST return an ACTION node with a single atomic browser action, "
                f"not a {node_type} node. The goal to achieve in a single action: {fallback}"
            )

        if node_type == "ACTION":
            node.type = NodeType.ACTION
            return

        if node_type not in {"AND", "OR"}:
            raise ParseError("node_type must be ACTION, AND, or OR.")

        node.type = NodeType.AND if node_type == "AND" else NodeType.OR

        node.description = expansion.get("node_description", node.description)

        children = expansion.get("node_expansion", [])
        if not isinstance(children, list) or not children:
            raise ParseError("AND/OR node requires a non-empty 'expansion' list.")

        node.children = []
        for child_text in children:
            if not isinstance(child_text, str):
                continue
            score = None
            clean_text = child_text.strip()
            # if node_type == "OR":
            #     score, clean_text = self._extract_score(clean_text)
            child = Node(type=NodeType.UNKNOWN, description=clean_text, parent=node, score=score)
            node.add_child(child)

    # def _extract_score(self, text: str) -> tuple[float | None, str]:
    #     if "(score:" not in text:
    #         return None, text
    #     try:
    #         before, after = text.split("(score:", 1)
    #         score_part = after.split(")", 1)[0]
    #         score = float(score_part.strip())
    #         cleaned = (before + after.split(")", 1)[1]).strip()
    #         return score, cleaned
    #     except (ValueError, IndexError):
    #         return None, text

    def _select_promising_child(self, node: Node) -> Node:
        valid_children = self._get_valid_children(node)
        return max(valid_children, key=lambda child: child.score or 0.0)

    def _describe_local_tree(self, node: Node) -> str:
        parts: list[str] = []
        if node.parent is None:
            return "root_node"
        parts.append(f"PARENT: {str(node.parent)}")
        if node.parent.children:
            siblings = [str(child) for child in node.parent.children if child is not node]
            if siblings:
                parts.append("SIBLINGS:\n" + "\n".join(siblings))
        if node.parent.parent and node.parent.parent.children:
            parent_siblings = [
                str(child) for child in node.parent.parent.children if child is not node.parent
            ]
            if parent_siblings:
                parts.append("PARENT SIBLINGS:\n" + "\n".join(parent_siblings))
        return "\n".join(parts)

    def _rollback_context(self, node: Node):
        pass

    def _set_context(self, node: Node):
        pass

    def _get_global_tree(self) -> list[Node]:
        """
        Returns the global tree as a list of nodes.

        Args:
            None

        Returns:
            List[Node]: List of nodes in the global tree.
        """
        if not getattr(self, "root_node", None):
            return []

        def walk_tree(node: Node, nodes_list: list[Node]):
            if node.status == NodeStatus.DELETED:
                return
            nodes_list.append(node)
            for child in node.children:
                walk_tree(child, nodes_list)
            return nodes_list

        global_tree = walk_tree(self.stack[0][0], [])
        return global_tree

    def _get_nodes_from_ids(self, node_ids: list[str], global_tree: list[Node]) -> List[Node]:
        """
        Returns the nodes from the global tree that have specified ids.

        Args:
            node_ids: list of node ids to get from the global tree.
            global_tree: list of nodes in the global tree.

        Returns:
            List[Node]: List of nodes from the global tree that have specified ids.
        """
        return [node for node in global_tree if node.id in node_ids]

    def _prune_nodes_from_global_tree(
        self, pruned_node_ids: list[str], global_tree: list[Node]
    ) -> None:
        """
        Sets the status of the nodes with specified ids to PRUNED.

        Args:
            pruned_node_ids: list of node ids to prune from the global tree.
            global_tree: list of nodes in the global tree.

        Returns:
            None
        """
        nodes_to_prune = self._get_nodes_from_ids(node_ids=pruned_node_ids, global_tree=global_tree)
        for node in nodes_to_prune:
            # node.status = NodeStatus.PRUNED
            node.status = NodeStatus.DELETED
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
            node.description = updated_node_ids[node.id]
        return

    def _global_tree_update(
        self,
        model_name: str,
        task_description: str,
        task_constraints: list[str],
        obs_history: list[dict],
    ) -> None:
        """
        Updates the global tree based on the task description, task constraints, observation and other information.

        Args:
            task_description: string containing the task description.
            task_constraints: list of strings of the task constraints.
            observation: axtree of the current webpage.

        Returns:
            None
        """
        global_tree = self._get_global_tree()
        node = self.pending_node
        global_tree_info = "\n\n".join(
            [
                f"NODE ID: {node.id}\nNODE TYPE: {node.type.name}\nNODE STATUS: {node.status.name}\nNODE DESCRIPTION: {node.description}\nNODE ACTION: {node.action}"
                for node in global_tree
            ]
        )
        system_message = GlobalTreeUpdatePrompt(self.action_set, self.flags.action).system_message()
        user_message = GlobalTreeUpdatePrompt.user_prompt(
            task_description=task_description,
            node_id=self.pending_node.id,
            task_constraints=task_constraints or None,
            task_progress_summary=self.task_progress_summary,
            notes_summary=self.previous_notes,
            observation=obs_history[-1].get("axtree_txt", ""),
            global_tree_info=self.global_tree,  # TODO: Implement the global_tree attribute: a list of nodes ordered by their ids
        )
        result = self._call_json_prompt(model_name, system_message, user_message)
        pruned_nodes = result.get("prune", [])
        updated_nodes = result.get("update", {})
        # TODO: prune and update the nodes in the global_tree
        return

    def _propagate_failure(self, node: Node):
        parent = node.parent
        if parent is None:
            return

        def mark_deleted_subtree(n: Node, deleted_ids: set):
            if n.status == NodeStatus.DELETED:
                deleted_ids.add(n.id)
            else:
                n.status = NodeStatus.DELETED
                deleted_ids.add(n.id)
            for c in getattr(n, "children", []) or []:
                mark_deleted_subtree(c, deleted_ids)

        deleted_ids: set = set()
        # If failure happened inside an ordered AND plan, short-circuit the remaining siblings.
        if parent.type == NodeType.AND and parent.children:
            for sibling in parent.children:
                mark_deleted_subtree(sibling, deleted_ids)

            # Ensure the AND node is treated as failed (it may later be repaired/pruned).
            if parent.status not in {NodeStatus.PRUNED, NodeStatus.DELETED}:
                parent.status = NodeStatus.FAIL

        elif parent.type == NodeType.OR:
            node.status = NodeStatus.DELETED
            deleted_ids.add(node.id)
            mark_deleted_subtree(node, deleted_ids)

        else:
            raise ValueError(f"Unexpected node type: {node.type}")

        if deleted_ids:
            self.stack = [(n, st) for (n, st) in self.stack if n.id not in deleted_ids]

    # def _synchronize_stack(self):
    #     pass

    def _revise_and(self, node: Node) -> bool:
        # The revision should probably:
        # 1. Remove all not valid children
        # 2. Remove the not valid children from the stack
        # 3. Create all the new children nodes
        node.revision_count += 1
        return False

    def _revise_or(self, node: Node) -> bool:
        # The revision should probably:
        # 1. Remove all not valid children
        # 2. Remove the not valid children from the stack
        # 3. Create all the new children nodes
        node.revision_count += 1
        return False

    def _closed_statuses(self):
        return {NodeStatus.SUCCESS, NodeStatus.DELETED, NodeStatus.PRUNED}

    def _has_successful_children_and(self, node: Node) -> bool:
        # return all(c.status == NodeStatus.SUCCESS for c in node.children)
        return all(
            c.status == NodeStatus.SUCCESS or c.status == NodeStatus.DELETED for c in node.children
        )

    def _has_successful_children_or(self, node: Node) -> bool:
        return any(c.status == NodeStatus.SUCCESS for c in node.children)

    def _has_valid_children_and(self, node: Node) -> bool:
        # return all(c.status not in {NodeStatus.PRUNED, NodeStatus.DELETED} for c in node.children)
        return all(c.status not in {NodeStatus.PRUNED} for c in node.children)

    def _has_valid_children_or(self, node: Node) -> bool:
        return any(c.status not in {NodeStatus.PRUNED, NodeStatus.DELETED} for c in node.children)

    def _check_and_complete(self, node: Node) -> bool:
        valid_children = self._get_valid_children(node)
        if not valid_children:
            return False
        return all(c.status == NodeStatus.SUCCESS for c in valid_children)

    def _get_valid_children(self, node: Node) -> list[Node]:
        return [c for c in node.children if c.status not in {NodeStatus.PRUNED, NodeStatus.DELETED}]

    def _is_root(self, node: Node) -> bool:
        return node.parent is None

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


def main():
    root = Node(type=NodeType.UNKNOWN, description="Solve task")
    agent = HPA(chat_llm=None, action_set=None)
    agent.reset(root_node=root, goal="Complete the task")
    next_action = agent.run_until_action()
    if next_action is not None:
        logging.info("Next action: %s", next_action.text)


if __name__ == "__main__":
    main()
