import json
import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)

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
        max_depth: int = 3,
    ):
        self.chat_llm = chat_llm
        self.action_set = action_set
        self.flags = flags
        self.budget = budget
        self.max_depth = max_depth
        self.stack: List[Tuple[Node, NodeState]] = []
        self.task_constraints: list[str] = []
        self.task_progress_summary: str | None = None
        self.previous_notes: str | None = None
        self.pending_node: Node | None = None
        self.completed_nodes: list[Node] = []
        self.previous_attempt_summaries: list[str] = []
        self._counter: int = 0
        self.retries: int = 0
        self.max_retries: int = 3

    def get_plan(self) -> (list[str], list[str]):
        completed_plan = [node.prompt_description for node in self.completed_nodes]
        pending_plan = [
            node.prompt_description
            for node, state in reversed(self.stack)
            if node.id != "0"
            and state != NodeState.EXITING
            and (node.status == NodeStatus.UNVISITED or node.type == NodeType.UNKNOWN)
        ]

        return completed_plan, pending_plan

    def get_tree_context(self, expanding_node: Node) -> str:
        """Build a tree view of the plan from root, expanding only the ancestor path.

        Nodes along the path from root to ``expanding_node`` have their children
        shown (with status).  Other nodes appear as single lines so the LLM sees
        the full structure without irrelevant subtree detail.
        """
        ancestor_ids: set[str] = set()
        current = expanding_node
        while current is not None:
            ancestor_ids.add(current.id)
            current = current.parent

        root = expanding_node
        while root.parent is not None:
            root = root.parent

        lines: list[str] = []

        def _render(node: Node, depth: int, show_deleted_as_failed: bool = False) -> None:
            if node.status == NodeStatus.DELETED:
                if show_deleted_as_failed:
                    indent = "  " * depth
                    lines.append(f"{indent}[FAILED] {node.id}: {node.description}")
                return

            indent = "  " * depth

            status_prefix = ""
            if node.status == NodeStatus.SUCCESS:
                status_prefix = "[SUCCESS] "
            elif node.status == NodeStatus.FAIL:
                err = f": {node.action_error}" if node.action_error else ""
                status_prefix = f"[FAIL{err}] "
            elif node.status == NodeStatus.PRUNED:
                status_prefix = "[PRUNED] "
            elif node.status == NodeStatus.VISITED:
                status_prefix = "[VISITED] "

            type_label = ""
            if node.type in {NodeType.AND, NodeType.OR}:
                type_label = f" ({node.type.name})"

            marker = ""
            if node.id == expanding_node.id:
                marker = "  ← EXPAND THIS NODE"

            lines.append(
                f"{indent}{status_prefix}{node.id}{type_label}: {node.description}{marker}"
            )

            if node.id in ancestor_ids:
                is_or = node.type == NodeType.OR
                for child in node.children:
                    _render(child, depth + 1, show_deleted_as_failed=is_or)

        _render(root, 0)
        result = "\n".join(lines)

        if self.previous_attempt_summaries:
            prev = "\n\n".join(
                f"### Attempt {i + 1}:\n{summary}"
                for i, summary in enumerate(self.previous_attempt_summaries)
            )
            result += f"\n\n## Previous Failed Attempts:\n{prev}"

        return result

    def _save_attempt_summary(self) -> None:
        """Capture a full tree snapshot before a retry reset."""
        root = self.stack[0][0] if self.stack else None
        if root is None:
            return

        lines: list[str] = []

        def _walk(node: Node, depth: int) -> None:
            indent = "  " * depth
            status = node.status.name
            lines.append(f"{indent}[{status}] {node.id}: {node.description}")
            for child in node.children:
                _walk(child, depth + 1)

        _walk(root, 0)
        self.previous_attempt_summaries.append("\n".join(lines))

    def set_goal(self, obs_first: dict):
        self.goal = obs_first["goal"]
        self.stack = [(Node(type=NodeType.UNKNOWN, description=self.goal), NodeState.ENTERING)]

    def get_action_node(self, expansion_function) -> Node | None:
        while self.retries < self.max_retries:
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
            if self.stack or self.completed_nodes:
                self._save_attempt_summary()
            self.retries += 1
            #self.completed_nodes = []
            self.stack = [(Node(type=NodeType.UNKNOWN, description=self.goal), NodeState.ENTERING)]
        

        return None

    def complete_pending(
        self,
        model_name: str,
        constraints: str,
        progress: str,
        suggestion: str,
        obs_history: list[dict],
    ):
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
            constraints=constraints,
            progress=progress,
            suggestion=suggestion,
            obs_history=obs_history,
        )

    # Algo 2 from the HPA paper
    def _process_node_entering(self, node: Node, expansion_function) -> Node:
        node.execution_count += 1

        if node.type == NodeType.UNKNOWN:
            tree_context = self.get_tree_context(node)
            print(f"\n{'='*60}\nExpanding node {node.id}\n{'='*60}\n{tree_context}\n{'='*60}\n")
            expansion_function(node, tree_context)

        if node.type == NodeType.ACTION:
            self.stack.append((node, NodeState.EXITING))
            return node

        if node.type == NodeType.AND:
            if self._has_successful_children_and(node):
                return node
            self.stack.append((node, NodeState.EXITING))
            if self._has_valid_children_and(node):
                for child in reversed(node.children):
                    if child.status not in self._closed_statuses():
                        self.stack.append((child, NodeState.ENTERING))
            else:
                node.status = NodeStatus.FAIL

        elif node.type == NodeType.OR:
            if self._has_successful_children_or(node):
                return node
            self.stack.append((node, NodeState.EXITING))
            if self._has_valid_children_or(node):
                child = self._select_promising_child(node)
                self.stack.append((child, NodeState.ENTERING))
            else:
                node.status = NodeStatus.FAIL

        return node

    # Algo 3 from the HPA paper
    def _process_node_exiting(self, node: Node):
        if node.type == NodeType.ACTION:
            self.completed_nodes.append(node)
            if node.status in {NodeStatus.FAIL, NodeStatus.PRUNED}:
                self.stack.append((node, NodeState.FAILED))

        elif node.type == NodeType.AND:
            if self._has_successful_children_and(node) and self._check_and_complete(node):
                node.status = NodeStatus.SUCCESS
                return
            node.status = NodeStatus.FAIL
            self.stack.append((node, NodeState.FAILED))

        elif node.type == NodeType.OR:
            if self._has_successful_children_or(node):
                node.status = NodeStatus.SUCCESS
                return
            node.status = NodeStatus.FAIL
            self.stack.append((node, NodeState.FAILED))

    # Algo 4 from the HPA paper
    def _process_node_failed(self, node: Node):
        if node.type == NodeType.ACTION:
            node.status = NodeStatus.PRUNED
            self._propagate_failure(node)
            return

        elif node.type == NodeType.AND:
            if self._has_successful_children_and(node) and self._check_and_complete(node):
                node.status = NodeStatus.SUCCESS
                return
            if not self._get_valid_children(node):
                node.status = NodeStatus.PRUNED
                self._propagate_failure(node)

        elif node.type == NodeType.OR:
            if self._has_valid_children_or(node):
                node.status = NodeStatus.VISITED
                self.stack.append((node, NodeState.ENTERING))
                return
            node.status = NodeStatus.PRUNED
            self._propagate_failure(node)

    def _select_promising_child(self, node: Node) -> Node:
        valid_children = self._get_valid_children(node)
        return max(valid_children, key=lambda child: child.score or 0.0)

    def _get_global_tree(self) -> list[Node]:
        """
        Returns the global tree as a list of nodes.

        Args:
            None

        Returns:
            List[Node]: List of nodes in the global tree.
        """
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

    def _is_protected_from_pruning(self, node: Node) -> bool:
        """A node is protected if it already succeeded or is an untried
        alternative under an OR parent (the tree logic should decide its fate,
        not the global-update LLM)."""
        if node.status == NodeStatus.SUCCESS:
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
                print(
                    f"[PRUNE BLOCKED] Refusing to prune protected node {node.id} "
                    f"(status={node.status.name}, parent_type="
                    f"{node.parent.type.name if node.parent else 'ROOT'})"
                )
                continue
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
        constraints: str,
        progress: str,
        suggestion: str,
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
            constraints=constraints,
            progress=progress,
            suggestion=suggestion,
            observation=obs_history[-1].get("axtree_txt", ""),
            global_tree_info=global_tree_info,
        )
        result = self._call_json_prompt(model_name, system_message, user_message)
        pruned_node_ids = result.get("prune", [])
        updated_node_ids = result.get("update", {})
        self._prune_nodes_from_global_tree(pruned_node_ids=pruned_node_ids, global_tree=global_tree)
        self._update_nodes_in_global_tree(
            updated_node_ids=updated_node_ids, global_tree=global_tree
        )
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

    def _closed_statuses(self):
        return {NodeStatus.SUCCESS, NodeStatus.DELETED, NodeStatus.PRUNED}

    def _has_successful_children_and(self, node: Node) -> bool:
        return all(
            c.status == NodeStatus.SUCCESS or c.status == NodeStatus.DELETED for c in node.children
        )

    def _has_successful_children_or(self, node: Node) -> bool:
        return any(c.status == NodeStatus.SUCCESS for c in node.children)

    def _has_valid_children_and(self, node: Node) -> bool:
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
