import json
import logging
from typing import List, Tuple

from bgym import AbstractActionSet

from agentlab.agents import dynamic_prompting as dp
from agentlab.agents.dynamic_prompting import ActionFlags
from agentlab.llm.llm_utils import (
    Discussion,
    HumanMessage,
    ParseError,
    SystemMessage,
    retry,
)

from .andor_tree import Node, NodeState, NodeStatus, NodeType
from .prompts import (
    GlobalTreeUpdatePrompt,
    NodeExpansionPrompt,
    NotesSummaryPrompt,
    ObservationSummaryPrompt,
    TaskConstraintsPrompt,
)


class HPA:
    def __init__(
        self,
        chat_llm,
        action_flags: ActionFlags,
        action_set: AbstractActionSet,
        budget: int = 1000,
        max_revision_count: int = 3,
        max_prompt_tokens: int | None = None,
        max_trunc_itr: int = 20,
    ):
        self.chat_llm = chat_llm
        self.action_flags = action_flags
        self.action_set = action_set
        self.budget = budget  # remove
        self.max_revision_count = max_revision_count
        self.max_prompt_tokens = max_prompt_tokens
        self.max_trunc_itr = max_trunc_itr
        self.root_node: Node = Node(type=NodeType.UNKNOWN, description="")
        self.stack: List[Tuple[Node, NodeState]] = []
        self._counter = 0
        self.task_constraints: list[str] = []
        self.task_progress_summary: str | None = None
        self.notes_summary: str | None = None
        self.observation_history: list[str] = []
        self.action_history: list[str] = []
        self.task_feedback: str | None = None

    def reset(self):
        self._counter = 0
        self.root_node = Node(type=NodeType.UNKNOWN, description="")
        self.stack = []
        self.task_constraints = []
        self.task_progress_summary = None
        self.notes_summary = None
        self.observation_history = []
        self.action_history = []
        self.task_feedback = None

    def run_until_action(self, goal: str | None, obs: dict) -> Node | None:

        if self._counter == 0:
            self.root_node.description = goal
            self.stack = [(self.root_node, NodeState.ENTERING)]

        while self.stack:
            node, state = self.stack.pop()

            if node.status == NodeStatus.PRUNED:
                self._propagate_failure(node)
                continue

            if node.status == NodeStatus.DELETED:
                continue

            if state == NodeState.ENTERING:
                processed_node = self._process_node_entering(node, obs)
                if processed_node.type == NodeType.ACTION:
                    return processed_node

            elif state == NodeState.EXITING:
                self._process_node_exiting(node)

            elif state == NodeState.FAILED:
                self._process_node_failed(node)

            self._counter += 1
            if self._counter >= self.budget:
                break

        return None

    def finalize_action(self, node: Node, obs: dict):
        action_error = obs.get("last_action_error")
        success = True if action_error == "" else False
        if success:
            node.status = NodeStatus.SUCCESS
        else:
            node.status = NodeStatus.FAIL
            node.action_error = action_error

        # self._global_tree_update(task_description=self.root_node.description, task_constraints=self.task_constraints, observation=obs.get("axtree_txt", ""))
        self._update_observations(node, obs)

    # Algo 2 from the HPA paper
    def _process_node_entering(self, node: Node, obs: dict) -> Node:
        if node.parent and node.parent.type == NodeType.OR:
            # TODO: The role of this function is not clear, let's check it later what it is supposed to do.
            self._rollback_context(node.parent)

        # TODO: The role of this function is not clear, let's check it later what it is supposed to do.
        self._set_context(node)
        node.execution_count += 1

        if node.type == NodeType.UNKNOWN:
            self._expand_node(node, obs)
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
            #  If a failed ACTION node belongs to an
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

    def _expand_node(self, node: Node, obs: dict) -> Node:
        if self.chat_llm is None:
            node.description = "report_infeasible"
            node.metadata["node_description"] = "LLM not available"
            node.type = NodeType.ACTION
            return node

        task_description = node.description or "Complete the task."
        observation = obs.get("axtree_txt") or obs.get("dom_txt") or obs.get("pruned_html") or ""

        if not self.task_constraints:
            task_constraints = self._infer_task_constraints(task_description, observation)
        else:
            task_constraints = self.task_constraints
        obs_summary = self._infer_observation_summary(
            task_description=task_description,
            task_constraints=task_constraints,
            observation=observation,
        )
        notes_summary = self._infer_notes_summary(
            task_description=task_description,
            task_constraints=task_constraints,
            observation=observation,
        )
        # TODO: Check if we need to add obs and notes to the inference
        expansion = self._infer_node_expansion(
            task_description=task_description,
            task_constraints=task_constraints,
            observation=observation,
            node=node,
        )

        node.metadata.update(
            {
                "task_constraints": task_constraints,
                "observation_summary": obs_summary.get("observation_summary"),
                "observation_highlights": obs_summary.get("observation_highlights"),
                "task_progress": obs_summary.get("task_progress"),
                "task_feedback": obs_summary.get("task_feedback"),
                "notes_summary": notes_summary.get("new_notes"),
            }
        )

        node.type = self._apply_expansion(node, expansion)
        return node

    def _infer_task_constraints(self, task_description: str, observation: str | None = None) -> list[str]:
        system_message = TaskConstraintsPrompt.system_message
        user_message = TaskConstraintsPrompt.user_prompt(
            task_objective=task_description,
            current_observation=observation,
        )
        result = self._call_json_prompt(system_message, user_message)
        constraints = result.get("task_constraints", [])
        if not isinstance(constraints, list):
            raise ParseError("Expected 'task_constraints' to be a list.")
        self.task_constraints = [str(c) for c in constraints if c]
        return self.task_constraints

    def _infer_observation_summary(
        self,
        task_description: str,
        task_constraints: list[str],
        observation: str,
    ) -> dict:
        system_message = ObservationSummaryPrompt.system_message
        user_message = ObservationSummaryPrompt.user_prompt(
            task_description=task_description,
            task_constraints=task_constraints or None,
            task_progress_summary=self.task_progress_summary,
            observation_history=self.observation_history or None,
            action_history=self.action_history or None,
            notes_summary=self.notes_summary,
            observation=observation,
        )
        result = self._call_json_prompt(system_message, user_message)
        self.task_progress_summary = result.get("task_progress")
        self.task_feedback = result.get("task_feedback")
        return result

    def _infer_notes_summary(
        self,
        task_description: str,
        task_constraints: list[str],
        observation: str,
    ) -> dict:
        system_message = NotesSummaryPrompt.system_message
        user_message = NotesSummaryPrompt.user_prompt(
            task_description=task_description,
            task_constraints=task_constraints or None,
            task_progress_summary=self.task_progress_summary,
            action_history=self.action_history or None,
            notes=self.notes_summary,
            observation=observation,
        )
        result = self._call_json_prompt(system_message, user_message)
        new_notes = result.get("new_notes")
        if new_notes:
            if self.notes_summary:
                self.notes_summary = f"{self.notes_summary}\n{new_notes}".strip()
            else:
                self.notes_summary = str(new_notes)
        return result

    def _infer_node_expansion(
        self,
        task_description: str,
        task_constraints: list[str],
        observation: str,
        node: Node,
    ) -> dict:
        system_message = NodeExpansionPrompt(self.action_set, self.action_flags).system_message()
        user_message = NodeExpansionPrompt.user_prompt(
            task_description=task_description,
            task_constraints=task_constraints or None,
            task_progress_summary=self.task_progress_summary,
            notes_summary=self.notes_summary,
            observation=observation,
            node_id=str(node.id),
            node_description=node.description,
            local_tree_info=self._describe_local_tree(node),
        )
        return self._call_json_prompt(system_message, user_message)

    def _apply_expansion(self, node: Node, expansion: dict) -> NodeType:
        node_type = str(expansion.get("node_type", "")).upper().strip()
        if node_type == "ACTION":
            action = expansion.get("expansion")
            if not isinstance(action, str) or not action.strip():
                raise ParseError("ACTION node requires a non-empty 'expansion' string.")
            node.action = action.strip()
            return NodeType.ACTION

        if node_type not in {"AND", "OR"}:
            raise ParseError("node_type must be ACTION, AND, or OR.")

        node.description = expansion.get("node_description", node.description)

        children = expansion.get("expansion", [])
        if not isinstance(children, list) or not children:
            raise ParseError("AND/OR node requires a non-empty 'expansion' list.")

        node.children = []
        for child_text in children:
            if not isinstance(child_text, str):
                continue
            score = None
            clean_text = child_text.strip()
            if node_type == "OR":
                score, clean_text = self._extract_score(clean_text)
            child = Node(type=NodeType.UNKNOWN, description=clean_text, parent=node, score=score)
            node.add_child(child)
        return NodeType.AND if node_type == "AND" else NodeType.OR

    def _extract_score(self, text: str) -> tuple[float | None, str]:
        if "(score:" not in text:
            return None, text
        try:
            before, after = text.split("(score:", 1)
            score_part = after.split(")", 1)[0]
            score = float(score_part.strip())
            cleaned = (before + after.split(")", 1)[1]).strip()
            return score, cleaned
        except (ValueError, IndexError):
            return None, text

    def _call_json_prompt(self, system_message: str, user_message: str | dp.Shrinkable) -> dict:
        if isinstance(user_message, dp.Shrinkable):
            if self.max_prompt_tokens is not None:
                user_message = dp.fit_tokens(
                    shrinkable=user_message,
                    max_prompt_tokens=self.max_prompt_tokens,
                    model_name=self.chat_llm.model_name,
                    max_iterations=self.max_trunc_itr,
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

    def _select_promising_child(self, node: Node) -> Node:
        valid_children = self._get_valid_children(node)
        return max(valid_children, key=lambda child: child.score or 0.0)

    def _describe_local_tree(self, node: Node) -> str:
        parts: list[str] = []
        if node.parent is None:
            return "root_node"
        parts.append(f"parent_node_id: {node.parent.id}")
        parts.append(f"parent_node_description: {node.parent.description}")
        if node.parent.children:
            siblings = [
                f"{child.id}: {child.description} (status={child.status.name})"
                for child in node.parent.children
                if child is not node
            ]
            if siblings:
                parts.append("siblings:\n" + "\n".join(siblings))
        if node.parent.parent and node.parent.parent.children:
            parent_siblings = [
                f"{child.id}: {child.description} (status={child.status.name})"
                for child in node.parent.parent.children
                if child is not node.parent
            ]
            if parent_siblings:
                parts.append("parent_siblings:\n" + "\n".join(parent_siblings))
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
        global_tree = walk_tree(self.root_node, [])
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
            
    def _prune_nodes_from_global_tree(self, pruned_node_ids: list[str], global_tree: list[Node]) -> None:
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
            node.status = NodeStatus.PRUNED
        return

    def _update_nodes_in_global_tree(self, updated_node_ids: dict[str, str], global_tree: list[Node]) -> None:
        """
        Updates the description of the nodes with specified ids.

        Args:
            updated_node_ids: dictionary of node ids to update and their new descriptions.
            global_tree: list of nodes in the global tree.

        Returns:
            None
        """
        nodes_to_update = self._get_nodes_from_ids(node_ids=list(updated_node_ids.keys()), global_tree=global_tree)
        for node in nodes_to_update:
            node.description = updated_node_ids[node.id]
        return
    
    def _global_tree_update(self, task_description: str, task_constraints: list[str], observation: str) -> None:
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
        global_tree_info = "\n\n".join([f"NODE ID: {node.id}\nNODE TYPE: {node.type.name}\nNODE STATUS: {node.status.name}\nNODE DESCRIPTION: {node.description}\nNODE ACTION: {node.action}" for node in global_tree])
        system_message = GlobalTreeUpdatePrompt(self.action_set, self.action_flags).system_message()
        user_message = GlobalTreeUpdatePrompt.user_prompt(
            task_description=task_description,
            task_constraints=task_constraints or None,
            task_progress_summary=self.task_progress_summary,
            notes_summary=self.notes_summary,
            observation=observation,
            global_tree_info=global_tree_info,
        )
        result = self._call_json_prompt(system_message=system_message, user_message=user_message)
        pruned_node_ids = result.get("prune", [])
        updated_node_ids = result.get("update", {})
        self._prune_nodes_from_global_tree(pruned_node_ids=pruned_node_ids, global_tree=global_tree)
        self._update_nodes_in_global_tree(updated_node_ids=updated_node_ids, global_tree=global_tree)
        return 

    def _update_observations(self, node: Node, obs: dict):
        observation = obs.get("axtree_txt") or obs.get("dom_txt") or obs.get("pruned_html")
        if observation:
            self.observation_history.append(observation)
        if node.type == NodeType.ACTION:
            action_history_text = f"{node.id}: {node.description}; Playwright Action: {node.action}"
            if node.action_error:
                action_history_text += f"; Error: {node.action_error}"
            else:
                action_history_text += "; SUCCESS"
            self.action_history.append(action_history_text)

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
        return all(c.status == NodeStatus.SUCCESS for c in node.children)

    def _has_successful_children_or(self, node: Node) -> bool:
        return any(c.status == NodeStatus.SUCCESS for c in node.children)

    def _has_valid_children_and(self, node: Node) -> bool:
        return all(c.status not in {NodeStatus.PRUNED, NodeStatus.DELETED} for c in node.children)

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


def main():
    root = Node(type=NodeType.UNKNOWN, description="Solve task")
    agent = HPA(chat_llm=None, action_set=None)
    agent.reset(root_node=root, goal="Complete the task")
    next_action = agent.run_until_action()
    if next_action is not None:
        logging.info("Next action: %s", next_action.text)


if __name__ == "__main__":
    main()
