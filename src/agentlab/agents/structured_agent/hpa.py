import json
import logging
from typing import List, Tuple

from agentlab.llm.llm_utils import (
    Discussion,
    HumanMessage,
    ParseError,
    SystemMessage,
    retry,
)

from .andor_tree import Node, NodeState, NodeStatus, NodeType
from .prompts import (
    NodeExpansionPrompt,
    NotesSummaryPrompt,
    ObservationSummaryPrompt,
    TaskConstraintsPrompt,
)


class HPA:
    def __init__(
        self,
        chat_llm,
        action_set,
        budget: int = 1000,
        max_revision_count: int = 3,
    ):
        self.chat_llm = chat_llm
        self.action_set = action_set
        self.budget = budget  # remove
        self.max_revision_count = max_revision_count
        self.stack: List[Tuple[Node, NodeState]] = []
        self._counter = 0
        self.task_constraints: list[str] = []
        self.task_progress_summary: str | None = None
        self.notes_summary: str | None = None
        self.observation_history: list[str] = []
        self.action_history: list[str] = []
        self.task_feedback: str | None = None

    def reset(self):
        self.stack = [(Node(type=NodeType.UNKNOWN, text="Solve task"), NodeState.ENTERING)]
        self._counter = 0
        self.task_constraints = []
        self.task_progress_summary = None
        self.notes_summary = None
        self.observation_history = []
        self.action_history = []
        self.task_feedback = None

    def run_until_action(self, goal: str | None, obs: dict) -> Node | None:
        while self.stack:
            node, state = self.stack.pop()

            if node.status == NodeStatus.PRUNED:
                self._propagate_failure(node)
                continue

            if node.status == NodeStatus.DELETED:
                continue

            if state == NodeState.ENTERING:
                action_node = self._process_node_entering(node, goal, obs)
                if action_node.type == NodeType.ACTION:
                    return action_node

            elif state == NodeState.EXITING:
                self._process_node_exiting(node)

            elif state == NodeState.FAILED:
                self._process_node_failed(node)

            self._counter += 1
            if self._counter >= self.budget:
                break

        return None

    def finalize_action(self, node: Node, obs: dict, success: bool):
        if success:
            self._global_tree_update()
            self._update_observations(node, obs)
            node.status = NodeStatus.SUCCESS
        else:
            node.status = NodeStatus.FAIL

    # Algo 2 from the HPA paper
    def _process_node_entering(self, node: Node, goal: str | None, obs: dict) -> Node:
        if node.parent and node.parent.type == NodeType.OR:
            # TODO: The role of this function is not clear, let's check it later what it is supposed to do.
            self._rollback_context(node.parent)
        
        # TODO: The role of this function is not clear, let's check it later what it is supposed to do.
        self._set_context(node)
        node.execution_count += 1

        if node.type == NodeType.UNKNOWN:
            self._expand_node(node, goal, obs)
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
        #return None
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
            if not self._has_valid_children_and(node) and node.revision_count < self.max_revision_count:
                revised = self._revise_and(node)
                self._synchronize_stack()
                if revised:
                    node.status = NodeStatus.VISITED
                    self.stack.append((node, NodeState.ENTERING))
            else:
                node.status = NodeStatus.PRUNED
                self._synchronize_stack()
                self._propagate_failure(node)

        elif node.type == NodeType.OR:
            if self._has_valid_children_or(node):
                node.status = NodeStatus.VISITED
                self.stack.append((node, NodeState.ENTERING))
                return

            if node.revision_count < self.max_revision_count:
                revised = self._revise_or(node)
                self._synchronize_stack()
                if revised:
                    node.status = NodeStatus.VISITED
                    self.stack.append((node, NodeState.ENTERING))
            else:
                node.status = NodeStatus.PRUNED
                self._synchronize_stack()
                self._propagate_failure(node)

    def _expand_node(self, node: Node, goal: str | None, obs: dict) -> Node:
        if self.chat_llm is None:
            node.text = "report_infeasible"
            node.metadata["node_description"] = "LLM not available"
            node.type = NodeType.ACTION
            return node

        task_description = goal or "Complete the task."
        observation = obs.get("axtree_txt") or obs.get("dom_txt") or obs.get("pruned_html") or ""

        task_constraints = self._infer_task_constraints(task_description, observation)
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

    def _infer_task_constraints(self, task_description: str, observation: str) -> list[str]:
        system_message = TaskConstraintsPrompt.system_message
        user_message = TaskConstraintsPrompt.user_message(
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
        user_message = ObservationSummaryPrompt.user_message(
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
        user_message = NotesSummaryPrompt.user_message(
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
        system_message = NodeExpansionPrompt.system_message
        user_message = NodeExpansionPrompt.user_message(
            task_description=task_description,
            task_constraints=task_constraints or None,
            task_progress_summary=self.task_progress_summary,
            notes_summary=self.notes_summary,
            observation=observation,
            node_id=str(node.id),
            node_description=node.text,
            local_tree_info=self._describe_local_tree(node),
        )
        return self._call_json_prompt(system_message, user_message)

    def _apply_expansion(self, node: Node, expansion: dict) -> NodeType:
        node_type = str(expansion.get("node_type", "")).upper().strip()
        if node_type == "ACTION":
            action = expansion.get("expansion")
            if not isinstance(action, str) or not action.strip():
                raise ParseError("ACTION node requires a non-empty 'expansion' string.")
            node.text = action.strip()
            node.metadata["node_description"] = expansion.get("node_description", node.text)
            return NodeType.ACTION

        if node_type not in {"AND", "OR"}:
            raise ParseError("node_type must be ACTION, AND, or OR.")

        node.metadata["node_description"] = expansion.get("node_description", node.text)
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
            child = Node(type=NodeType.UNKNOWN, text=clean_text, parent=node, score=score)
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

    def _call_json_prompt(self, system_message: str, user_message: str) -> dict:
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
        parts.append(f"parent_node_description: {node.parent.text}")
        if node.parent.children:
            siblings = [
                f"{child.id}: {child.text} (status={child.status.name})"
                for child in node.parent.children
                if child is not node
            ]
            if siblings:
                parts.append("siblings:\n" + "\n".join(siblings))
        if node.parent.parent and node.parent.parent.children:
            parent_siblings = [
                f"{child.id}: {child.text} (status={child.status.name})"
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

    def _global_tree_update(self):
        pass

    def _update_observations(self, node: Node, obs: dict):
        observation = obs.get("axtree_txt") or obs.get("dom_txt") or obs.get("pruned_html")
        if observation:
            self.observation_history.append(observation)
        if node.type == NodeType.ACTION and node.text:
            self.action_history.append(node.text)

    def _propagate_failure(self, node: Node):
        pass

    def _synchronize_stack(self):
        pass

    def _revise_and(self, node: Node) -> bool:
        node.revision_count += 1
        return False

    def _revise_or(self, node: Node) -> bool:
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
    root = Node(type=NodeType.UNKNOWN, text="Solve task")
    agent = HPA(chat_llm=None, action_set=None)
    agent.reset(root_node=root, goal="Complete the task")
    next_action = agent.run_until_action()
    if next_action is not None:
        logging.info("Next action: %s", next_action.text)


if __name__ == "__main__":
    main()
