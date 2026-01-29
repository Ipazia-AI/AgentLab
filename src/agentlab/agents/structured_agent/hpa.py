import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple


class NodeType(Enum):
    UNKNOWN = auto()
    ACTION = auto()
    AND = auto()
    OR = auto()


class NodeStatus(Enum):
    UNVISITED = auto()
    VISITED = auto()
    SUCCESS = auto()
    FAILED = auto()
    PRUNED = auto()
    DELETED = auto()


class NodeState(Enum):
    ENTERING = auto()
    EXITING = auto()
    FAILED = auto()


@dataclass
class HPANode:
    node_type: NodeType
    parent: Optional["HPANode"] = None
    description: str | None = None
    children: List["HPANode"] = field(default_factory=list)
    status: NodeStatus = NodeStatus.UNVISITED
    execution_count: int = 0
    revision_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_child(self, child: "HPANode"):
        self.children.append(child)


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
        self.budget = budget # remove
        self.max_revision_count = max_revision_count
        self.stack: List[Tuple[HPANode, NodeState]] = []

    # and_or_tree_traversal
    def and_or_tree_traversal(self, root_node: HPANode, env, goal: str | None = None):
        self.stack = [(root_node, NodeState.ENTERING)]
        counter = 0

        while self.stack:
            node, state = self.stack.pop()

            if node.status == NodeStatus.PRUNED:
                self._propagate_failure(node)
                continue

            if node.status == NodeStatus.DELETED:
                continue

            if state == NodeState.ENTERING:
                self._process_node_entering(node, env, goal)

            elif state == NodeState.EXITING:
                self._process_node_exiting(node)

            elif state == NodeState.FAILED:
                self._process_node_failed(node)

            counter += 1
            if counter >= self.budget:
                break

        return root_node.status

    # Algo 2 from the HPA paper
    def _process_node_entering(self, node: HPANode, env, goal: str | None):
        if node.parent and node.parent.node_type == NodeType.OR:
            self._rollback_context(node.parent)

        self._set_context(node)
        node.execution_count += 1

        if node.node_type == NodeType.UNKNOWN:
            node.node_type = self._populate_node_type(node, goal)
            self.stack.append((node, NodeState.EXITING))

        if node.node_type == NodeType.ACTION:
            self.stack.append((node, NodeState.EXITING))
            success = self._perform_action(node, env)
            if success:
                self._global_tree_update()
                self._update_observations(node)
                node.status = NodeStatus.SUCCESS
            else:
                node.status = NodeStatus.FAILED
            return

        if node.node_type == NodeType.AND:
            if self._is_successful(node):
                return
            if self._has_valid_child(node):
                for child in reversed(node.children):
                    if child.status not in self._closed_statuses():
                        self.stack.append((child, NodeState.ENTERING))
            else:
                node.status = NodeStatus.FAILED

        if node.node_type == NodeType.OR:
            if self._is_successful(node):
                return
            if self._is_valid_or(node):
                child = self._select_promising_child(node)
                self.stack.append((child, NodeState.ENTERING))
            else:
                node.status = NodeStatus.FAILED

    # Algo 3 from the HPA paper
    def _process_node_exiting(self, node: HPANode):
        if node.node_type == NodeType.ACTION:
            if node.status in {NodeStatus.FAILED, NodeStatus.PRUNED}:
                self.stack.append((node, NodeState.FAILED))
            return
        else:
            node.status = NodeStatus.SUCCESS

        if node.node_type == NodeType.AND:
            if self._is_successful_and(node):
                if self._check_and_complete(node):
                    node.status = NodeStatus.SUCCESS
                    return
            node.status = NodeStatus.FAILED
            self.stack.append((node, NodeState.FAILED))

        elif node.node_type == NodeType.OR:
            if self._is_successful_or(node):
                node.status = NodeStatus.SUCCESS
            else:
                node.status = NodeStatus.FAILED
                self.stack.append((node, NodeState.FAILED))

    # Algo 4 from the HPA paper
    def _process_node_failed(self, node: HPANode):
        if node.node_type == NodeType.ACTION:
            node.status = NodeStatus.PRUNED
            self._propagate_failure(node)
            return

        elif node.node_type == NodeType.AND:
            if self._has_success(node) and self._check_and_complete(node):
                node.status = NodeStatus.SUCCESS
                return

            if not self._is_valid_and(node):
                if node.revision_count < self.max_revision_count:
                    revised = self._revise_and(node)
                    self._synchronize_stack()
                    if revised:
                        node.status = NodeStatus.VISITED
                        self.stack.append((node, NodeState.ENTERING))
                else:
                    self._prune(node)
                    self._synchronize_stack()
                    self._propagate_failure(node)

        elif node.node_type == NodeType.OR:
            if self._is_valid_or(node):
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
                self._prune(node)
                self._synchronize_stack()
                self._propagate_failure(node)

    def _populate_node_type(self, node: HPANode, goal: str | None) -> NodeType:
        return NodeType.ACTION

    def _perform_action(self, node: HPANode, env) -> bool:
        # Need to implement action execution logic
        try:
            env.step(node.description)
            return True
        except Exception as exc:
            logging.warning(f"HPA action failed: {exc}")
            return False

    def _select_promising_child(self, node: HPANode) -> HPANode:
        # Need to implement child selection logic
        return node.children[0]

    def _rollback_context(self, node: HPANode): 
        pass

    def _set_context(self, node: HPANode): 
        pass

    def _global_tree_update(self): 
        pass

    def _update_observations(self, node: HPANode): 
        pass

    def _propagate_failure(self, node: HPANode): 
        pass

    def _synchronize_stack(self): 
        pass

    def _revise_and(self, node: HPANode) -> bool:
        node.revision_count += 1
        return False

    def _revise_or(self, node: HPANode) -> bool:
        node.revision_count += 1
        return False

    def _prune(self, node: HPANode):
        node.status = NodeStatus.PRUNED

    def _closed_statuses(self):
        return {NodeStatus.SUCCESS, NodeStatus.FAILED, NodeStatus.PRUNED}

    def _is_successful(self, node: HPANode) -> bool: 
        return node.status == NodeStatus.SUCCESS
    
    def _is_successful_and(self, node: HPANode) -> bool: 
        return all(c.status == NodeStatus.SUCCESS for c in node.children)
    
    def _is_successful_or(self, node: HPANode) -> bool: 
        return any(c.status == NodeStatus.SUCCESS for c in node.children)

    def _has_valid_child(self, node: HPANode) -> bool: 
        return any(c.status not in {NodeStatus.PRUNED, NodeStatus.DELETED} for c in node.children)
    
    def _has_success(self, node: HPANode) -> bool: 
        if node.node_type == NodeType.AND:
            return all(c.status == NodeStatus.SUCCESS for c in node.children)
        if node.node_type == NodeType.OR:
            return any(c.status == NodeStatus.SUCCESS for c in node.children)
        return False
    
    def _is_valid_and(self, node: HPANode) -> bool: 
        return all(c.status not in {NodeStatus.PRUNED, NodeStatus.DELETED} for c in node.children)
    
    def _is_valid_or(self, node: HPANode) -> bool: 
        return any(c.status not in {NodeStatus.PRUNED, NodeStatus.DELETED} for c in node.children)
    
    def _check_and_complete(self, node: HPANode) -> bool: 
        return all(c.status == NodeStatus.SUCCESS for c in node.children)


def main():
    root = HPANode(NodeType.UNKNOWN, description="Solve task")
    agent = HPA(chat_llm=None, action_set=None)
    agent.and_or_tree_traversal(root_node=root, env=None, goal="Complete the task")


if __name__ == "__main__":
    main()
