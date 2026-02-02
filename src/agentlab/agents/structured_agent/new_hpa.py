# hpa.py
# Paper-faithful Hierarchical Planning Algorithm (HPA)
# Planner-only: no env, no LLM. Delegates actions to StructuredAgent via handler.

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Protocol


class NodeType(Enum):
    UNKNOWN = auto()
    ACTION = auto()
    AND = auto()
    OR = auto()


class NodeStatus(Enum):
    UNKNOWN = auto()
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
    node_type: NodeType = NodeType.UNKNOWN
    status: NodeStatus = NodeStatus.UNKNOWN
    parent: Optional["HPANode"] = None
    children: List["HPANode"] = field(default_factory=list)
    execution_count: int = 0
    revision_count: int = 0
    max_revisions: int = 3
    metadata: dict = field(default_factory=dict)


class HPAHandler(Protocol):
    # Context + typing
    def set_context(self, node: HPANode): ...
    def rollback_context(self, node: HPANode): ...
    def populate_node_type(self, node: HPANode, goal: Optional[str]): ...

    # Action + planning
    def perform_action(self, node: HPANode) -> bool: ...
    def select_promising_child(self, node: HPANode) -> HPANode: ...

    # Repair / revision
    def revise_and(self, node: HPANode) -> bool: ...
    def revise_or(self, node: HPANode) -> bool: ...

    # Post action hook
    def update_after_action(self, node: HPANode): ...


class HPA:
    def __init__(self, handler: HPAHandler, budget: int = 500):
        self.handler = handler
        self.budget = budget
        self.stack: List[Tuple[HPANode, NodeState]] = []

    # ---------------- Algorithm 1 ----------------
    def search(self, root: HPANode, goal: Optional[str] = None):
        self.stack = [(root, NodeState.ENTERING)]
        counter = 0

        while self.stack:
            node, state = self.stack.pop()

            if node.status == NodeStatus.PRUNED:
                self._propagate_failure(node)
                continue
            if node.status == NodeStatus.DELETED:
                continue

            if state == NodeState.ENTERING:
                self._process_node_entering(node, goal)
            elif state == NodeState.EXITING:
                self._process_node_exiting(node)
            elif state == NodeState.FAILED:
                self._process_node_failed(node)

            counter += 1
            if counter >= self.budget:
                break

        return root.status

    # ---------------- Algorithm 2 ----------------
    def _process_node_entering(self, node: HPANode, goal: Optional[str]):
        if node.parent and node.parent.node_type == NodeType.OR:
            self.handler.rollback_context(node.parent)

        self.handler.set_context(node)
        node.execution_count += 1

        if node.node_type == NodeType.UNKNOWN:
            node.node_type = self.handler.populate_node_type(node, goal)
            # self.stack.append((node, NodeState.EXITING))
            # return

        if node.node_type == NodeType.ACTION:
            self.stack.append((node, NodeState.EXITING))
            success = self.handler.perform_action(node)
            if success:
                self.handler.update_after_action(node)
                node.status = NodeStatus.SUCCESS
            else:
                node.status = NodeStatus.FAILED
            return

        if node.node_type == NodeType.AND:
            if self._has_success(node):
                return
            if self._has_valid_child(node):
                for child in reversed(node.children):
                    if child.status not in self._closed():
                        self.stack.append((child, NodeState.ENTERING))
            else:
                node.status = NodeStatus.FAILED
            self.stack.append((node, NodeState.EXITING))
            return

        if node.node_type == NodeType.OR:
            if self._has_success(node):
                return
            if self._is_valid_or(node):
                child = self.handler.select_promising_child(node)
                self.stack.append((child, NodeState.ENTERING))
            else:
                node.status = NodeStatus.FAILED
            self.stack.append((node, NodeState.EXITING))
            return

    # ---------------- Algorithm 3 ----------------
    def _process_node_exiting(self, node: HPANode):
        if node.node_type == NodeType.ACTION:
            if node.status in {NodeStatus.FAILED, NodeStatus.PRUNED}:
                self.stack.append((node, NodeState.FAILED))
            return

        if node.node_type == NodeType.AND:
            if not self._has_success(node):
                node.status = NodeStatus.FAILED
                self.stack.append((node, NodeState.FAILED))
                return
            if self._check_and_complete(node):
                node.status = NodeStatus.SUCCESS
                return
            return

        if node.node_type == NodeType.OR:
            if self._has_success(node):
                node.status = NodeStatus.SUCCESS
            else:
                node.status = NodeStatus.FAILED
                self.stack.append((node, NodeState.FAILED))
            return

    # ---------------- Algorithm 4 ----------------
    def _process_node_failed(self, node: HPANode):
        if node.node_type == NodeType.ACTION:
            node.status = NodeStatus.PRUNED
            self._propagate_failure(node)
            return

        if node.node_type == NodeType.AND:
            if self._has_success(node) and self._check_and_complete(node):
                node.status = NodeStatus.SUCCESS
                return

            if not self._is_valid_and(node):
                if node.revision_count < node.max_revisions:
                    node.revision_count += 1
                    revised = self.handler.revise_and(node)
                    if revised:
                        node.status = NodeStatus.UNKNOWN
                        self.stack.append((node, NodeState.ENTERING))
                        return
                node.status = NodeStatus.PRUNED
                self._propagate_failure(node)
                return

            node.status = NodeStatus.UNKNOWN
            self.stack.append((node, NodeState.ENTERING))
            return

        if node.node_type == NodeType.OR:
            if self._is_valid_or(node):
                node.status = NodeStatus.UNKNOWN
                self.stack.append((node, NodeState.ENTERING))
                return

            if node.revision_count < node.max_revisions:
                node.revision_count += 1
                revised = self.handler.revise_or(node)
                if revised:
                    node.status = NodeStatus.UNKNOWN
                    self.stack.append((node, NodeState.ENTERING))
                    return

            node.status = NodeStatus.PRUNED
            self._propagate_failure(node)
            return

    # ---------------- Predicates ----------------
    def _closed(self):
        return {NodeStatus.PRUNED, NodeStatus.DELETED}

    def _has_valid_child(self, node: HPANode) -> bool:
        return any(c.status not in self._closed() for c in node.children)

    def _has_success(self, node: HPANode) -> bool:
        if node.node_type == NodeType.AND:
            return all(c.status == NodeStatus.SUCCESS for c in node.children)
        if node.node_type == NodeType.OR:
            return any(c.status == NodeStatus.SUCCESS for c in node.children)
        return False

    def _is_valid_and(self, node: HPANode) -> bool:
        return all(c.status not in self._closed() for c in node.children)

    def _is_valid_or(self, node: HPANode) -> bool:
        return any(c.status not in self._closed() for c in node.children)

    def _check_and_complete(self, node: HPANode) -> bool:
        return all(c.status == NodeStatus.SUCCESS for c in node.children)

    # ---------------- Utils ----------------
    def _propagate_failure(self, node: HPANode):
        if node.parent:
            self.stack.append((node.parent, NodeState.FAILED))
