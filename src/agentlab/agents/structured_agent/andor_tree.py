from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Optional

from pydantic import BaseModel, Field


class TreeContextEntry(BaseModel):
    depth: int
    node_id: str
    description: str
    status: str
    type_label: str
    marker_expand: bool = False
    action_error: str | None = None

    def format(self) -> str:
        indent = "  " * self.depth
        status_prefix = ""
        if self.status == "FAIL":
            err = f": {self.action_error}" if self.action_error else ""
            status_prefix = f"[FAIL{err}] "
        elif self.status:
            status_prefix = f"[{self.status}] "
        type_label = f" ({self.type_label})" if self.type_label else ""
        marker = "  ← EXPAND THIS NODE" if self.marker_expand else ""
        return f"{indent}{status_prefix}{self.node_id}{type_label}: {self.description}{marker}"


class NodeType(Enum):
    UNKNOWN = auto()
    ACTION = auto()
    AND = auto()
    OR = auto()


class NodeStatus(Enum):
    UNVISITED = auto()
    VISITED = auto()
    SUCCESS = auto()
    FAIL = auto()
    PRUNED = auto()
    DELETED = auto()


CLOSED_STATUSES = {NodeStatus.SUCCESS, NodeStatus.DELETED, NodeStatus.PRUNED}


class NodeState(Enum):
    ENTERING = auto()
    EXITING = auto()
    FAILED = auto()


class Node(BaseModel):
    id: str = "0"
    type: NodeType
    status: NodeStatus = NodeStatus.UNVISITED
    description: str
    action: str | None = None
    action_error: str | None = None

    parent: Optional["Node"] = None
    children: list["Node"] = Field(default_factory=list)

    metadata: dict[str, Any] = Field(default_factory=dict)

    revision_count: int = 0
    execution_count: int = 0

    @property
    def depth(self) -> int:
        """Return the depth of this node in the tree (root = 0)."""
        return self.id.count(".")

    def add_child(self, child: "Node"):
        child.id = self.id + f".{len(self.children)+1}"
        self.children.append(child)

    def __str__(self) -> str:
        return f"Node(id={self.id}, type={self.type.name}, status={self.status.name}, description={self.description}, action={self.action}, action_error={self.action_error}, parent={self.parent.id if self.parent else None}, children={len(self.children)})"

    @property
    def closed(self) -> bool:
        return self.status in CLOSED_STATUSES

    @property
    def prompt_description(self) -> str:
        return f"{self.id}: {self.description}"

    @property
    def valid_children(self) -> list["Node"]:
        return [c for c in self.children if c.status not in {NodeStatus.PRUNED, NodeStatus.DELETED}]

    @property
    def successful_children_and(self) -> bool:
        return all(
            c.status == NodeStatus.SUCCESS or c.status == NodeStatus.DELETED for c in self.children
        )

    @property
    def successful_children_or(self) -> bool:
        return any(c.status == NodeStatus.SUCCESS for c in self.children)

    @property
    def valid_children_and(self) -> bool:
        return all(c.status not in {NodeStatus.PRUNED} for c in self.children)

    @property
    def valid_children_or(self) -> bool:
        return any(c.status not in {NodeStatus.PRUNED, NodeStatus.DELETED} for c in self.children)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "description": self.description,
            "type": self.type.name,
            "status": self.status.name,
            "depth": self.depth,
            "parent_id": None if self.parent is None else self.parent.id,
            "parent_type": None if self.parent is None else self.parent.type.name,
        }

    def to_tree_context_entry(self) -> list[TreeContextEntry]:
        ancestor_ids: set[str] = set()
        current = self
        while current is not None:
            ancestor_ids.add(current.id)
            current = current.parent

        root = self
        while root.parent is not None:
            root = root.parent

        entries: list[TreeContextEntry] = []

        def _render(node: Node, depth: int, show_deleted_as_failed: bool = False) -> None:
            if node.status == NodeStatus.DELETED:
                if show_deleted_as_failed:
                    entries.append(
                        TreeContextEntry(
                            depth=depth,
                            node_id=node.id,
                            description=node.description,
                            status="FAILED",
                            type_label="",
                            marker_expand=node.id == self.id,
                        )
                    )
                return

            type_label = node.type.name if node.type in {NodeType.AND, NodeType.OR} else ""
            entries.append(
                TreeContextEntry(
                    depth=depth,
                    node_id=node.id,
                    description=node.description,
                    status=node.status.name,
                    type_label=type_label,
                    marker_expand=node.id == self.id,
                    action_error=node.action_error,
                )
            )

            if node.id in ancestor_ids:
                is_or = node.type == NodeType.OR
                for child in node.children:
                    _render(child, depth + 1, show_deleted_as_failed=is_or)

        _render(root, 0)
        return entries


class TreeExpansionTrace(BaseModel):
    retry: int
    expanded_node: Node
    children_after: list[Node]
    tree_context_before: list[TreeContextEntry]
    tree_context_after: list[TreeContextEntry]
