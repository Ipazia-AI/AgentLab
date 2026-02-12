from enum import Enum, auto
from typing import Any, Optional

from pydantic import BaseModel, Field


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

    score: Optional[float] = None
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
