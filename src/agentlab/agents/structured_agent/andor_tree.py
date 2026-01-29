from collections.abc import Callable
from enum import Enum, auto
from typing import Any, Literal, Protocol
from uuid import UUID, uuid4

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
    FAILED = auto()
    PRUNED = auto()
    DELETED = auto()


class NodeState(Enum):
    ENTERING = auto()
    EXITING = auto()
    FAILED = auto()


class Node(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    type: NodeType
    status: NodeStatus = NodeStatus.UNVISITED
    text: str

    parent: "Node" | None = None
    children: list["Node"] = Field(default_factory=list)

    score: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    revision_count: int = 0
    execution_count: int = 0
    
    def add_child(self, child: "Node"):
        self.children.append(child)

