from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .belief import BeliefState


@dataclass(frozen=True)
class Task:
    name: str
    args: tuple[str, ...] = ()


@dataclass(frozen=True)
class Method:
    name: str
    task: str
    guard: Callable[[BeliefState], bool]
    decompose: Callable[[BeliefState], list[Task]]


@dataclass(frozen=True)
class Operator:
    name: str
    task: str
    make_skill: Callable[[Task], "Skill"]


class HTNPlanner:
    def __init__(self):
        self._methods: dict[str, list[Method]] = {}
        self._operators: dict[str, Operator] = {}

    def add_method(self, method: Method) -> None:
        self._methods.setdefault(method.task, []).append(method)

    def add_operator(self, operator: Operator) -> None:
        self._operators[operator.task] = operator

    def next_operator(self, tasks: list[Task], belief: BeliefState) -> Operator | None:
        while tasks:
            task = tasks.pop()
            operator = self._operators.get(task.name)
            if operator:
                tasks.append(task)
                return operator
            candidates = self._methods.get(task.name, [])
            chosen = None
            for method in candidates:
                if method.guard(belief):
                    chosen = method
                    break
            if chosen is None:
                return None
            subtasks = chosen.decompose(belief)
            if not subtasks:
                continue
            tasks.extend(reversed(subtasks))
        return None


class Skill:
    def step(self, belief: BeliefState, obs: dict) -> "SkillResult":
        raise NotImplementedError


@dataclass
class SkillResult:
    action: str | None
    done: bool
    info: dict
