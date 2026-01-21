from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class Predicate:
    name: str
    args: tuple[str, ...] = ()

    def __str__(self) -> str:
        if not self.args:
            return self.name
        return f"{self.name}({', '.join(self.args)})"


def pred(name: str, *args: str) -> Predicate:
    return Predicate(name=name, args=tuple(args))


def format_predicates(predicates: Iterable[Predicate]) -> list[str]:
    return [str(p) for p in predicates]
