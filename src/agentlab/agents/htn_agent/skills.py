from __future__ import annotations

from dataclasses import dataclass

from .belief import BeliefState, Element
from .htn import Skill, SkillResult


def _click_action(bid: str) -> str:
    return f'click("{bid}")'


def _type_action(bid: str, text: str) -> str:
    escaped = text.replace('"', '\\"')
    return f'type("{bid}", "{escaped}")'


def _match_element(elements: list[Element], role: str, name_hint: str) -> Element | None:
    lowered = name_hint.lower()
    for element in elements:
        if role != "any" and element.role != role:
            continue
        if lowered in element.name.lower():
            return element
    return None


@dataclass
class FindAndClickSkill(Skill):
    role: str
    name_hint: str

    def step(self, belief: BeliefState, obs: dict) -> SkillResult:
        element = _match_element(belief.elements, self.role, self.name_hint)
        if not element:
            return SkillResult(action=None, done=True, info={"reason": "no_match"})
        return SkillResult(action=_click_action(element.bid), done=True, info={"bid": element.bid})


@dataclass
class FillFieldSkill(Skill):
    field_name: str
    value: str

    def step(self, belief: BeliefState, obs: dict) -> SkillResult:
        element = _match_element(belief.elements, "input", self.field_name)
        if not element:
            return SkillResult(action=None, done=True, info={"reason": "no_field"})
        return SkillResult(
            action=_type_action(element.bid, self.value),
            done=True,
            info={"bid": element.bid},
        )


@dataclass
class SubmitSkill(Skill):
    def step(self, belief: BeliefState, obs: dict) -> SkillResult:
        for label in ["submit", "save", "create", "update"]:
            element = _match_element(belief.elements, "button", label)
            if element:
                return SkillResult(action=_click_action(element.bid), done=True, info={"bid": element.bid})
        return SkillResult(action=None, done=True, info={"reason": "no_submit"})


@dataclass
class ExploreSkill(Skill):
    def step(self, belief: BeliefState, obs: dict) -> SkillResult:
        for element in belief.elements:
            if element.role in {"button", "link"}:
                return SkillResult(action=_click_action(element.bid), done=True, info={"bid": element.bid})
        return SkillResult(action=None, done=True, info={"reason": "no_action"})
