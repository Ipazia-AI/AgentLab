from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

from .predicates import Predicate, pred


@dataclass
class Element:
    bid: str
    role: str
    name: str


@dataclass
class GoalSpec:
    raw_text: str
    intent: str = "unknown"
    fields: dict[str, str] = field(default_factory=dict)


@dataclass
class BeliefState:
    predicates: dict[Predicate, float] = field(default_factory=dict)
    elements: list[Element] = field(default_factory=list)
    goal: GoalSpec | None = None

    def set(self, predicate: Predicate, confidence: float = 1.0) -> None:
        self.predicates[predicate] = max(self.predicates.get(predicate, 0.0), confidence)

    def has(self, predicate: Predicate, min_conf: float = 0.5) -> bool:
        return self.predicates.get(predicate, 0.0) >= min_conf

    def find(self, name: str, min_conf: float = 0.5) -> list[Predicate]:
        return [p for p, c in self.predicates.items() if p.name == name and c >= min_conf]

    def update(self, obs: dict) -> None:
        self.elements = self._extract_elements(obs)
        self._extract_navigation(obs)
        self._extract_page_type(obs)
        self._extract_actions()
        self._extract_fields(obs)
        self._extract_errors(obs)

    def ensure_goal(self, raw_goal: str | None) -> None:
        if self.goal or not raw_goal:
            return
        intent = self._infer_intent(raw_goal)
        fields = self._parse_goal_fields(raw_goal)
        self.goal = GoalSpec(raw_text=raw_goal, intent=intent, fields=fields)
        if fields:
            for field_name, value in fields.items():
                self.set(pred("GoalField", field_name, value), confidence=0.7)
        self.set(pred("GoalIntent", intent), confidence=0.7)

    def summary(self, max_items: int = 12) -> list[str]:
        items = sorted(self.predicates.items(), key=lambda item: item[1], reverse=True)
        return [f"{str(p)}:{c:.2f}" for p, c in items[:max_items]]

    def _extract_elements(self, obs: dict) -> list[Element]:
        elements: list[Element] = []
        text_sources = []
        if obs.get("axtree_txt"):
            text_sources.append(obs["axtree_txt"])
        if obs.get("pruned_html"):
            text_sources.append(obs["pruned_html"])
        combined = "\n".join(text_sources)
        pattern = re.compile(r"^\[(?P<bid>[^\]]+)\]\s*(?P<rest>.+)$", re.MULTILINE)
        for match in pattern.finditer(combined):
            bid = match.group("bid").strip()
            rest = match.group("rest").strip()
            role = self._infer_role(rest)
            name = self._infer_name(rest)
            if name:
                elements.append(Element(bid=bid, role=role, name=name))
        return elements

    def _infer_role(self, text: str) -> str:
        lowered = text.lower()
        if "button" in lowered:
            return "button"
        if "link" in lowered:
            return "link"
        if "textbox" in lowered or "input" in lowered:
            return "input"
        return "unknown"

    def _infer_name(self, text: str) -> str:
        cleaned = re.sub(r"\b(button|link|textbox|input)\b", "", text, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned[:80]

    def _extract_navigation(self, obs: dict) -> None:
        for url in obs.get("open_pages_urls", []) or []:
            self.set(pred("OnURL", url), confidence=0.8)
        for title in obs.get("open_pages_titles", []) or []:
            self.set(pred("OnTitle", title), confidence=0.8)

    def _extract_page_type(self, obs: dict) -> None:
        html = obs.get("pruned_html", "") or ""
        if "<form" in html.lower():
            self.set(pred("OnPageType", "form"), confidence=0.6)
        if "<table" in html.lower():
            self.set(pred("OnPageType", "list"), confidence=0.6)

    def _extract_actions(self) -> None:
        for element in self.elements:
            if element.role in {"button", "link"}:
                self.set(pred("ActionAvailable", element.role, element.name, element.bid), confidence=0.7)

    def _extract_fields(self, obs: dict) -> None:
        html = obs.get("pruned_html", "") or ""
        for label in re.findall(r"<label[^>]*>([^<]+)</label>", html, flags=re.IGNORECASE):
            normalized = " ".join(label.split())
            if normalized:
                self.set(pred("FieldLabel", normalized), confidence=0.6)

    def _extract_errors(self, obs: dict) -> None:
        error = obs.get("last_action_error", "")
        if error:
            self.set(pred("ActionError"), confidence=0.9)
        for text in self._scan_text(obs, ["error", "failed", "invalid"]):
            self.set(pred("ErrorBanner", text), confidence=0.5)

    def _scan_text(self, obs: dict, keywords: Iterable[str]) -> list[str]:
        haystack = " ".join(
            str(part)
            for part in [
                obs.get("axtree_txt", ""),
                obs.get("pruned_html", ""),
                obs.get("dom_txt", ""),
            ]
        ).lower()
        hits = []
        for keyword in keywords:
            if keyword in haystack:
                hits.append(keyword)
        return hits

    def _infer_intent(self, raw_goal: str) -> str:
        lowered = raw_goal.lower()
        if any(k in lowered for k in ["create", "new", "submit"]):
            return "create"
        if any(k in lowered for k in ["update", "edit", "change"]):
            return "update"
        if any(k in lowered for k in ["find", "search", "locate"]):
            return "search"
        return "unknown"

    def _parse_goal_fields(self, raw_goal: str) -> dict[str, str]:
        fields: dict[str, str] = {}
        for match in re.finditer(r"([A-Za-z][A-Za-z _-]{2,})\s*[:=]\s*([^,.;]+)", raw_goal):
            field = " ".join(match.group(1).split())
            value = match.group(2).strip()
            if field and value:
                fields[field] = value
        return fields
