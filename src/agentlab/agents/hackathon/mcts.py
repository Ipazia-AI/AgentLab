import json
import logging
import math
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from agentlab.llm.llm_utils import Discussion, HumanMessage, SystemMessage


@dataclass
class MCTSNode:
    obs: dict
    parent: Optional["MCTSNode"]
    action_from_parent: Optional[str]
    depth: int = 0
    children: dict[str, "MCTSNode"] = field(default_factory=dict)
    visit_count: int = 0
    value_sum: float = 0.0

    @property
    def value(self) -> float:
        return self.value_sum / self.visit_count if self.visit_count else 0.0


class MCTSPlanner:
    def __init__(
        self,
        chat_llm,
        action_set,
        max_simulations: int = 6,
        expansion_width: int = 4,
        max_depth: int = 1,
        uct_c: float = 1.4,
        max_obs_chars: int = 4000,
    ) -> None:
        self.chat_llm = chat_llm
        self.action_set = action_set
        self.max_simulations = max_simulations
        self.expansion_width = expansion_width
        self.max_depth = max_depth
        self.uct_c = uct_c
        self.max_obs_chars = max_obs_chars

    def search(self, root_obs: dict, env, obs_preprocessor=None, goal: str | None = None):
        root = MCTSNode(obs=root_obs, parent=None, action_from_parent=None, depth=0)
        candidates = []

        for _ in range(self.max_simulations):
            node = self._select(root)
            if node.depth >= self.max_depth:
                value = self._evaluate(node.obs, goal)
                self._backpropagate(node, value)
                continue

            actions = self._expand(node, goal)
            if not actions:
                value = self._evaluate(node.obs, goal)
                self._backpropagate(node, value)
                continue

            action = self._pick_untried_action(node, actions)
            child = node.children.get(action)
            if child is None:
                child = MCTSNode(
                    obs=node.obs,
                    parent=node,
                    action_from_parent=action,
                    depth=node.depth + 1,
                )
                node.children[action] = child

            value, next_obs, reward = self._simulate(
                action, env, obs_preprocessor=obs_preprocessor, goal=goal
            )
            if next_obs is not None:
                child.obs = next_obs
            self._backpropagate(child, value)
            candidates.append((action, value, reward))

        best_action = None
        if root.children:
            best_action = max(root.children.values(), key=lambda n: n.value).action_from_parent
        elif candidates:
            best_action = max(candidates, key=lambda c: c[1])[0]

        info = {
            "selected_action": best_action,
            "candidates": candidates,
            "root_value": root.value,
            "root_visits": root.visit_count,
        }
        return best_action, info

    def _select(self, node: MCTSNode) -> MCTSNode:
        while node.children:
            node = max(
                node.children.values(),
                key=lambda child: self._uct_score(node, child),
            )
        return node

    def _uct_score(self, parent: MCTSNode, child: MCTSNode) -> float:
        if child.visit_count == 0:
            return float("inf")
        exploitation = child.value
        exploration = self.uct_c * math.sqrt(
            math.log(parent.visit_count + 1) / child.visit_count
        )
        return exploitation + exploration

    def _expand(self, node: MCTSNode, goal: str | None) -> list[str]:
        obs_text = self._obs_to_text(node.obs)
        action_space = self.action_set.describe(with_long_description=False, with_examples=False)
        prompt = (
            "Propose a short list of candidate actions to explore the task. "
            "Prefer safe actions (scrolling, opening details, focusing fields). "
            "Return ONLY a JSON array of action strings.\n\n"
            f"Goal:\n{goal or ''}\n\n"
            f"Observation:\n{obs_text}\n\n"
            f"Action space:\n{action_space}\n"
        )
        messages = Discussion([SystemMessage("You propose valid browser actions."), HumanMessage(prompt)])
        response = self.chat_llm(messages)
        actions = self._parse_json_list(response["content"])
        valid_actions = []
        for action in actions:
            if self._is_action_valid(action):
                valid_actions.append(action)
            if len(valid_actions) >= self.expansion_width:
                break
        return valid_actions

    def _pick_untried_action(self, node: MCTSNode, actions: list[str]) -> str:
        for action in actions:
            if action not in node.children:
                return action
        return actions[0]

    def _simulate(self, action: str, env, obs_preprocessor=None, goal: str | None = None):
        try:
            obs, reward, terminated, truncated, _ = env.step(action)
        except Exception as exc:
            logging.warning(f"MCTS simulate failed for action {action}: {exc}")
            return -1.0, None, None

        processed_obs = obs_preprocessor(obs) if obs_preprocessor else obs
        value = reward if reward is not None else 0.0
        value += self._evaluate(processed_obs, goal)

        if not terminated and not truncated:
            self._backtrack(env)

        return value, processed_obs, reward

    def _backtrack(self, env) -> None:
        if not self._is_action_valid("go_back()"):
            return
        try:
            env.step("go_back()")
        except Exception as exc:
            logging.info(f"MCTS backtrack failed: {exc}")

    def _evaluate(self, obs: dict, goal: str | None) -> float:
        obs_text = self._obs_to_text(obs)
        prompt = (
            "Score how close the agent is to completing the goal. "
            "Return a single number between 0 and 1.\n\n"
            f"Goal:\n{goal or ''}\n\n"
            f"Observation:\n{obs_text}\n"
        )
        messages = Discussion([SystemMessage("You are a strict evaluator."), HumanMessage(prompt)])
        response = self.chat_llm(messages)
        score = self._parse_float(response["content"])
        return max(0.0, min(1.0, score))

    def _obs_to_text(self, obs: dict) -> str:
        if not isinstance(obs, dict):
            return str(obs)[: self.max_obs_chars]
        parts = []
        for key, val in obs.items():
            if isinstance(val, str):
                parts.append(f"[{key}]\n{val}")
        if not parts:
            return str(obs)[: self.max_obs_chars]
        text = "\n\n".join(parts)
        return text[: self.max_obs_chars]

    def _parse_json_list(self, text: str) -> list[str]:
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
        except json.JSONDecodeError:
            pass
        match = re.search(r"\[(.*)\]", text, re.DOTALL)
        if not match:
            return []
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
        except json.JSONDecodeError:
            return []
        return []

    def _parse_float(self, text: str) -> float:
        match = re.search(r"[-+]?[0-9]*\.?[0-9]+", text)
        if not match:
            return 0.0
        try:
            return float(match.group(0))
        except ValueError:
            return 0.0

    def _is_action_valid(self, action: str) -> bool:
        try:
            self.action_set.to_python_code(action)
            return True
        except Exception:
            return False
