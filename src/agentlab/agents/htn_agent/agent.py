from __future__ import annotations

from copy import copy
from dataclasses import dataclass, field
from typing import Any

import bgym
from bgym import Benchmark
from browsergym.core.observation import extract_screenshot
from browsergym.utils.obs import (
    flatten_axtree_to_str,
    flatten_dom_to_str,
    overlay_som,
    prune_html,
)
from browsergym.experiments.agent import Agent, AgentInfo

from agentlab.agents.agent_args import AgentArgs
from agentlab.llm.tracking import cost_tracker_decorator

from .belief import BeliefState
from .htn import HTNPlanner, Method, Operator, Skill, SkillResult, Task
from .skills import ExploreSkill, FillFieldSkill, FindAndClickSkill, SubmitSkill


@dataclass
class HTNObsConfig:
    use_dom: bool = True
    use_axtree: bool = True
    use_som: bool = False
    extract_visible_tag: bool = True
    extract_clickable_tag: bool = True
    extract_coords: str | bool = False
    filter_visible_elements_only: bool = True
    filter_with_bid_only: bool = False
    filter_som_only: bool = False


@dataclass
class HTNAgentArgs(AgentArgs):
    action_set: bgym.HighLevelActionSetArgs = None  # type: ignore
    obs_config: HTNObsConfig = field(default_factory=HTNObsConfig)

    def set_benchmark(self, benchmark: Benchmark, demo_mode: bool):
        self.action_set = benchmark.high_level_action_set_args
        if demo_mode and hasattr(self.action_set, "demo_mode"):
            self.action_set.demo_mode = "all_blue"

    def make_agent(self):
        return HTNToolUseAgent(action_set=self.action_set, obs_config=self.obs_config)


class HTNController:
    def __init__(self, planner: HTNPlanner):
        self.planner = planner
        self.task_stack: list[Task] = []
        self.current_skill: Skill | None = None
        self.current_task: Task | None = None

    def seed(self, goal_task: Task) -> None:
        self.task_stack = [goal_task]
        self.current_skill = None
        self.current_task = None

    def step(self, belief: BeliefState, obs: dict) -> SkillResult:
        if self.current_skill is None:
            operator = self.planner.next_operator(self.task_stack, belief)
            if operator is None:
                return SkillResult(action=None, done=True, info={"reason": "no_operator"})
            self.current_task = self.task_stack[-1]
            self.current_skill = operator.make_skill(self.current_task)
        result = self.current_skill.step(belief, obs)
        if result.done:
            self.current_skill = None
            if self.task_stack:
                self.task_stack.pop()
        return result


class HTNToolUseAgent(Agent):
    def __init__(
        self,
        action_set: bgym.HighLevelActionSetArgs | None = None,
        obs_config: HTNObsConfig | None = None,
    ):
        self.obs_config = obs_config or HTNObsConfig()
        self.action_set_args = action_set or bgym.HighLevelActionSetArgs(subsets=["bid"])
        self.action_set = self.action_set_args.make_action_set()
        self.belief = BeliefState()
        self.planner = build_default_planner()
        self.controller = HTNController(self.planner)
        self._initialized = False

    def obs_preprocessor(self, obs: dict) -> dict:
        obs = copy(obs)
        page = obs.pop("page", None)
        if page is not None:
            obs["screenshot"] = extract_screenshot(page)
        if self.obs_config.use_dom and obs.get("dom_object") is not None:
            obs["dom_txt"] = flatten_dom_to_str(
                obs["dom_object"],
                extra_properties=obs.get("extra_element_properties"),
                with_visible=self.obs_config.extract_visible_tag,
                with_clickable=self.obs_config.extract_clickable_tag,
                with_center_coords=self.obs_config.extract_coords == "center",
                with_bounding_box_coords=self.obs_config.extract_coords == "box",
                filter_visible_only=self.obs_config.filter_visible_elements_only,
                filter_with_bid_only=self.obs_config.filter_with_bid_only,
                filter_som_only=self.obs_config.filter_som_only,
            )
            obs["pruned_html"] = prune_html(obs["dom_txt"])
        if self.obs_config.use_axtree and obs.get("axtree_object") is not None:
            obs["axtree_txt"] = flatten_axtree_to_str(
                obs["axtree_object"],
                extra_properties=obs.get("extra_element_properties"),
                with_visible=self.obs_config.extract_visible_tag,
                with_clickable=self.obs_config.extract_clickable_tag,
                with_center_coords=self.obs_config.extract_coords == "center",
                with_bounding_box_coords=self.obs_config.extract_coords == "box",
                filter_visible_only=self.obs_config.filter_visible_elements_only,
                filter_with_bid_only=self.obs_config.filter_with_bid_only,
                filter_som_only=self.obs_config.filter_som_only,
            )
        if self.obs_config.use_som and obs.get("screenshot") is not None:
            obs["screenshot_som"] = overlay_som(
                obs["screenshot"], extra_properties=obs.get("extra_element_properties")
            )
        return obs

    @cost_tracker_decorator
    def get_action(self, obs: Any) -> tuple[str | None, AgentInfo]:
        if isinstance(obs, dict):
            raw_goal = obs.get("goal") or obs.get("goal_text")
            self.belief.ensure_goal(raw_goal)
            self.belief.update(obs)
        if not self._initialized:
            self.controller.seed(Task("SolveTask"))
            self._initialized = True
        result = self.controller.step(self.belief, obs)
        agent_info = AgentInfo(
            think=None,
            chat_messages=[],
            stats={},
            extra_info={
                "belief_summary": self.belief.summary(),
                "current_task": self.controller.current_task.name if self.controller.current_task else None,
                "current_skill": type(self.controller.current_skill).__name__
                if self.controller.current_skill
                else None,
                "skill_info": result.info,
            },
        )
        return result.action, agent_info


def build_default_planner() -> HTNPlanner:
    planner = HTNPlanner()

    def guard_create(belief: BeliefState) -> bool:
        return bool(belief.goal and belief.goal.intent == "create")

    def guard_update(belief: BeliefState) -> bool:
        return bool(belief.goal and belief.goal.intent == "update")

    def guard_default(_: BeliefState) -> bool:
        return True

    def decompose_create(_: BeliefState) -> list[Task]:
        return [
            Task("OpenCreateForm"),
            Task("FillGoalFields"),
            Task("SubmitForm"),
        ]

    def decompose_update(_: BeliefState) -> list[Task]:
        return [
            Task("SearchRecord"),
            Task("FillGoalFields"),
            Task("SubmitForm"),
        ]

    def decompose_default(_: BeliefState) -> list[Task]:
        return [Task("Explore")]

    planner.add_method(
        Method(
            name="SolveCreate",
            task="SolveTask",
            guard=guard_create,
            decompose=decompose_create,
        )
    )
    planner.add_method(
        Method(
            name="SolveUpdate",
            task="SolveTask",
            guard=guard_update,
            decompose=decompose_update,
        )
    )
    planner.add_method(
        Method(
            name="SolveDefault",
            task="SolveTask",
            guard=guard_default,
            decompose=decompose_default,
        )
    )

    planner.add_operator(
        Operator(
            name="OpenCreateForm",
            task="OpenCreateForm",
            make_skill=lambda _: FindAndClickSkill(role="button", name_hint="new"),
        )
    )
    planner.add_operator(
        Operator(
            name="SearchRecord",
            task="SearchRecord",
            make_skill=lambda _: FindAndClickSkill(role="input", name_hint="search"),
        )
    )
    planner.add_operator(
        Operator(
            name="FillGoalFields",
            task="FillGoalFields",
            make_skill=lambda task: _build_fill_skill(task),
        )
    )
    planner.add_operator(
        Operator(
            name="SubmitForm",
            task="SubmitForm",
            make_skill=lambda _: SubmitSkill(),
        )
    )
    planner.add_operator(
        Operator(
            name="Explore",
            task="Explore",
            make_skill=lambda _: ExploreSkill(),
        )
    )
    return planner


def _build_fill_skill(task: Task) -> Skill:
    if task.args:
        field, value = task.args
        return FillFieldSkill(field_name=field, value=value)
    return FillFieldSkill(field_name="description", value="")
