"""Telos host agent: plan with Telos, ground actions for BrowserGym."""

from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from bgym import Benchmark, HighLevelActionSetArgs
from browsergym.experiments.agent import Agent, AgentInfo
from telos import ActionNode, Result, Status, Telos
from telos.errors import PlannerError

from agentlab.agents import dynamic_prompting as dp
from agentlab.agents.agent_args import AgentArgs
from agentlab.agents.telos_agent.actor import Actor
from agentlab.agents.telos_agent.obs import (
    goal_from_obs,
    make_preprocessor,
    observation_to_text,
)
from agentlab.agents.telos_agent.planner_llm import AgentLabPlannerLLM
from agentlab.agents.telos_agent.telemetry import (
    build_telos_plan,
    format_telos_plan_markdown,
    make_recording_planner,
)
from agentlab.llm.tracking import LLMTracker, cost_tracker_decorator, set_tracker

if TYPE_CHECKING:
    from agentlab.llm.chat_api import BaseModelArgs


def _default_obs_flags() -> dp.ObsFlags:
    return dp.ObsFlags(
        use_html=False,
        use_ax_tree=True,
        use_focused_element=True,
        use_error_logs=True,
        use_screenshot=False,
        use_som=False,
        extract_visible_tag=True,
        extract_clickable_tag=True,
        extract_coords="False",
        filter_visible_elements_only=False,
    )


@dataclass
class TelosAgentArgs(AgentArgs):
    planner_model_args: "BaseModelArgs" = None
    actor_model_args: "BaseModelArgs" = None
    horizon_T: int = 30
    max_retry: int = 4
    obs_flags: dp.ObsFlags = None
    action_set_args: HighLevelActionSetArgs = None

    def __post_init__(self):
        if self.obs_flags is None:
            self.obs_flags = _default_obs_flags()
        if self.action_set_args is None:
            self.action_set_args = HighLevelActionSetArgs(subsets=["bid"], multiaction=False)
        try:
            planner = self.planner_model_args.model_name.replace("/", "_")
            actor_args = self.actor_model_args or self.planner_model_args
            actor = actor_args.model_name.replace("/", "_")
            self.agent_name = f"TelosAgent-{planner}-{actor}"
        except AttributeError:
            self.agent_name = "TelosAgent"

    def set_benchmark(self, benchmark: Benchmark, demo_mode: bool):
        if benchmark.name.startswith("miniwob"):
            self.obs_flags.use_html = True
        self.obs_flags.use_tabs = benchmark.is_multi_tab
        self.action_set_args = deepcopy(benchmark.high_level_action_set_args)
        if demo_mode:
            self.action_set_args.demo_mode = "all_blue"

    def set_reproducibility_mode(self):
        self.planner_model_args.temperature = 0
        if self.actor_model_args is not None:
            self.actor_model_args.temperature = 0

    def prepare(self):
        self.planner_model_args.prepare_server()
        if (
            self.actor_model_args is not None
            and self.actor_model_args is not self.planner_model_args
        ):
            self.actor_model_args.prepare_server()

    def close(self):
        self.planner_model_args.close_server()
        if (
            self.actor_model_args is not None
            and self.actor_model_args is not self.planner_model_args
        ):
            self.actor_model_args.close_server()

    def make_agent(self) -> Agent:
        return TelosAgent(
            planner_model_args=self.planner_model_args,
            actor_model_args=self.actor_model_args or self.planner_model_args,
            action_set_args=self.action_set_args,
            obs_flags=self.obs_flags,
            horizon_T=self.horizon_T,
            max_retry=self.max_retry,
        )


class TelosAgent(Agent):
    def __init__(
        self,
        planner_model_args: "BaseModelArgs",
        actor_model_args: "BaseModelArgs",
        action_set_args: HighLevelActionSetArgs | Any,
        obs_flags: dp.ObsFlags,
        horizon_T: int = 30,
        max_retry: int = 4,
        session: Any = None,
        action_set: Any = None,
    ):
        self.horizon_T = horizon_T
        self.max_retry = max_retry
        self.obs_flags = obs_flags
        self._obs_preprocessor = make_preprocessor(obs_flags)
        self.action_set = action_set if action_set is not None else action_set_args.make_action_set()
        self._planner_llm = AgentLabPlannerLLM(planner_model_args.make_model())
        self._recording_llm, self._recorder = make_recording_planner(self._planner_llm)
        self._actor = Actor(
            chat_model=actor_model_args.make_model(),
            action_set=self.action_set,
            max_retry=max_retry,
        )
        self._completed_steps: list[str] = []
        self._action_history = {
            "telos_high_level_action": [],
            "actor_grounded_action": [],
        }
        self._session = session

    def obs_preprocessor(self, obs: dict) -> dict:
        return self._obs_preprocessor(obs)

    @cost_tracker_decorator
    def get_action(self, obs: dict) -> tuple[str | None, AgentInfo]:
        observation = observation_to_text(obs, use_html=self.obs_flags.use_html)
        last_action_error = obs.get("last_action_error") or ""
        if self._session is None:
            self._session = Telos(
                planner_llm=self._recording_llm,
                planner=self._recorder,
                horizon_T=self.horizon_T,
                goal=goal_from_obs(obs),
                action_set=self.action_set.describe(
                    with_long_description=True, with_examples=True
                ),
            )

        with set_tracker(suffix="planner") as planner_tracker:
            outcome = self._step_with_retry(observation)
        if isinstance(outcome, Result) and outcome.status in (
            Status.HORIZON_REACHED,
            Status.FAILED,
        ):
            grounded = None
            actor_stats = LLMTracker("actor").stats
        else:
            with set_tracker(suffix="actor") as actor_tracker:
                if isinstance(outcome, Result) and outcome.status == Status.GOAL_REACHED:
                    outcome_description = "Goal reached."
                else:
                    outcome_description = outcome.description
                grounded = self._actor.ground(
                    instruction=outcome_description,
                    axtree_txt=obs.get("axtree_txt") or observation,
                    action_history=self._action_history,
                )
                if grounded.action is not None:
                    self._action_history["telos_high_level_action"].append(outcome_description)
                    self._action_history["actor_grounded_action"].append(grounded.action)
            actor_stats = actor_tracker.stats

        agent_info = self._agent_info(outcome, grounded, last_action_error)
        agent_info.stats.update(planner_tracker.stats)
        agent_info.stats.update(actor_stats)
        if isinstance(outcome, ActionNode):
            self._completed_steps.append(outcome.description)
        return None if grounded is None else grounded.action, agent_info

    def _step_with_retry(self, observation: str) -> ActionNode | Result:
        last_error = None
        for _ in range(self.max_retry):
            try:
                self._recorder.reset()
                return self._session.step(observation)
            except PlannerError as exc:
                last_error = exc
        raise last_error

    def _agent_info(
        self, outcome: ActionNode | Result, grounded, last_action_error: str
    ) -> AgentInfo:
        state = self._session.snapshot() if self._session is not None else None
        tree_view = state.tree.render_view() if state is not None else None
        is_result = isinstance(outcome, Result)
        think = (
            f"Telos terminal: {outcome.status.name}"
            if is_result
            else f"[{outcome.id}] {outcome.description}"
        )
        plan = build_telos_plan(
            state,
            self._recorder,
            outcome,
            grounded,
            last_action_error,
            self._completed_steps,
        )
        stats = {
            "telos_step": state.t if state is not None else 0,
            "actor_n_retry": grounded.n_retry if grounded else 0,
            "busted_retry": grounded.busted_retry if grounded else 0,
        }
        extra_info = {
            "node_id": None if is_result else outcome.id,
            "node_description": None if is_result else outcome.description,
            "terminal": outcome.status.name if is_result else None,
            "tree_view": tree_view,
            "actor_raw": grounded.raw if grounded else None,
            "telos_plan": plan,
        }
        return AgentInfo(
            think=think,
            chat_messages=grounded.chat_messages if grounded and grounded.chat_messages else [],
            stats=stats,
            markdown_page=format_telos_plan_markdown(plan),
            extra_info=extra_info,
        )
