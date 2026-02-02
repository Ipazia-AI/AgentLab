import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List, Optional

from agentlab.agents import dynamic_prompting as dp
from agentlab.agents.generic_agent.generic_agent import GenericAgent, GenericAgentArgs
from agentlab.agents.generic_agent.generic_agent_prompt import GenericPromptFlags, MainPrompt
from agentlab.llm.base_api import BaseModelArgs
from agentlab.llm.llm_utils import Discussion, ParseError, SystemMessage, retry
from agentlab.agents.structured_agent.new_hpa import HPA, HPAHandler, HPANode, NodeStatus, NodeType
from agentlab.experiments.loop import StepInfo

logger = logging.getLogger(__name__)


@dataclass
class StructuredAgentArgs(GenericAgentArgs):
    def __post_init__(self):
        try:
            self.agent_name = f"StructuredAgent-{self.chat_model_args.model_name}".replace("/", "_")
        except AttributeError:
            pass

    def make_agent(self):
        return StructuredAgent(self.chat_model_args, self.flags, self.max_retry)
    
    def prepare(self):
        self.chat_model_args.prepare_server()

    def close(self):
        self.chat_model_args.close_server()


class StructuredAgent(GenericAgent):
    def __init__(
        self,
        chat_model_args: BaseModelArgs,
        flags: GenericPromptFlags,
        max_retry: int = 4,
    ):
        super().__init__(chat_model_args, flags, max_retry)
        self.hpa = HPA(handler=self, budget=500)
        self._env = None
        self._episode_info: List[StepInfo] = []
        self._current_obs: Optional[dict] = None
        self._exp_dir: Optional[Path] = None
        self._save_screenshot = True
        self._save_som = False

    def set_env(self, env: Any) -> None:
        """Inject the environment so HPA handler can call env.step during search."""
        self._env = env

    @property
    def env(self):
        return self._env

    # -------- Public API --------
    def run(self, root: HPANode, goal: Optional[str] = None):
        return self.hpa.search(root, goal)

    def run_episode(
        self,
        env: Any,
        exp_dir: Path,
        seed: int,
        obs_preprocessor: Optional[Callable[[dict], dict]] = None,
        save_screenshot: bool = True,
        save_som: bool = False,
    ) -> List[StepInfo]:
        """Run one episode using HPA's internal loop. Called by ExpArgs.run() when present."""
        self.set_env(env)
        self._exp_dir = Path(exp_dir)
        self._save_screenshot = save_screenshot
        self._save_som = save_som
        self._episode_info = []

        obs_preprocessor = obs_preprocessor or (lambda x: x)
        step_info = StepInfo(step=0)
        step_info.from_reset(env, seed=seed, obs_preprocessor=obs_preprocessor)
        self._episode_info.append(step_info)
        self._current_obs = step_info.obs

        task_info = getattr(env, "task", None) or {}
        if isinstance(task_info, dict):
            goal_str = task_info.get("goal", task_info.get("task_name", "Complete the task"))
        else:
            goal_str = str(task_info) if task_info else "Complete the task"

        root = HPANode()
        try:
            self.hpa.search(root, goal_str)
        except Exception as e:
            logger.warning("HPA search raised: %s", e)
            if self._episode_info:
                last = self._episode_info[-1]
                last.agent_info["err_msg"] = str(e)

        return self._episode_info


    # -------- HPA Handler Hooks --------
    def set_context(self, node: HPANode):
        # AXTree focus / frame setup could go here
        pass


    def rollback_context(self, node: HPANode) -> None:
        if self._env is None:
            return
        try:
            self._env.step("go_back()")
        except Exception:
            pass

    def populate_node_type(self, node: HPANode, goal: Optional[str]) -> NodeType:
        """LLM decides node type."""
        prompt = f"Classify node as ACTION, AND, or OR. Goal: {goal}"
        text = self._call_llm_simple(prompt)
        decision = text.strip().upper() if text else ""
        return getattr(NodeType, decision, NodeType.ACTION)


    def perform_action(self, node: HPANode) -> bool:
        if self._env is None or self._current_obs is None:
            return False
        try:
            action = self._select_action(node)
            if action is None:
                logger.warning("_select_action returned None; skipping step.")
                return False
            obs, reward, terminated, truncated, env_info = self._env.step(action)
            preprocessed_obs = self.obs_preprocessor(obs) if obs is not None else obs
            self._current_obs = preprocessed_obs
            node.metadata.update({
                "obs": obs,
                "reward": reward,
                "terminated": terminated,
                "truncated": truncated,
                "env_info": env_info,
            })
            step_num = len(self._episode_info)
            step_info = StepInfo(step=step_num, action=action)
            step_info.obs = preprocessed_obs
            step_info.reward = reward if reward is not None else 0
            step_info.raw_reward = env_info.get("RAW_REWARD_GLOBAL") if env_info else None
            step_info.terminated = terminated
            step_info.truncated = truncated
            step_info.task_info = env_info.get("task_info") if env_info else None
            step_info.agent_info = {}
            if env_info:
                step_info.profiling.env_start = env_info.get("action_exec_start", 0)
                step_info.profiling.env_stop = env_info.get("action_exec_stop", 0)
            step_info.make_stats()
            self._episode_info.append(step_info)
            if self._exp_dir is not None:
                step_info.save_step_info(
                    self._exp_dir,
                    save_screenshot=self._save_screenshot,
                    save_som=self._save_som,
                )
            return reward is not None and reward > 0
        except Exception as e:
            logger.warning("perform_action failed: %s", e)
            return False


    def update_after_action(self, node: HPANode):
        # Update memory, notes, AXTree, etc.
        pass


    def select_promising_child(self, node: HPANode) -> HPANode:
        # Simple heuristic: first non-closed child
        for c in node.children:
            if c.status not in {NodeStatus.PRUNED, NodeStatus.DELETED}:
                return c
        return node.children[0]


    def revise_and(self, node: HPANode) -> bool:
        # LLM-based repair of AND node
        return False


    def revise_or(self, node: HPANode) -> bool:
        # LLM-based expansion of OR alternatives
        return False


    # -------- Internal Helpers --------
    def _call_llm_simple(self, prompt: str) -> str:
        """Single user-message call; returns assistant content."""
        messages = [{"role": "user", "content": prompt}]
        res = self.chat_llm(messages)
        content = res.get("content", "") if isinstance(res, dict) else getattr(res, "content", str(res))
        return content or ""

    def _select_action(self, node: HPANode) -> Optional[str]:
        """Select one env-executable action using the same MainPrompt + retry/parse flow as GenericAgent.get_action."""
        if self._current_obs is None:
            return None
        obs_history = [s.obs for s in self._episode_info if s.obs is not None]
        if not obs_history:
            obs_history = [self._current_obs]
        actions = [s.action for s in self._episode_info if getattr(s, "action", None) is not None]
        main_prompt = MainPrompt(
            action_set=self.action_set,
            obs_history=obs_history,
            actions=actions,
            memories=[],
            thoughts=[],
            previous_plan="",
            step=max(0, len(self._episode_info) - 1),
            flags=self.flags,
        )
        max_prompt_tokens, max_trunc_itr = self._get_maxes()
        system_prompt = SystemMessage(dp.SystemPrompt().prompt)
        human_prompt = dp.fit_tokens(
            shrinkable=main_prompt,
            max_prompt_tokens=max_prompt_tokens,
            model_name=self.chat_model_args.model_name,
            max_iterations=max_trunc_itr,
            additional_prompts=system_prompt,
        )
        chat_messages = Discussion([system_prompt, human_prompt])
        try:
            ans_dict = retry(
                self.chat_llm,
                chat_messages,
                n_retry=self.max_retry,
                parser=main_prompt._parse_answer,
            )
            return ans_dict.get("action")
        except ParseError as e:
            logger.warning("_select_action parse failed: %s", e)
            return None