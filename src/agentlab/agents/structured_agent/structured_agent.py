import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

from agentlab.agents import dynamic_prompting as dp
from agentlab.agents.generic_agent.generic_agent import GenericAgent, GenericAgentArgs
from agentlab.agents.generic_agent.generic_agent_prompt import GenericPromptFlags, MainPrompt
from agentlab.llm.base_api import BaseModelArgs
from agentlab.llm.llm_utils import Discussion, ParseError, SystemMessage, retry
from agentlab.agents.structured_agent.new_hpa import HPA, HPAHandler, HPANode, NodeStatus, NodeType
from agentlab.experiments.loop import StepInfo, _send_chat_info

try:
    from browsergym.core.chat import Chat
except ImportError:
    Chat = None

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
        """Run one episode using HPA search (goal + stack). Each step is executed via perform_action. Single search call to avoid infinite loop when action is None."""
        self.set_env(env)
        self._exp_dir = Path(exp_dir)
        self._save_screenshot = save_screenshot
        self._save_som = save_som
        self._episode_info = []
        self.reset(seed=seed)
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
            while not step_info.is_done:
                self.hpa.search(root, goal_str)
                step_info = self._episode_info[-1]
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
            # Same step semantics as ExpArgs.run(): current step has full obs (incl. goal_object) for get_action/MainPrompt, then action, agent_info
            step_num = len(self._episode_info)
            if step_num == 0:
                step_info = StepInfo(step=step_num)
            else:
                step_info = self._episode_info[-1]
            # Use copy so step_info.obs has full obs (goal_object etc.); _current_obs is kept intact across save_step_info (see below)
            step_info.obs = (
                self._current_obs.copy() if isinstance(self._current_obs, dict) else self._current_obs
            )

            # Agent phase: same as step_info.from_action(agent) for xray (profiling + chat_messages)
            step_info.profiling.agent_start = time.time()
            action, agent_info = self._select_action(node)
            step_info.profiling.agent_stop = time.time()
            step_info.action = action
            step_info.agent_info = agent_info
            step_info.make_stats()

            if action is None:
                step_info.truncated = True

            # self._episode_info.append(step_info)
            if self._exp_dir is not None:
                step_info.save_step_info(
                    self._exp_dir,
                    save_screenshot=self._save_screenshot,
                    save_som=self._save_som,
                )

            # Send think + action to chat for agent-xray (same as loop)
            if Chat is not None and hasattr(self._env.unwrapped, "chat") and isinstance(self._env.unwrapped.chat, Chat):
                _send_chat_info(self._env.unwrapped.chat, action, step_info.agent_info)

            if action is None:
                logger.warning("_select_action returned None; skipping step.")
                return False

            # Env phase: use from_step so profiling and obs are recorded like in the loop
            next_step = StepInfo(step=step_info.step + 1)
            self._episode_info.append(next_step)
            next_step.from_step(self._env, action, self.obs_preprocessor)

            # Copy obs before save_step_info: save_step_info sets obs["goal_object"] = None for storage,
            # so we must keep a copy for _current_obs (and MainPrompt) with goal_object intact (same as standard loop passing obs.copy() to get_action before save).
            self._current_obs = (
                next_step.obs.copy() if isinstance(next_step.obs, dict) else next_step.obs
            )
            node.metadata.update({
                "obs": next_step.obs,
                "reward": next_step.reward,
                "terminated": next_step.terminated,
                "truncated": next_step.truncated,
            })

            if self._exp_dir is not None:
                next_step.save_step_info(
                    self._exp_dir,
                    save_screenshot=self._save_screenshot,
                    save_som=self._save_som,
                )

            # Success = action was executed and episode is not done (same idea as loop: keep going while not is_done)
            return not next_step.is_done
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
    def _empty_agent_info(self) -> dict:
        """Minimal agent_info dict for xray when no action is selected."""
        return {
            "think": None,
            "chat_messages": [],
            "stats": {},
            "extra_info": {"chat_model_args": asdict(self.chat_model_args)},
        }

    def _call_llm_simple(self, prompt: str) -> str:
        """Single user-message call; returns assistant content."""
        messages = [{"role": "user", "content": prompt}]
        res = self.chat_llm(messages)
        content = res.get("content", "") if isinstance(res, dict) else getattr(res, "content", str(res))
        return content or ""

    def _select_action(self, node: HPANode) -> Tuple[Optional[str], dict]:
        """Mirror of GenericAgent.get_action: same state (obs_history, actions, memories, thoughts, plan), same prompt and retry flow; returns (action, agent_info) for xray."""
        if self._current_obs is None:
            return None, self._empty_agent_info()
        self.obs_history.append(self._current_obs)
        main_prompt = MainPrompt(
            action_set=self.action_set,
            obs_history=self.obs_history,
            actions=self.actions,
            memories=self.memories,
            thoughts=self.thoughts,
            previous_plan=self.plan,
            step=self.plan_step,
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
            ans_dict["busted_retry"] = 0
            ans_dict["n_retry"] = (len(chat_messages) - 3) / 2
        except ParseError as e:
            logger.warning("_select_action parse failed: %s", e)
            ans_dict = dict(
                action=None,
                n_retry=self.max_retry + 1,
                busted_retry=1,
            )
        stats = getattr(self.chat_llm, "get_stats", lambda: {})()
        stats["n_retry"] = ans_dict["n_retry"]
        stats["busted_retry"] = ans_dict.get("busted_retry", 0)
        if ans_dict.get("action") is not None:
            self.plan = ans_dict.get("plan", self.plan)
            self.plan_step = ans_dict.get("step", self.plan_step)
            self.actions.append(ans_dict["action"])
            self.memories.append(ans_dict.get("memory", None))
            self.thoughts.append(ans_dict.get("think", None))
        agent_info = {
            "think": ans_dict.get("think"),
            "chat_messages": chat_messages,
            "stats": stats,
            "extra_info": {"chat_model_args": asdict(self.chat_model_args)},
        }
        return ans_dict.get("action"), agent_info