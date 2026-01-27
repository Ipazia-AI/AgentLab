import logging
from dataclasses import dataclass
from datetime import datetime

from agentlab.agents.agentq.evaluators import AbsoluteCritic, TournamentCritic
from agentlab.agents.agentq.mcts import MCTS
from agentlab.agents.agentq.selectors import AheadKSelector, MaxVisitSelector
from agentlab.agents.generic_agent.generic_agent import (
    AgentInfo,
    GenericAgent,
    GenericAgentArgs,
)
from agentlab.agents.generic_agent.generic_agent_prompt import GenericPromptFlags
from agentlab.llm.chat_api import BaseModelArgs
from agentlab.llm.llm_utils import AIMessage, Discussion, SystemMessage
from agentlab.llm.tracking import cost_tracker_decorator

logger = logging.getLogger(__name__)


@dataclass
class AgentQArgs(GenericAgentArgs):
    """
    AgentQ configuration with mixed Q-value support (Agent Q paper Eq. 10).
    """

    critique_model_args: BaseModelArgs | None = None
    mcts_budget: int = 5
    mcts_max_workers: int = 4
    mcts_rollout_depth: int = 3
    critic_type: str = "tournament"  # "absolute" | "tournament"
    selection_strategy: str = "max_visit"  # "max_visit" | "ahead_k"
    ahead_k: int = 1
    use_real_rollouts: bool = False
    use_env_reward: bool = False  # True = paper-faithful sparse (0/1), False = LLM critique
    # Mixed Q-value parameters (Eq. 10)
    alpha: float = 0.5  # Q = α*Q̃ + (1-α)*Q̂ | 0=critic only, 1=MCTS only
    q_threshold: float = 0.1  # Min Q-gap for DPO pairs (θ_threshold)
    action_timeout: int = 10  # Timeout per action in seconds
    timeout_penalty: float = 0.2  # Penalty multiplier for repeated timeouts
    sync_mcts: bool = False  # Run MCTS iterations synchronously (debug-friendly)
    iteration_timeout: float | None = None  # Timeout per MCTS iteration (seconds)
    mcts_debug_logging: bool = False  # Enable DEBUG logs for MCTS
    browser_fork_logging: bool = False  # Enable browser_forking INFO logs

    def __post_init__(self):
        """Override parent to set correct agent name."""
        try:
            self.agent_name = f"AgentQ-{self.chat_model_args.model_name}".replace("/", "_")
        except AttributeError:
            pass

    def make_agent(self):
        return AgentQ(
            chat_model_args=self.chat_model_args,
            critique_model_args=self.critique_model_args,
            flags=self.flags,
            max_retry=self.max_retry,
            mcts_budget=self.mcts_budget,
            mcts_max_workers=self.mcts_max_workers,
            mcts_rollout_depth=self.mcts_rollout_depth,
            critic_type=self.critic_type,
            selection_strategy=self.selection_strategy,
            ahead_k=self.ahead_k,
            use_real_rollouts=self.use_real_rollouts,
            use_env_reward=self.use_env_reward,
            alpha=self.alpha,
            q_threshold=self.q_threshold,
            action_timeout=self.action_timeout,
            timeout_penalty=self.timeout_penalty,
            sync_mcts=self.sync_mcts,
            iteration_timeout=self.iteration_timeout,
            mcts_debug_logging=self.mcts_debug_logging,
            browser_fork_logging=self.browser_fork_logging,
        )

    def prepare(self):
        self.chat_model_args.prepare_server()
        if self.critique_model_args:
            self.critique_model_args.prepare_server()

    def close(self):
        self.chat_model_args.close_server()
        if self.critique_model_args:
            self.critique_model_args.close_server()


class AgentQ(GenericAgent):
    """
    AgentQ: MCTS-based web agent with mixed Q-value support.

    Implements the Agent Q paper (arXiv 2408.07199) approach:
    - Tournament-style action ranking (AI process supervision)
    - Mixed Q-value: Q = α*Q̃ + (1-α)*Q̂ (Eq. 10)
    - Optional paper-faithful sparse rewards (use_env_reward=True)
    - In-context DPO learning from preference pairs
    """

    def __init__(
        self,
        chat_model_args: BaseModelArgs,
        critique_model_args: BaseModelArgs | None,
        flags: GenericPromptFlags,
        max_retry: int = 4,
        mcts_budget: int = 5,
        mcts_max_workers: int = 4,
        mcts_rollout_depth: int = 3,
        critic_type: str = "tournament",
        selection_strategy: str = "max_visit",
        ahead_k: int = 1,
        use_real_rollouts: bool = False,
        use_env_reward: bool = False,
        alpha: float = 0.5,
        q_threshold: float = 0.1,
        action_timeout: int = 10,
        timeout_penalty: float = 0.2,
        sync_mcts: bool = False,
        iteration_timeout: float | None = None,
        mcts_debug_logging: bool = False,
        browser_fork_logging: bool = False,
    ):
        super().__init__(chat_model_args, flags, max_retry)
        self.critique_llm = (
            critique_model_args.make_model() if critique_model_args else None
        )
        self.mcts_budget = mcts_budget
        self.mcts_max_workers = mcts_max_workers
        self.mcts_rollout_depth = mcts_rollout_depth
        self.use_real_rollouts = use_real_rollouts
        self.use_env_reward = use_env_reward
        self.alpha = alpha
        self.q_threshold = q_threshold
        self.action_timeout = action_timeout
        self.timeout_penalty = timeout_penalty
        self.sync_mcts = sync_mcts
        self.iteration_timeout = iteration_timeout
        self.mcts_debug_logging = mcts_debug_logging
        self.browser_fork_logging = browser_fork_logging

        # Instantiate modular components
        critic = (
            TournamentCritic(self.chat_llm)
            if critic_type == "tournament"
            else AbsoluteCritic(self.chat_llm)
        )
        selector = (
            AheadKSelector(k=ahead_k) if selection_strategy == "ahead_k" else MaxVisitSelector()
        )

        # Initialize MCTS engine
        self.mcts = MCTS(
            chat_llm=self.chat_llm,
            critique_llm=self.critique_llm,
            action_set=self.action_set,
            flags=self.flags,
            rollout_depth=self.mcts_rollout_depth,
            critic=critic,
            selector=selector,
            use_real_rollouts=use_real_rollouts,
            use_env_reward=use_env_reward,
            alpha=alpha,
            q_threshold=q_threshold,
            action_timeout=action_timeout,
            timeout_penalty=timeout_penalty,
            sync_mcts=sync_mcts,
            iteration_timeout=iteration_timeout,
            debug_logging=mcts_debug_logging,
        )

        if not browser_fork_logging:
            logging.getLogger("agentlab.agents.browser_forking").setLevel(logging.WARNING)

        # Buffer for in-context DPO learning (preference pairs from tree)
        self.dpo_pairs = []
        self.max_dpo_pairs = 20

    def reset(self, seed=None):
        """Reset agent state, including DPO pairs buffer."""
        super().reset(seed)
        # Clear DPO pairs when starting a new episode
        self.dpo_pairs = []
        logger.debug("AgentQ reset: cleared DPO pairs buffer")

    @cost_tracker_decorator
    def get_action(self, obs):
        if isinstance(obs, tuple):
            obs = obs[0]

        # Preprocess obs (add pruned_html, axtree_txt, etc.)
        try:
            obs = self.obs_preprocessor(obs)
        except Exception:
            # Fallback for mock/incomplete observations
            if "pruned_html" not in obs:
                obs["pruned_html"] = obs.get("dom_txt", "HTML missing")
            if "axtree_txt" not in obs:
                obs["axtree_txt"] = "AXTree missing"

        # Update history
        self.obs_history.append(obs)

        # Extract Goal
        goal = self._extract_goal(obs)

        # Convert action history to strings
        history_strings = [a if isinstance(a, str) else a.get("text", str(a)) for a in self.actions]

        # Run MCTS Search with DPO pairs for in-context learning
        logger.info(
            f"AgentQ | Starting MCTS search | Budget: {self.mcts_budget} | Workers: {self.mcts_max_workers} | DPO pairs: {len(self.dpo_pairs)} | Time: {datetime.now().strftime('%H:%M:%S')}"
        )
        best_action, root_node = self.mcts.search(
            root_obs=obs,
            root_history=history_strings,
            goal=goal,
            budget=self.mcts_budget,
            dpo_pairs=self.dpo_pairs,
            max_workers=self.mcts_max_workers,
        )

        if not best_action:
            # Fallback to standard GenericAgent behavior
            logger.warning(
                f"AgentQ | MCTS returned no action, falling back to GenericAgent | Time: {datetime.now().strftime('%H:%M:%S')}"
            )
            # Remove the obs we already added, since GenericAgent.get_action will add it again
            self.obs_history.pop()
            # Pass original obs (not preprocessed) to GenericAgent
            # GenericAgent will handle preprocessing internally
            if isinstance(obs, tuple):
                original_obs = obs[0] if len(obs) > 0 else {}
            else:
                # Try to get original obs before preprocessing
                # If we can't, just pass it - GenericAgent will handle it
                original_obs = obs
            return super().get_action(original_obs)

        # Generate DPO preference pairs from the search tree
        new_pairs = self.mcts.generate_dpo_pairs(root_node, goal)
        if new_pairs:
            self.dpo_pairs.extend(new_pairs)
            # Keep buffer size manageable
            if len(self.dpo_pairs) > self.max_dpo_pairs:
                self.dpo_pairs = self.dpo_pairs[-self.max_dpo_pairs :]
            logger.info(
                f"AgentQ | Generated {len(new_pairs)} DPO pairs | Buffer size: {len(self.dpo_pairs)} | Time: {datetime.now().strftime('%H:%M:%S')}"
            )

        # Update agent state
        self.actions.append(best_action)
        thought = f"MCTS Selected: {best_action}"
        self.thoughts.append(thought)
        self.memories.append(None)

        # Build chat messages for browsergym chat interface
        # Include goal and action history for user visibility
        chat_messages = Discussion()

        # Add system message with goal
        system_content = (
            f"Goal: {goal}\n\nYou are using MCTS (Monte Carlo Tree Search) to select actions."
        )
        chat_messages.add_message(SystemMessage(system_content))

        # Add user message with current observation context (if available)
        if obs.get("chat_messages"):
            # Browsergym chat_messages use format: {'role': str, 'message': str, 'timestamp': float}
            # Convert to standard format
            for msg in obs["chat_messages"]:
                if isinstance(msg, dict):
                    role = msg.get("role", "user")
                    # Handle browsergym format (uses 'message') vs standard format (uses 'content')
                    content = msg.get("content") or msg.get("message", "")
                    if content:
                        chat_messages.add_message({"role": role, "content": content})
        else:
            # Otherwise, create a user message with goal
            chat_messages.add_message({"role": "user", "content": f"Task: {goal}"})

        # Add assistant message with the selected action and reasoning
        assistant_content = f"Action: {best_action}\n\nReasoning: {thought}"
        if len(self.actions) > 1:
            assistant_content += (
                f"\n\nPrevious actions: {', '.join(str(a) for a in self.actions[:-1])}"
            )
        chat_messages.add_message(AIMessage(assistant_content))

        # Return format expected by BrowserGym/AgentLab
        agent_info = AgentInfo(
            think=thought,
            chat_messages=chat_messages,
            stats=self.chat_llm.get_stats(),
            extra_info={"mcts_budget": self.mcts_budget, "dpo_pairs_count": len(self.dpo_pairs)},
        )
        return best_action, agent_info

    def _extract_goal(self, obs: dict) -> str:
        """Extract goal string from observation."""
        if "goal" in obs and obs["goal"]:
            return obs["goal"]

        if "goal_object" in obs:
            g_obj = obs["goal_object"]
            if isinstance(g_obj, tuple):
                g_obj = g_obj[0]
            if isinstance(g_obj, dict):
                return g_obj.get("text", "Complete the task.")
            return str(g_obj)

        return "Complete the task."
