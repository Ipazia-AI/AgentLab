import hashlib
import logging
import math
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from browsergym.core.action.base import AbstractActionSet

from agentlab.agents import dynamic_prompting as dp
from agentlab.agents.agentq.evaluators import (
    AbsoluteCritic,
    BaseCritic,
    TournamentCritic,
)
from agentlab.agents.agentq.selectors import ActionSelector, MaxVisitSelector
from agentlab.agents.browser_forking import (
    browser_fork_and_rollout,
    execute_action_in_fork,
)
from agentlab.agents.generic_agent.generic_agent_prompt import GenericPromptFlags

logger = logging.getLogger(__name__)


@dataclass
class MCTSNode:
    """
    MCTS tree node with mixed Q-value support (Agent Q paper Eq. 10).

    Q(h_t, a) = α * Q̃(h_t, a) + (1-α) * Q̂(h_t, a)

    Where:
        Q̃ = empirical MCTS value (value/visits from backpropagation)
        Q̂ = AI critic ranking value (fast_reward from tournament critic)
    """

    obs: dict[str, Any]
    history: list[str]
    parent: Optional["MCTSNode"] = None
    action: Optional[str] = None  # Action that led to this node
    children: list["MCTSNode"] = field(default_factory=list)
    visits: int = 0
    value: float = 0.0  # Cumulative backprop reward Q̃ (outcome supervision)
    fast_reward: float = 0.0  # Critic ranking Q̂ (process supervision)
    is_terminal: Optional[bool] = None  # Cache for terminal judge results
    timeout_count: int = 0

    @property
    def q_empirical(self) -> float:
        """Q̃: Empirical MCTS value (average outcome reward)."""
        return self.value / self.visits if self.visits > 0 else 0.0

    @property
    def q_critic(self) -> float:
        """Q̂: AI critic ranking value (process supervision)."""
        return self.fast_reward

    def q_mixed(self, alpha: float = 0.5) -> float:
        """
        Mixed Q-value combining MCTS exploration with AI process supervision.

        From Agent Q paper Equation 10:
            Q(h_t, a) = α * Q̃(h_t, a) + (1-α) * Q̂(h_t, a)

        Args:
            alpha: Weight for empirical MCTS value (0-1).
                   Higher alpha → trust MCTS more.
                   Lower alpha → trust critic more.

        For unvisited nodes (visits=0), returns pure critic value Q̂.
        """
        if self.visits == 0:
            return self.q_critic
        return alpha * self.q_empirical + (1 - alpha) * self.q_critic

    def ucb1(self, exploration_constant: float = 1.41, alpha: float = 0.5) -> float:
        """
        UCB1 selection using mixed Q-value.

        UCB = Q_mixed + c * sqrt(ln(N_parent) / N_child)
        """
        q = self.q_mixed(alpha)

        if self.visits == 0:
            # High exploration bonus for unvisited nodes
            return q + exploration_constant * 10

        if self.parent and self.parent.visits > 0:
            exploration = exploration_constant * math.sqrt(
                math.log(self.parent.visits) / self.visits
            )
            return q + exploration

        return q


class MCTS:
    """
    Monte Carlo Tree Search for web agent action selection.

    Implements Agent Q paper's mixed Q-value (Eq. 10):
        Q(h_t, a) = α * Q̃(h_t, a) + (1-α) * Q̂(h_t, a)

    Where:
        Q̃ = empirical MCTS value from backpropagation (outcome supervision)
        Q̂ = AI critic ranking value (process supervision)
        α = mixing coefficient (alpha parameter)
    """

    def __init__(
        self,
        chat_llm,
        critique_llm,
        action_set: AbstractActionSet,
        flags: GenericPromptFlags,
        headless: bool = True,
        rollout_depth: int = 3,
        critic: Optional[BaseCritic] = None,
        selector: Optional[ActionSelector] = None,
        use_real_rollouts: bool = False,
        use_env_reward: bool = False,
        alpha: float = 0.5,
        q_threshold: float = 0.1,
        action_timeout: int = 10,
        timeout_penalty: float = 0.2,
        sync_mcts: bool = False,
        iteration_timeout: Optional[float] = None,
        debug_logging: bool = False,
    ):
        """
        Initialize MCTS with mixed Q-value support.

        Args:
            chat_llm: LLM for action generation
            critique_llm: LLM for critique/evaluation
            action_set: BrowserGym action set
            flags: Prompt flags
            headless: Run browser in headless mode
            rollout_depth: Depth of rollout simulations
            critic: Critic for action evaluation (default: AbsoluteCritic)
            selector: Action selector (default: MaxVisitSelector)
            use_real_rollouts: If True, execute all rollout actions in real browser.
            use_env_reward: If True, use sparse environment reward (0/1) instead of
                           LLM critique scores. More faithful to Agent Q paper.
            alpha: Mixing coefficient for Q-value (Eq. 10).
                   0.0 = pure critic (process supervision)
                   1.0 = pure MCTS (outcome supervision)
                   0.5 = equal mix (default, as in paper)
            q_threshold: Minimum Q-value gap for DPO pair generation (θ_threshold in paper).
            action_timeout: Timeout in seconds for single action execution (default: 10).
        """
        self.chat_llm = chat_llm
        self.critique_llm = critique_llm
        self.action_set = action_set
        self.flags = flags
        self.action_flags = flags.action
        self.obs_flags = flags.obs
        self.headless = headless
        self.rollout_depth = rollout_depth
        self.use_real_rollouts = use_real_rollouts
        self.use_env_reward = use_env_reward
        self.alpha = alpha
        self.q_threshold = q_threshold
        self.action_timeout = action_timeout
        self.timeout_penalty = timeout_penalty
        self.action_timeouts: dict[str, int] = {}
        self.sync_mcts = sync_mcts
        self.iteration_timeout = iteration_timeout
        self.debug_logging = debug_logging

        # Modular Evaluator/Selector
        self.critic = critic or AbsoluteCritic(critique_llm)
        self.selector = selector or MaxVisitSelector()
        self.tree_lock = threading.Lock()
        self._cache_lock = threading.Lock()
        self._obs_prompt_cache: dict[str, str] = {}
        self._terminal_cache: dict[tuple[str, str], bool] = {}
        self._critic_cache: dict[tuple[str, str, str], float] = {}
        self._rank_cache: dict[tuple[str, str, tuple[str, ...]], list[tuple[str, float]]] = {}
        self._timing_enabled = os.environ.get("AGENTQ_MCTS_TIMING") == "1" or debug_logging
        if debug_logging:
            logging.getLogger(__name__).setLevel(logging.DEBUG)

    def _log_timing(self, label: str, start_time: float) -> None:
        if not self._timing_enabled:
            return
        elapsed = time.perf_counter() - start_time
        logger.debug("MCTS | Timing | %s: %.3fs", label, elapsed)

    def _obs_cache_key(self, obs: Optional[dict]) -> str:
        if not obs:
            return "no_obs"
        url = obs.get("url", "")
        titles = obs.get("open_pages_titles", [])
        title = titles[0] if titles else ""
        focused = obs.get("focused_element_bid", "")
        error = obs.get("last_action_error", "")
        axtree = obs.get("axtree_txt", "")[:2000]
        pruned_html = obs.get("pruned_html", "")[:2000]
        raw = "\n".join([url, title, str(focused), str(error), axtree, pruned_html])
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _get_obs_prompt_cached(self, obs: Optional[dict]) -> str:
        cache_key = self._obs_cache_key(obs)
        with self._cache_lock:
            cached = self._obs_prompt_cache.get(cache_key)
        if cached is not None:
            return cached
        prompt = dp.Observation(obs or {}, self.obs_flags).prompt
        with self._cache_lock:
            self._obs_prompt_cache[cache_key] = prompt
        return prompt

    def search(
        self,
        root_obs,
        root_history,
        goal: str,
        budget: int = 5,
        dpo_pairs: list[dict] = None,
        max_workers: int = 4,
    ):
        """
        Run MCTS search with parallel iterations.
        """
        logger.info(
            f"MCTS Search started | Budget: {budget} | Max workers: {max_workers} | Goal: {goal[:60]}... | Time: {datetime.now().strftime('%H:%M:%S')}"
        )

        root = MCTSNode(obs=root_obs, history=root_history)
        root.visits = 1

        if self.sync_mcts or os.environ.get("AGENTQ_SYNC_MCTS") == "1":
            for i in range(budget):
                logger.info(
                    "MCTS | Running iteration %s/%s synchronously | Time: %s",
                    i + 1,
                    budget,
                    datetime.now().strftime("%H:%M:%S"),
                )
                try:
                    self.run_iteration(root, goal, dpo_pairs, None, None)
                except Exception as e:
                    logger.error(
                        "MCTS | Iteration %s/%s failed: %s | Time: %s",
                        i + 1,
                        budget,
                        e,
                        datetime.now().strftime("%H:%M:%S"),
                        exc_info=True,
                    )
        else:
            sim_workers = int(os.environ.get("AGENTQ_MCTS_SIM_WORKERS", str(max_workers)))
            if sim_workers < 1:
                sim_workers = 1
            with ThreadPoolExecutor(max_workers=max_workers) as executor, ThreadPoolExecutor(
                max_workers=sim_workers
            ) as sim_executor:
                # Run iterations in parallel
                # Note: For small budgets, running them all at once is fine.
                # Tree updates are protected by tree_lock.
                futures = []
                for i in range(budget):
                    logger.info(
                        f"MCTS | Submitting iteration {i+1}/{budget} to thread pool | Time: {datetime.now().strftime('%H:%M:%S')}"
                    )
                    futures.append(
                        executor.submit(
                            self.run_iteration, root, goal, dpo_pairs, executor, sim_executor
                        )
                    )

                # Wait for all iterations to complete
                completed = 0
                if self.iteration_timeout is None:
                    timeout = None
                elif self.iteration_timeout <= 0:
                    timeout = None
                else:
                    timeout = float(self.iteration_timeout)
                heartbeat_interval = float(os.environ.get("AGENTQ_MCTS_HEARTBEAT_SECONDS", "30"))
                if timeout is None:
                    logger.warning(
                        "MCTS | Iteration timeout disabled; long-running iterations may block completion"
                    )
                for i, future in enumerate(futures, 1):
                    try:
                        if timeout is None:
                            heartbeat_start = time.perf_counter()
                            while True:
                                try:
                                    future.result(timeout=heartbeat_interval)
                                    break
                                except TimeoutError:
                                    elapsed = time.perf_counter() - heartbeat_start
                                    logger.debug(
                                        "MCTS | Heartbeat | Iteration %s/%s still running after %.1fs | Time: %s",
                                        i,
                                        budget,
                                        elapsed,
                                        datetime.now().strftime("%H:%M:%S"),
                                    )
                        else:
                            future.result(timeout=timeout)
                        completed += 1
                        logger.debug(
                            f"MCTS | Iteration {i}/{budget} completed | Time: {datetime.now().strftime('%H:%M:%S')}"
                        )
                    except TimeoutError:
                        logger.error(
                            "MCTS | Iteration %s/%s timed out after %.2fs | Time: %s",
                            i,
                            budget,
                            timeout,
                            datetime.now().strftime("%H:%M:%S"),
                        )
                        future.cancel()
                    except Exception as e:
                        logger.error(
                            f"MCTS | Iteration {i}/{budget} failed: {e} | Time: {datetime.now().strftime('%H:%M:%S')}",
                            exc_info=True,
                        )

                logger.info(
                    f"MCTS | All {completed}/{budget} iterations completed | Time: {datetime.now().strftime('%H:%M:%S')}"
                )

        # Select best action
        best_child = self.selector.select(root)
        if not best_child and root.children:
            # Filter empty actions (should not happen with proper generation)
            valid_children = [c for c in root.children if c.action and c.action.strip()]
            if valid_children:
                best_child = max(valid_children, key=lambda c: c.visits)
            else:
                best_child = max(root.children, key=lambda c: c.visits)
                logger.warning(
                    f"MCTS | No valid children, using fallback | Children: {len(root.children)}"
                )

        if best_child:
            # Final validation of selected action
            if not best_child.action or not best_child.action.strip():
                logger.warning(
                    f"MCTS | Selected child has empty action, returning None | Time: {datetime.now().strftime('%H:%M:%S')}"
                )
                return None, root
            avg_value = best_child.value / best_child.visits if best_child.visits > 0 else 0
            logger.info(
                f"MCTS | Best action selected: {best_child.action} | Value: {avg_value:.3f} | Visits: {best_child.visits} | Time: {datetime.now().strftime('%H:%M:%S')}"
            )
            return best_child.action, root

        logger.warning(
            f"MCTS | No best child found, returning None | Time: {datetime.now().strftime('%H:%M:%S')}"
        )
        return None, root

    def run_iteration(
        self,
        root: MCTSNode,
        goal: str,
        dpo_pairs: list[dict],
        executor: Optional[ThreadPoolExecutor],
        sim_executor: Optional[ThreadPoolExecutor],
    ):
        """A single MCTS iteration with proper virtual loss handling."""
        thread_id = threading.current_thread().name
        virtual_loss_applied = False
        leaf = None

        try:
            # 1. Selection (Thread-safe) with virtual loss
            with self.tree_lock:
                leaf = self.select(root)
                leaf.visits += 1  # Virtual loss
                virtual_loss_applied = True

            # 2. Check terminal (thread-safe caching inside)
            if self._is_terminal_cached(leaf, goal):
                # Terminal node: just backpropagate reward
                with self.tree_lock:
                    self._backprop_with_virtual_loss(leaf, 1.0 if leaf.is_terminal else 0.0)
                return

            # 3. Expansion (contains LLM calls, outside lock)
            with self.tree_lock:
                needs_expansion = not leaf.children

            if needs_expansion:
                self.expand(leaf, goal, dpo_pairs=dpo_pairs)

            # 4. Simulation
            with self.tree_lock:
                unvisited = [c for c in leaf.children if c.visits == 0]

            if unvisited:
                # Parallel simulation of children
                if sim_executor is None:
                    logger.debug("MCTS | Simulate children sequentially (no sim executor)")
                    for child in unvisited:
                        res_obs, reward = self.simulate(child, goal, self.rollout_depth)
                        with self.tree_lock:
                            child.obs = (
                                res_obs if res_obs else (child.parent.obs if child.parent else None)
                            )
                            self.backpropagate(child, reward)
                else:
                    logger.debug(
                        "MCTS | Simulate children in parallel | Children: %s",
                        len(unvisited),
                    )
                    self._simulate_children(unvisited, goal, sim_executor)
                # Virtual loss cleanup for leaf (children handle their own backprop)
                with self.tree_lock:
                    leaf.visits -= 1
                    virtual_loss_applied = False
            else:
                # Single simulation for this leaf
                res_obs, reward = self.simulate(leaf, goal, self.rollout_depth)
                with self.tree_lock:
                    if res_obs is not None:
                        leaf.obs = res_obs
                    self._backprop_with_virtual_loss(leaf, reward)
                    virtual_loss_applied = False

        except Exception as e:
            logger.warning(f"MCTS iteration failed [{thread_id}]: {e}")
        finally:
            # Always cleanup virtual loss if still applied
            if virtual_loss_applied and leaf is not None:
                with self.tree_lock:
                    leaf.visits -= 1

    def _simulate_children(self, children: list[MCTSNode], goal: str, executor: ThreadPoolExecutor):
        """Simulate unvisited children in parallel."""
        futures = {executor.submit(self.simulate, c, goal, self.rollout_depth): c for c in children}

        for future in futures:
            child = futures[future]
            try:
                res_obs, reward = future.result()
                with self.tree_lock:
                    child.obs = res_obs if res_obs else (child.parent.obs if child.parent else None)
                    self.backpropagate(child, reward)
            except Exception as e:
                logger.warning(f"Child simulation failed for {child.action}: {e}")
                with self.tree_lock:
                    child.obs = child.parent.obs if child.parent else None
                    self.backpropagate(child, child.fast_reward * 0.3)

    def _backprop_with_virtual_loss(self, node: MCTSNode, reward: float):
        """Backpropagate reward and account for virtual loss (visits already incremented)."""
        node.value += reward
        curr = node.parent
        while curr:
            curr.visits += 1
            curr.value += reward
            curr = curr.parent

    def _is_terminal_cached(self, node: MCTSNode, goal: str) -> bool:
        """Thread-safe terminal check with caching."""
        with self.tree_lock:
            if node.is_terminal is not None:
                return node.is_terminal
        obs_key = self._obs_cache_key(node.obs)
        cache_key = (goal, obs_key)
        with self._cache_lock:
            cached = self._terminal_cache.get(cache_key)
        if cached is not None:
            with self.tree_lock:
                node.is_terminal = cached
            return cached

        # Compute outside lock (may involve LLM call)
        result = self._compute_is_terminal(node, goal)

        with self.tree_lock:
            node.is_terminal = result
        return result

    def _compute_is_terminal(self, node: MCTSNode, goal: str) -> bool:
        """Compute terminal status (without caching)."""
        if node.obs is None:
            return False

        # Check environment reward first (fast path)
        if node.obs.get("metadata", {}).get("reward", 0.0) >= 1.0:
            return True

        # LLM judge (expensive)
        from agentlab.agents.agentq.prompts import TerminalJudgePrompt

        start_time = time.perf_counter()
        obs_prompt = self._get_obs_prompt_cached(node.obs)
        judge = TerminalJudgePrompt(
            goal=goal, obs_summary=obs_prompt, screenshot=node.obs.get("screenshot")
        )

        try:
            response = self.critique_llm(judge.to_messages())
            result = judge.parse_answer(str(response))
            self._log_timing("terminal_judge", start_time)
            obs_key = self._obs_cache_key(node.obs)
            with self._cache_lock:
                self._terminal_cache[(goal, obs_key)] = result
            return result
        except Exception as e:
            logger.warning(f"Terminal judge failed: {e}")
            return False

    def select(self, node: MCTSNode) -> MCTSNode:
        """
        Traverse tree selecting node with max UCB (using mixed Q-value) until a leaf.

        Uses alpha-weighted Q-value: Q = α*Q̃ + (1-α)*Q̂
        For unvisited nodes, Q̂ (critic prior) dominates.
        """
        curr = node
        while curr.children:
            # If any child is unvisited, select based on mixed Q (which equals Q̂ for visits=0)
            unvisited = [c for c in curr.children if c.visits == 0]
            if unvisited:
                return max(unvisited, key=self._score_child)

            # All children visited: use UCB1 with mixed Q
            curr = max(curr.children, key=self._score_child)
        return curr

    def expand(self, node: MCTSNode, goal: str, dpo_pairs: list[dict] = None):
        """
        Generate K candidate actions and rank them using the critic.
        Assigns fast_reward based on tournament ranking.
        """
        # Determine which observation to use for ranking
        if node.obs is not None:
            obs_to_use = node.obs
        elif node.parent and node.parent.obs is not None:
            # Use parent's obs if current node doesn't have one (e.g., action failed)
            obs_to_use = node.parent.obs
            logger.debug(f"expand() using parent obs for node with action: {node.action}")
        else:
            # No observation available at all - this shouldn't happen for root, but handle gracefully
            logger.warning(
                "expand() called on node with None obs and no parent obs, cannot rank actions"
            )
            # Try to generate actions anyway, but with low priority
            actions = self.generate_candidate_actions(node, goal, n=3, dpo_pairs=dpo_pairs)
            if not actions:
                return
            # Assign low equal ranks since we can't properly evaluate
            for action in actions:
                child = MCTSNode(
                    obs=None,
                    history=node.history + [action],
                    parent=node,
                    action=action,
                    fast_reward=0.1,  # Lower priority when we can't evaluate
                )
                node.children.append(child)
            return

        # 1. Generate K candidate actions using the Actor
        actions_start = time.perf_counter()
        actions = self.generate_candidate_actions(node, goal, n=3, dpo_pairs=dpo_pairs)
        self._log_timing("action_generation", actions_start)

        # Filter empty actions (generate_candidate_actions already validates)
        valid_actions = [a for a in actions if a and a.strip()]
        if not valid_actions:
            logger.warning("No valid actions generated for expansion")
            return

        # 2. Rank using critic (tournament or absolute scoring)
        if isinstance(self.critic, TournamentCritic) and len(valid_actions) > 1:
            # Tournament: batch ranking in single LLM call
            obs_key = self._obs_cache_key(obs_to_use)
            rank_key = (goal, obs_key, tuple(valid_actions))
            with self._cache_lock:
                cached_ranked = self._rank_cache.get(rank_key)
            if cached_ranked is not None:
                ranked = cached_ranked
            else:
                rank_start = time.perf_counter()
                ranked = self.critic.rank_actions_tournament(
                    goal, obs_to_use, valid_actions, self.obs_flags
                )
                self._log_timing("critic_rank_tournament", rank_start)
                with self._cache_lock:
                    self._rank_cache[rank_key] = ranked
        else:
            # Absolute scoring: evaluate each action independently
            obs_key = self._obs_cache_key(obs_to_use)
            ranked = []
            for action in valid_actions:
                cache_key = (goal, obs_key, action)
                with self._cache_lock:
                    cached_score = self._critic_cache.get(cache_key)
                if cached_score is None:
                    eval_start = time.perf_counter()
                    cached_score = self.critic.evaluate(
                        goal, action, obs_to_use, self.obs_flags
                    )
                    self._log_timing("critic_eval_absolute", eval_start)
                    with self._cache_lock:
                        self._critic_cache[cache_key] = cached_score
                ranked.append((action, cached_score))
            ranked.sort(key=lambda x: x[1], reverse=True)

        # 3. Create children with fast_reward from ranking
        # Skip any actions that are still invalid (shouldn't happen, but safety check)
        for action, rank in ranked:
            if not action or not action.strip():
                logger.warning("Skipping empty action in expansion")
                continue
            timeout_count = self.action_timeouts.get(action, 0)
            child = MCTSNode(
                obs=None,
                history=node.history + [action],
                parent=node,
                action=action,
                fast_reward=rank,
                timeout_count=timeout_count,
            )
            node.children.append(child)

    def generate_candidate_actions(
        self, node: MCTSNode, goal: str, n=3, dpo_pairs: list[dict] = None
    ) -> list[str]:
        """Generate K distinct candidate actions using the Actor LLM."""
        from agentlab.llm.llm_utils import extract_html_tags

        # Get observation (fallback to parent's or minimal)
        obs_to_use = (
            node.obs
            or (node.parent.obs if node.parent else None)
            or {
                "url": "",
                "axtree_txt": "No observation available",
                "pruned_html": "",
                "dom_txt": "",
                "last_action_error": None,
                "focused_element_bid": None,
            }
        )

        # Build prompts
        obs_prompt = self._get_obs_prompt_cached(obs_to_use)
        action_prompt = dp.ActionPrompt(self.action_set, self.action_flags)

        # DPO context (last 3 pairs)
        dpo_context = ""
        if dpo_pairs:
            examples = [
                f"State: {p.get('state_summary', '')[:150]}...\n"
                f"GOOD: {p.get('chosen', '')} | BAD: {p.get('rejected', '')}"
                for p in dpo_pairs[-3:]
            ]
            dpo_context = "## Learn from examples:\n" + "\n".join(examples) + "\n"

        prompt = f"""# Goal: {goal}
{f"## Extra: {self.flags.extra_instructions}" if self.flags.extra_instructions else ""}

# History: {node.history[-3:] if node.history else "None"}

{dpo_context}
{obs_prompt}

{action_prompt.prompt}

Suggest {n} distinct valid actions. Each action MUST be wrapped in <action> tags:
<action>
function_name('element_id')
</action>
"""
        try:
            llm_start = time.perf_counter()
            response = str(self.chat_llm([{"role": "user", "content": prompt}]))
            self._log_timing("actor_llm", llm_start)

            # Parse using agentlab's standard HTML tag extraction
            parsed = extract_html_tags(response, keys=["action"])
            actions = parsed.get("action", [])

            # Validate: must be function calls (word followed by parens)
            valid = [a.strip() for a in actions if a and a.strip() and "(" in a and ")" in a]

            if not valid:
                logger.warning(f"No valid actions parsed: {response[:200]}")
                return [self.action_set.example_action(abstract=False)]

            return valid[:n]

        except Exception as e:
            logger.warning(f"Action generation failed: {e}")
            return [self.action_set.example_action(abstract=False)]

    def simulate(self, node: MCTSNode, goal: str, rollout_depth: int = 3) -> tuple[dict, float]:
        """
        Execute the action leading to 'node' and perform rollout.

        Supports two modes controlled by self.use_real_rollouts:
        - False (default): Execute first action, then use fast_rewards for rollout
        - True: Execute all rollout actions in real browser (expensive but accurate)

        Return (first_step_obs, estimated_reward).
        """
        if not node.parent or not node.parent.obs:
            return None, 0.0

        parent_obs = node.parent.obs
        action = node.action

        if not action:
            return None, 0.0

        if self.use_real_rollouts:
            return self._simulate_with_real_rollouts(node, parent_obs, action, goal, rollout_depth)
        else:
            return self._simulate_with_fast_rewards(node, parent_obs, action, goal, rollout_depth)

    def _simulate_with_fast_rewards(
        self, node: MCTSNode, parent_obs: dict, action: str, goal: str, rollout_depth: int
    ) -> tuple[dict, float]:
        """
        Fast simulation: execute first action, then use fast_rewards for rollout.

        When use_env_reward=True: Returns pure environment reward (0/1), no imagined rollout.
        When use_env_reward=False: Uses LLM estimates for rollout (original behavior).
        """
        # Step 1: Execute the FIRST action in browser
        first_obs, first_reward, error = self._execute_single_action(parent_obs, action, goal)
        node.obs = first_obs

        if first_obs is None:
            if error and "timeout" in error.lower():
                self._record_timeout(action)
                node.timeout_count += 1
            logger.debug(f"Action execution failed for {action}")
            if node.parent and node.parent.obs:
                node.obs = node.parent.obs
            return node.parent.obs if node.parent else None, node.fast_reward * 0.5

        # If using environment reward, return immediately (no imagined rollout)
        # This is paper-faithful: Q̃ comes purely from real outcomes
        if self.use_env_reward:
            return first_obs, first_reward

        # Below: LLM-estimated rollout (only when use_env_reward=False)
        if first_reward >= 1.0:
            return first_obs, 1.0

        # Lightweight rollout using fast_rewards (no browser execution)
        cumulative = first_reward
        current_node = node

        for step in range(rollout_depth):
            if not current_node.children:
                obs_for_expand = current_node.obs or (
                    current_node.parent.obs if current_node.parent else None
                )
                if obs_for_expand:
                    current_node.obs = obs_for_expand
                    self.expand(current_node, goal)
                else:
                    break

            if not current_node.children:
                break

            best_child = max(current_node.children, key=lambda c: c.fast_reward)
            cumulative = max(cumulative, best_child.fast_reward)
            current_node = best_child

        # Terminal check only for high-reward paths
        if cumulative >= 0.8 and self.is_terminal_obs(first_obs, goal):
            return first_obs, 1.0

        return first_obs, cumulative

    def _simulate_with_real_rollouts(
        self, node: MCTSNode, parent_obs: dict, action: str, goal: str, rollout_depth: int
    ) -> tuple[dict, float]:
        """
        Full simulation: execute all actions in real browser.
        Paper-faithful when use_env_reward=True.
        """

        def generate_action(obs: dict, goal: str) -> list[str]:
            temp_node = MCTSNode(obs=obs, history=[])
            return self.generate_candidate_actions(temp_node, goal, n=1)

        # Critique function: returns env reward if use_env_reward, else LLM score
        def critique_action(
            goal: str, obs_summary: str, action: str, obs: dict, pre_summary: str, error: str
        ) -> dict:
            if self.use_env_reward:
                env_reward = obs.get("metadata", {}).get("reward", 0.0)
                return {"score": float(env_reward >= 1.0)}
            return self.critique(goal, obs_summary, action, obs, pre_summary, error)

        def check_terminal(obs: dict, goal: str) -> bool:
            if self.use_env_reward:
                # Paper-faithful: terminal when env says done
                return obs.get("metadata", {}).get("reward", 0.0) >= 1.0
            return self.is_terminal_obs(obs, goal)

        # Shorter timeout for faster failure detection
        rollout_timeout = self.action_timeout * (rollout_depth + 1) * 2

        first_obs, cumulative_reward = browser_fork_and_rollout(
            start_obs=parent_obs,
            initial_action=action,
            action_set=self.action_set,
            obs_flags=self.obs_flags,
            goal=goal,
            rollout_depth=rollout_depth,
            action_generator=generate_action,
            critique_fn=critique_action,
            terminal_fn=check_terminal,
            headless=self.headless,
            timeout=rollout_timeout,
        )

        if first_obs is None:
            self._record_timeout(action)
            node.timeout_count += 1
        node.obs = first_obs
        return first_obs, cumulative_reward

    def _execute_single_action(
        self, start_obs: dict, action: str, goal: str
    ) -> tuple[Optional[dict], float, Optional[str]]:
        """
        Execute a single action in a forked browser and return (new_obs, reward).

        Reward source depends on use_env_reward:
        - True: Sparse environment reward (0/1) from obs metadata (paper-faithful)
        - False: LLM critique score (0.0-1.0) for dense feedback
        """
        exec_start = time.perf_counter()
        new_obs, error = execute_action_in_fork(
            start_obs=start_obs,
            action=action,
            action_set=self.action_set,
            obs_flags=self.obs_flags,
            headless=self.headless,
            timeout=self.action_timeout,
        )
        self._log_timing("env_action_execute", exec_start)

        if new_obs is None or error:
            logger.warning(f"Action execution failed: {error}")
            return None, 0.0, error

        if self.use_env_reward:
            # Paper-faithful: use sparse environment reward (0 or 1)
            env_reward = new_obs.get("metadata", {}).get("reward", 0.0)
            return new_obs, float(env_reward >= 1.0), None
        else:
            # Dense feedback: use LLM critique score
            pre_obs_summary = self._get_obs_prompt_cached(start_obs)
            critique_start = time.perf_counter()
            critique_result = self.critique(
                goal,
                self._get_obs_prompt_cached(new_obs),
                action,
                new_obs,
                pre_obs_summary=pre_obs_summary,
                action_error=error,
            )
            self._log_timing("critique_llm", critique_start)
            return new_obs, critique_result.get("score", 0.5), None

    def is_terminal_obs(self, obs: dict, goal: str) -> bool:
        """Check if observation represents terminal state (without node caching)."""
        temp_node = MCTSNode(obs=obs, history=[])
        return self._compute_is_terminal(temp_node, goal)

    def critique(
        self, goal, obs_summary, action, obs=None, pre_obs_summary=None, action_error=None
    ):
        """
        Use the modular critic to evaluate.
        """
        try:
            score = self.critic.evaluate(
                goal, action, obs, self.obs_flags, pre_obs_summary, action_error
            )
            # For backward compatibility with logs, we return a dict
            return {"score": score, "reasoning": "Modular Critic evaluation"}
        except Exception as e:
            logger.warning(f"Modular Critique failed: {e}")
            return {"score": 0.5, "reasoning": f"Critique failed: {e}"}

    def backpropagate(self, node: MCTSNode, reward: float):
        """Standard backpropagation: increment visits and accumulate reward up to root."""
        curr = node
        while curr:
            curr.visits += 1
            curr.value += reward
            curr = curr.parent

    def _record_timeout(self, action: str) -> None:
        if not action:
            return
        self.action_timeouts[action] = self.action_timeouts.get(action, 0) + 1

    def _score_child(self, child: MCTSNode) -> float:
        base = child.ucb1(alpha=self.alpha)
        penalty = self.timeout_penalty * child.timeout_count
        return base - penalty

    def generate_dpo_pairs(self, root: MCTSNode, goal: str = "") -> list[dict]:
        """
        Extract (state, chosen, rejected) preference pairs from the MCTS tree.

        Uses mixed Q-value (Eq. 10) for scoring and applies threshold filtering:
            |Q(h_t, a^w) - Q(h_t, a^l)| > θ_threshold

        Args:
            root: Root node of the MCTS tree
            goal: Task goal string

        Returns:
            List of preference pairs with mixed Q-scores
        """
        pairs = []
        alpha = self.alpha
        threshold = self.q_threshold

        def collect_from_node(node: MCTSNode):
            if not node.children:
                return

            # Only consider children that have been visited
            visited = [c for c in node.children if c.visits > 0]
            if len(visited) < 2:
                for child in visited:
                    collect_from_node(child)
                return

            # Sort by mixed Q-value (descending)
            visited_sorted = sorted(visited, key=lambda c: c.q_mixed(alpha), reverse=True)
            best = visited_sorted[0]

            # Create pairs: best vs each other (with threshold filtering)
            for other in visited_sorted[1:]:
                q_best = best.q_mixed(alpha)
                q_other = other.q_mixed(alpha)
                q_gap = abs(q_best - q_other)

                # Apply threshold filter (θ_threshold from paper Algorithm 1)
                if q_gap >= threshold:
                    pairs.append(
                        {
                            "state_summary": self._summarize_state(node.obs),
                            "goal": goal,
                            "chosen": best.action,
                            "chosen_score": q_best,  # Mixed Q-value
                            "rejected": other.action,
                            "rejected_score": q_other,  # Mixed Q-value
                            "q_gap": q_gap,  # For debugging/analysis
                        }
                    )

            # Recurse down the best path
            collect_from_node(best)

        collect_from_node(root)
        return pairs

    def _summarize_state(self, obs: dict) -> str:
        """Create a compact summary of the observation for DPO context."""
        if obs is None:
            return "No observation"

        parts = []

        # URL is most important
        url = obs.get("url", "")
        if url:
            parts.append(f"URL: {url}")

        # Page title if available
        titles = obs.get("open_pages_titles", [])
        if titles:
            parts.append(f"Title: {titles[0]}")

        # Truncated AXTree or DOM
        axtree = obs.get("axtree_txt", "")
        if axtree:
            # Take first 300 chars of AXTree
            parts.append(f"Page: {axtree[:300]}...")
        elif "pruned_html" in obs:
            parts.append(f"HTML: {obs['pruned_html'][:300]}...")

        return " | ".join(parts) if parts else "Empty state"
