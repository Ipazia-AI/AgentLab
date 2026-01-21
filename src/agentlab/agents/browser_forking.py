"""
Browser Forking Utilities for AgentLab

This module provides reusable browser forking functionality that allows agents to:
- Fork browser state from an observation (URL, cookies, storage_state)
- Execute actions in isolated browser contexts
- Extract observations from forked browsers
- Perform rollouts with real browser execution

This is useful for:
- MCTS with real rollouts (measuring actual state vs predicting)
- Recursive task decomposition with state checkpoints
- Ablation studies comparing predicted vs measured rewards
- Any agentic exploration that needs true browser state

Usage:
    from agentlab.agents.browser_forking import BrowserFork, execute_action_in_fork
    
    # Simple single action execution
    new_obs, error = execute_action_in_fork(
        start_obs=current_obs,
        action="click('button[id=submit]')",
        action_set=my_action_set,
        obs_flags=my_obs_flags,
        headless=True
    )
    
    # Multi-step rollout
    with BrowserFork(start_obs, headless=True) as fork:
        for action in actions:
            obs, error = fork.execute(action, action_set)
            if error or fork.is_done:
                break
        final_obs = fork.current_obs
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from queue import Queue
from typing import Any, Callable, Optional

import browsergym.core.observation as bgym_obs
from browsergym.core.action.base import AbstractActionSet
from playwright.sync_api import Browser, BrowserContext, Page, sync_playwright

from agentlab.agents import dynamic_prompting as dp

logger = logging.getLogger(__name__)


@dataclass
class ForkResult:
    """Result of a browser fork operation."""
    obs: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    success: bool = False
    execution_time: float = 0.0


@dataclass
class RolloutStep:
    """A single step in a rollout."""
    action: str
    obs: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    reward: float = 0.0


@dataclass
class RolloutResult:
    """Result of a multi-step rollout."""
    steps: list[RolloutStep] = field(default_factory=list)
    final_obs: Optional[dict[str, Any]] = None
    cumulative_reward: float = 0.0
    terminated: bool = False
    success: bool = False


class BrowserFork:
    """
    Context manager for browser forking operations.
    
    Forks browser state from an observation and provides methods to execute
    actions and extract observations in the forked context.
    
    Example:
        with BrowserFork(start_obs, headless=True) as fork:
            obs, error = fork.execute(action, action_set)
            if not error:
                # Process observation
                pass
    """
    
    def __init__(
        self,
        start_obs: dict[str, Any],
        obs_flags: Optional[dp.ObsFlags] = None,
        headless: bool = True,
        timeout: int = 30000,
        slow_mo: int = 0,
    ):
        """
        Initialize browser fork.
        
        Args:
            start_obs: Observation dict containing url, storage_state, cookies
            obs_flags: Flags for observation preprocessing
            headless: Run browser in headless mode
            timeout: Default timeout for operations in ms
            slow_mo: Slow down operations by this many ms (useful for debugging)
        """
        self.start_obs = start_obs
        self.obs_flags = obs_flags
        self.headless = headless
        self.timeout = timeout
        self.slow_mo = slow_mo
        
        self._playwright = None
        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None
        self._page: Optional[Page] = None
        self._current_obs: Optional[dict] = None
        self._is_done = False

    def __enter__(self):
        """Start the forked browser session."""
        self._start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up browser resources."""
        self._cleanup()
        return False

    def _start(self):
        """Initialize playwright and navigate to starting state."""
        url = self.start_obs.get("url")
        if not url:
            raise ValueError("start_obs must contain 'url'")
        
        storage_state = self.start_obs.get("storage_state")
        
        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch(
            headless=self.headless,
            slow_mo=self.slow_mo
        )
        
        context_args = {}
        if storage_state:
            context_args["storage_state"] = storage_state
        
        self._context = self._browser.new_context(**context_args)
        # Use shorter timeout for faster failures (5s instead of 30s)
        navigation_timeout = min(self.timeout, 5000)  # Max 5s for navigation
        self._context.set_default_timeout(navigation_timeout)
        self._page = self._context.new_page()
        
        try:
            # Use "domcontentloaded" instead of "networkidle" for much faster navigation
            # networkidle waits for all network activity to stop, which never happens on Amazon
            # domcontentloaded fires when DOM is ready (2-5s vs 30s timeout)
            self._page.goto(url, timeout=navigation_timeout, wait_until="domcontentloaded")
            # Reduced settle time - only wait if page might still be loading
            # Most pages are ready after domcontentloaded
            time.sleep(0.1)  # Minimal settle time
            self._current_obs = self._extract_obs()
        except Exception as e:
            logger.error(f"Failed to navigate to {url}: {e}")
            self._cleanup()
            raise

    def _cleanup(self):
        """Clean up browser resources."""
        try:
            if self._browser:
                self._browser.close()
        except Exception as e:
            logger.warning(f"Error closing browser: {e}")
        
        try:
            if self._playwright:
                self._playwright.stop()
        except Exception as e:
            logger.warning(f"Error stopping playwright: {e}")
        
        self._browser = None
        self._context = None
        self._page = None
        self._playwright = None

    @property
    def page(self) -> Optional[Page]:
        """Access the underlying Playwright page."""
        return self._page

    @property
    def context(self) -> Optional[BrowserContext]:
        """Access the underlying browser context."""
        return self._context

    @property
    def current_obs(self) -> Optional[dict]:
        """Get the current observation."""
        return self._current_obs

    @property
    def is_done(self) -> bool:
        """Check if the fork session has terminated."""
        return self._is_done

    def execute(
        self,
        action: str,
        action_set: AbstractActionSet,
        wait_after: float = 0.1  # Reduced default from 0.5s for better performance
    ) -> tuple[Optional[dict], Optional[str]]:
        """
        Execute an action in the forked browser.
        
        Args:
            action: Action string to execute
            action_set: ActionSet to translate action to Python code
            wait_after: Seconds to wait after action execution (reduced default for performance)
            
        Returns:
            (observation, error) tuple. error is None on success.
        """
        if self._page is None:
            return None, "Browser fork not initialized"
        
        error = None
        try:
            # Pre-extract for BrowserGym
            bgym_obs._pre_extract(self._page)
            
            # Translate and execute action
            python_code = action_set.to_python_code(action)
            exec_context = {
                "page": self._page,
                "DEMO_MODE": False,
                "send_message_to_user": lambda x: logger.info(f"AGENT MESSAGE: {x}"),
                "report_infeasible_instructions": lambda x: logger.warning(f"INFEASIBLE: {x}")
            }
            exec(python_code, exec_context)
            
        except Exception as e:
            error = str(e)
            logger.warning(f"Action execution failed: {error}")
        
        # Smart waiting: only wait if action might trigger navigation
        # For non-navigation actions (click, fill), minimal wait is sufficient
        # For navigation actions, wait for DOM to be ready
        if wait_after > 0:
            # Check if action likely triggers navigation (goto, click link, etc.)
            is_navigation_action = any(keyword in action.lower() for keyword in ['goto', 'navigate', 'click'])
            if is_navigation_action:
                # For navigation, wait for DOM to be ready (faster than fixed sleep)
                try:
                    self._page.wait_for_load_state("domcontentloaded", timeout=2000)  # 2s max
                except Exception:
                    pass  # If timeout, continue anyway
                time.sleep(0.1)  # Minimal additional settle
            else:
                # For non-navigation actions, minimal wait
                time.sleep(0.1)
        
        # Extract new observation
        try:
            self._current_obs = self._extract_obs()
        except Exception as e:
            logger.error(f"Failed to extract observation: {e}")
            if error is None:
                error = f"Observation extraction failed: {e}"
        
        return self._current_obs, error

    def _extract_obs(self) -> dict[str, Any]:
        """Extract observation from current page state."""
        if self._page is None or self._context is None:
            return {}
        
        bgym_obs._pre_extract(self._page)
        dom_snapshot = bgym_obs.extract_dom_snapshot(self._page)
        axtree = bgym_obs.extract_merged_axtree(self._page)
        focused_bid = bgym_obs.extract_focused_element_bid(self._page)
        screenshot = bgym_obs.extract_screenshot(self._page)
        extra_props = bgym_obs.extract_dom_extra_properties(dom_snapshot, scale_factor=1)

        obs = {
            "url": self._page.url,
            "screenshot": screenshot,
            "dom_object": dom_snapshot,
            "axtree_object": axtree,
            "extra_element_properties": extra_props,
            "focused_element_bid": focused_bid,
            "open_pages_urls": [self._page.url],
            "open_pages_titles": [self._page.title()],
            "active_page_index": 0,
            "last_action_error": "",
            "cookies": self._context.cookies(),
            "storage_state": self._context.storage_state()
        }
        
        # Apply preprocessing if flags provided
        if self.obs_flags:
            preprocessor = dp.make_obs_preprocessor(self.obs_flags)
            obs = preprocessor(obs)
        
        return obs

    def get_storage_state(self) -> Optional[dict]:
        """Get current browser storage state (cookies, localStorage, etc.)."""
        if self._context is None:
            return None
        return self._context.storage_state()


def execute_action_in_fork(
    start_obs: dict[str, Any],
    action: str,
    action_set: AbstractActionSet,
    obs_flags: Optional[dp.ObsFlags] = None,
    headless: bool = True,
    timeout: int = 60,
) -> tuple[Optional[dict], Optional[str]]:
    """
    Execute a single action in a forked browser context.
    
    This is a convenience function that handles threading to avoid
    asyncio conflicts with Playwright.
    
    Args:
        start_obs: Starting observation with url, storage_state
        action: Action string to execute
        action_set: ActionSet for action translation
        obs_flags: Optional observation preprocessing flags
        headless: Run headless
        timeout: Timeout in seconds
        
    Returns:
        (new_observation, error) tuple
    """
    result_queue = Queue()
    
    def _execute():
        try:
            with BrowserFork(start_obs, obs_flags, headless) as fork:
                obs, error = fork.execute(action, action_set)
                result_queue.put((obs, error))
        except Exception as e:
            result_queue.put((None, str(e)))
    
    thread = threading.Thread(target=_execute)
    thread.daemon = True
    thread.start()
    thread.join(timeout=timeout)
    
    if thread.is_alive():
        logger.error(f"Fork execution timed out after {timeout}s")
        return None, "Timeout"
    
    if result_queue.empty():
        return None, "No result from fork"
    
    return result_queue.get()


def execute_rollout_in_fork(
    start_obs: dict[str, Any],
    actions: list[str],
    action_set: AbstractActionSet,
    obs_flags: Optional[dp.ObsFlags] = None,
    critique_fn: Optional[Callable[[dict, str], float]] = None,
    terminal_fn: Optional[Callable[[dict], bool]] = None,
    headless: bool = True,
    timeout: int = 120,
) -> RolloutResult:
    """
    Execute a multi-step rollout in a forked browser.
    
    Args:
        start_obs: Starting observation
        actions: List of actions to execute sequentially
        action_set: ActionSet for action translation
        obs_flags: Optional observation preprocessing flags
        critique_fn: Optional function (obs, action) -> reward to score each step
        terminal_fn: Optional function (obs) -> bool to check termination
        headless: Run headless
        timeout: Total timeout in seconds
        
    Returns:
        RolloutResult with steps, final observation, and cumulative reward
    """
    result_queue = Queue()
    
    def _rollout():
        result = RolloutResult()
        try:
            with BrowserFork(start_obs, obs_flags, headless) as fork:
                for action in actions:
                    step = RolloutStep(action=action)
                    
                    obs, error = fork.execute(action, action_set)
                    step.obs = obs
                    step.error = error
                    
                    # Compute reward if critique function provided
                    if critique_fn and obs:
                        try:
                            step.reward = critique_fn(obs, action)
                        except Exception as e:
                            logger.warning(f"Critique function failed: {e}")
                            step.reward = 0.0
                    
                    result.steps.append(step)
                    result.cumulative_reward = max(result.cumulative_reward, step.reward)
                    
                    # Check termination
                    if terminal_fn and obs:
                        try:
                            if terminal_fn(obs):
                                result.terminated = True
                                result.success = True
                                break
                        except Exception as e:
                            logger.warning(f"Terminal function failed: {e}")
                    
                    # Stop on error
                    if error:
                        break
                
                result.final_obs = fork.current_obs
                result.success = not any(s.error for s in result.steps)
                
        except Exception as e:
            logger.error(f"Rollout failed: {e}")
            result.steps.append(RolloutStep(action="", error=str(e)))
        
        result_queue.put(result)
    
    thread = threading.Thread(target=_rollout)
    thread.daemon = True
    thread.start()
    thread.join(timeout=timeout)
    
    if thread.is_alive():
        logger.error(f"Rollout timed out after {timeout}s")
        return RolloutResult(steps=[RolloutStep(action="", error="Timeout")])
    
    if result_queue.empty():
        return RolloutResult(steps=[RolloutStep(action="", error="No result")])
    
    return result_queue.get()


# Convenience function for compatibility with existing code
def browser_fork_and_rollout(
    start_obs: dict[str, Any],
    initial_action: str,
    action_set: AbstractActionSet,
    obs_flags: dp.ObsFlags,
    goal: str,
    rollout_depth: int = 3,
    action_generator: Optional[Callable[[dict, str], list[str]]] = None,
    critique_fn: Optional[Callable[[str, str, dict, str], dict]] = None,
    terminal_fn: Optional[Callable[[dict, str], bool]] = None,
    headless: bool = True,
    timeout: int = 120,
) -> tuple[Optional[dict], float]:
    """
    Execute initial action and perform rollout with action generation.
    
    This matches the original AgentQ browser_fork_and_act interface for compatibility.
    
    Args:
        start_obs: Starting observation
        initial_action: First action to execute
        action_set: ActionSet for action translation
        obs_flags: Observation flags
        goal: Task goal string
        rollout_depth: How many additional steps after initial action
        action_generator: Function (obs, goal) -> [actions] for rollout
        critique_fn: Function (goal, obs_summary, action, obs) -> {'score': float}
        terminal_fn: Function (obs, goal) -> bool
        headless: Run headless
        timeout: Total timeout
        
    Returns:
        (first_step_obs, cumulative_reward)
    """
    result_queue = Queue()
    
    def _execute():
        first_obs = None
        cumulative_reward = 0.0
        
        try:
            with BrowserFork(start_obs, obs_flags, headless) as fork:
                # Execute initial action
                current_obs, error = fork.execute(initial_action, action_set)
                first_obs = current_obs
                
                if error:
                    result_queue.put((first_obs, 0.0))
                    return
                
                # Critique initial action
                if critique_fn and current_obs:
                    obs_summary = dp.Observation(current_obs, obs_flags).prompt
                    pre_summary = dp.Observation(start_obs, obs_flags).prompt
                    critique_result = critique_fn(goal, obs_summary, initial_action, current_obs, pre_summary, error)
                    cumulative_reward = critique_result.get('score', 0.5)
                
                # Check terminal
                if terminal_fn and current_obs and terminal_fn(current_obs, goal):
                    result_queue.put((first_obs, 1.0))
                    return
                
                # Rollout
                for step in range(rollout_depth):
                    if action_generator is None:
                        break
                    
                    # Generate next action
                    try:
                        next_actions = action_generator(current_obs, goal)
                        if not next_actions:
                            break
                        next_action = next_actions[0]
                    except Exception as e:
                        logger.warning(f"Action generation failed: {e}")
                        break
                    
                    # Execute
                    pre_obs = current_obs
                    current_obs, error = fork.execute(next_action, action_set)
                    
                    if error:
                        break
                    
                    # Critique
                    if critique_fn and current_obs:
                        obs_summary = dp.Observation(current_obs, obs_flags).prompt
                        pre_summary = dp.Observation(pre_obs, obs_flags).prompt
                        critique_result = critique_fn(goal, obs_summary, next_action, current_obs, pre_summary, error)
                        step_reward = critique_result.get('score', 0.5)
                        cumulative_reward = max(cumulative_reward, step_reward)
                    
                    # Check terminal
                    if terminal_fn and current_obs and terminal_fn(current_obs, goal):
                        cumulative_reward = 1.0
                        break
                
                result_queue.put((first_obs, cumulative_reward))
                
        except Exception as e:
            logger.error(f"Browser fork rollout failed: {e}")
            result_queue.put((first_obs, cumulative_reward))
    
    thread = threading.Thread(target=_execute)
    thread.daemon = True
    thread.start()
    thread.join(timeout=timeout)
    
    if thread.is_alive():
        logger.error(f"Browser fork rollout timed out after {timeout}s")
        return None, 0.0
    
    if result_queue.empty():
        return None, 0.0
    
    return result_queue.get()
