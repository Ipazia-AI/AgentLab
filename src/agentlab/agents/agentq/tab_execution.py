"""
AgentQ tab execution helpers: one context per worker, new tab per simulation.

Self-contained: tab lifecycle and action execution + observation extraction.
MCTS uses create_shared_context, get_url_from_obs, execute_action_in_tab, and
browser_tab_and_rollout; the latter two use execute_action_on_page and
execute_action_on_existing_page internally.
"""

import json
import logging
import os
import tempfile
import threading
import time
from typing import Any, Callable, Optional

import browsergym.core.observation as bgym_obs
from browsergym.core.action.base import AbstractActionSet
from playwright.sync_api import BrowserContext, Page, sync_playwright

from agentlab.agents import dynamic_prompting as dp

logger = logging.getLogger(__name__)


def execute_action_on_page(
    page: Page,
    context: BrowserContext,
    action: str,
    action_set: AbstractActionSet,
    obs_flags: Optional[dp.ObsFlags] = None,
    wait_after: float = 0.1,
) -> tuple[Optional[dict], Optional[str]]:
    """
    Execute a single action on a Playwright page and return (obs, error).
    Used by tab_execution for isolated-tab rollouts; keeps tab logic independent of browser_forking.
    """
    error = None
    try:
        bgym_obs._pre_extract(page)
        python_code = action_set.to_python_code(action)
        exec_context = {
            "page": page,
            "DEMO_MODE": False,
            "send_message_to_user": lambda x: logger.info("AGENT MESSAGE: %s", x),
            "report_infeasible_instructions": lambda x: logger.warning("INFEASIBLE: %s", x),
        }
        exec(python_code, exec_context)
    except Exception as e:
        error = str(e)
        logger.debug("Action execution failed: %s", error[:200] if len(error) > 200 else error)

    if wait_after > 0:
        time.sleep(wait_after)

    if error and "execution context was destroyed" in error.lower():
        return None, error

    try:
        obs = _extract_obs_from_page(page, context, obs_flags)
        return (obs, None) if error is None else (None, error)
    except Exception as e:
        err_msg = str(e)
        if "execution context was destroyed" not in err_msg.lower():
            logger.debug("Observation extraction failed: %s", err_msg)
        return None, error or err_msg


def execute_action_on_existing_page(
    page: Page,
    context: BrowserContext,
    url: str,
    action: str,
    action_set: AbstractActionSet,
    obs_flags: Optional[dp.ObsFlags] = None,
    navigation_timeout_ms: int = 5000,
    element_timeout_ms: int = 2000,
) -> tuple[Optional[dict], Optional[str]]:
    """
    Run one action on an existing Playwright page. Caller owns page lifecycle (create/close).
    Navigates to url, then executes action and extracts obs. No new_page() or close() inside.
    """
    if not url:
        return None, "url is required"
    try:
        page.set_default_timeout(element_timeout_ms)
        page.goto(url, timeout=navigation_timeout_ms, wait_until="domcontentloaded")
        time.sleep(0.1)
        return execute_action_on_page(page, context, action, action_set, obs_flags)
    except Exception as e:
        err = str(e)
        logger.debug("execute_action_on_existing_page failed: %s", err)
        return None, err


def _extract_obs_from_page(
    page: Page,
    context: BrowserContext,
    obs_flags: Optional[dp.ObsFlags] = None,
) -> dict[str, Any]:
    """Extract observation dict from current page state (mirrors BrowserFork._extract_obs)."""
    if page is None or context is None:
        return {}
    bgym_obs._pre_extract(page)
    dom_snapshot = bgym_obs.extract_dom_snapshot(page)
    axtree = bgym_obs.extract_merged_axtree(page)
    focused_bid = bgym_obs.extract_focused_element_bid(page)
    screenshot = bgym_obs.extract_screenshot(page)
    extra_props = bgym_obs.extract_dom_extra_properties(dom_snapshot, scale_factor=1)
    obs = {
        "url": page.url,
        "screenshot": screenshot,
        "dom_object": dom_snapshot,
        "axtree_object": axtree,
        "extra_element_properties": extra_props,
        "focused_element_bid": focused_bid,
        "open_pages_urls": [page.url],
        "open_pages_titles": [page.title()],
        "active_page_index": 0,
        "last_action_error": "",
        "cookies": context.cookies(),
        "storage_state": context.storage_state(),
    }
    if obs_flags:
        preprocessor = dp.make_obs_preprocessor(obs_flags)
        obs = preprocessor(obs)
    return obs


def get_url_from_obs(obs: dict[str, Any]) -> str:
    """
    Normalize URL from observation: use 'url' or first entry of 'open_pages_urls'.
    Envs (e.g. WorkArena/BrowserGym) may provide only open_pages_urls.
    """
    if not obs:
        return ""
    url = obs.get("url") or ""
    if url:
        return url
    open_urls = obs.get("open_pages_urls") or []
    if open_urls and len(open_urls) > 0:
        first = open_urls[0]
        return first if isinstance(first, str) else ""
    return ""


def execute_action_in_tab(
    context: BrowserContext,
    lock: threading.Lock,
    start_obs: dict[str, Any],
    action: str,
    action_set: AbstractActionSet,
    obs_flags: Optional[dp.ObsFlags] = None,
    headless: bool = True,
    timeout: int = 5,
) -> tuple[Optional[dict], Optional[str]]:
    """
    Execute a single action in a new tab of the shared context. Same I/O as execute_action_in_fork.
    Creates a new page, navigates to start_obs url, runs action, extracts obs, closes page.
    """
    url = get_url_from_obs(start_obs)
    if not url:
        return None, "url is required"
    nav_timeout_ms = min(timeout * 1000, 5000)
    element_timeout_ms = 2000
    page = None
    try:
        with lock:
            page = context.new_page()
        obs, error = execute_action_on_existing_page(
            page,
            context,
            url,
            action,
            action_set,
            obs_flags,
            navigation_timeout_ms=nav_timeout_ms,
            element_timeout_ms=element_timeout_ms,
        )
        return obs, error
    finally:
        if page is not None:
            try:
                with lock:
                    page.close()
            except Exception as e:
                logger.debug("Error closing tab after execute_action_in_tab: %s", e)


def browser_tab_and_rollout(
    context: BrowserContext,
    lock: threading.Lock,
    start_obs: dict[str, Any],
    initial_action: str,
    action_set: AbstractActionSet,
    obs_flags: dp.ObsFlags,
    goal: str,
    rollout_depth: int = 3,
    action_generator: Optional[Callable[[dict, str], list[str]]] = None,
    critique_fn: Optional[
        Callable[[str, str, str, dict, str, Optional[str]], dict]
    ] = None,
    terminal_fn: Optional[Callable[[dict, str], bool]] = None,
    headless: bool = True,
    timeout: int = 120,
) -> tuple[Optional[dict], float]:
    """
    Run initial action and rollout in a new tab of the shared context.
    Same I/O as browser_fork_and_rollout. One page, then close.
    critique_fn(goal, obs_summary, action, obs, pre_summary, error) -> {'score': float}.
    """
    url = get_url_from_obs(start_obs)
    if not url:
        return None, 0.0
    nav_timeout_ms = min(timeout * 1000, 30000)
    element_timeout_ms = 2000
    page = None
    first_obs = None
    cumulative_reward = 0.0
    try:
        with lock:
            page = context.new_page()
        if page is None:
            return None, 0.0
        page.set_default_timeout(element_timeout_ms)
        page.goto(url, timeout=nav_timeout_ms, wait_until="domcontentloaded")
        time.sleep(0.1)
        current_obs, error = execute_action_on_page(
            page, context, initial_action, action_set, obs_flags
        )
        first_obs = current_obs
        if error:
            return first_obs, 0.0
        if critique_fn and current_obs:
            obs_summary = dp.Observation(current_obs, obs_flags).prompt
            pre_summary = dp.Observation(start_obs, obs_flags).prompt
            critique_result = critique_fn(
                goal, obs_summary, initial_action, current_obs, pre_summary, error
            )
            cumulative_reward = critique_result.get("score", 0.5)
        if terminal_fn and current_obs and terminal_fn(current_obs, goal):
            return first_obs, 1.0
        for _ in range(rollout_depth):
            if action_generator is None:
                break
            try:
                next_actions = action_generator(current_obs, goal)
                if not next_actions:
                    break
                next_action = next_actions[0]
            except Exception as e:
                logger.warning("Action generation failed: %s", e)
                break
            pre_obs = current_obs
            current_obs, error = execute_action_on_page(
                page, context, next_action, action_set, obs_flags
            )
            if error:
                break
            if critique_fn and current_obs:
                obs_summary = dp.Observation(current_obs, obs_flags).prompt
                pre_summary = dp.Observation(pre_obs, obs_flags).prompt
                critique_result = critique_fn(
                    goal, obs_summary, next_action, current_obs, pre_summary, error
                )
                step_reward = critique_result.get("score", 0.5)
                cumulative_reward = max(cumulative_reward, step_reward)
            if terminal_fn and current_obs and terminal_fn(current_obs, goal):
                cumulative_reward = 1.0
                break
        return first_obs, cumulative_reward
    finally:
        if page is not None:
            try:
                with lock:
                    page.close()
            except Exception as e:
                logger.debug("Error closing tab after browser_tab_and_rollout: %s", e)


def create_shared_context(
    start_obs: dict[str, Any],
    headless: bool = True,
) -> tuple[Optional[Any], Optional[Any], Optional[BrowserContext]]:
    """
    Create a single browser context from start_obs (for shared "one browser, many tabs" mode).
    Returns (playwright, browser, context). Caller must close when done.
    """
    url = get_url_from_obs(start_obs)
    if not url:
        return None, None, None
    storage_state = start_obs.get("storage_state")
    context_args = {}
    if storage_state is not None:
        if isinstance(storage_state, dict):
            try:
                fd, path = tempfile.mkstemp(suffix=".json", prefix="agentq_shared_storage_")
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(storage_state, f, indent=2)
                context_args["storage_state"] = path
            except Exception as e:
                logger.warning("Failed to write storage_state to temp file: %s", e)
        else:
            context_args["storage_state"] = storage_state
    playwright = None
    browser = None
    try:
        playwright = sync_playwright().start()
        browser = playwright.chromium.launch(headless=headless, slow_mo=0)
        context = browser.new_context(**context_args)
        context.set_default_timeout(2000)
        logger.info(
            "Tab execution | New shared browser created (headless=%s) for MCTS simulations.",
            headless,
        )
        return playwright, browser, context
    except Exception as e:
        logger.warning("Failed to create shared tab context: %s", e)
        if browser:
            try:
                browser.close()
            except Exception:
                pass
        if playwright:
            try:
                playwright.stop()
            except Exception:
                pass
        return None, None, None
