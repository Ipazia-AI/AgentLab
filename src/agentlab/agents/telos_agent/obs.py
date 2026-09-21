"""Turn a BrowserGym observation dict into the strings Telos expects."""

from agentlab.agents import dynamic_prompting as dp
from agentlab.agents.dynamic_prompting import ObsFlags


def make_preprocessor(obs_flags: ObsFlags):
    return dp.make_obs_preprocessor(obs_flags)


def goal_from_obs(obs: dict) -> str:
    goal = obs.get("goal")
    if isinstance(goal, str) and goal.strip():
        return goal
    else:
        return str(goal)


def observation_to_text(obs: dict, use_html: bool = False) -> str:
    """Build the observation string passed to ``Telos.step``.

    Goal is a Telos session constant, not part of this string.
    """
    if use_html:
        body = obs.get("pruned_html")
    else:
        body = obs.get("axtree_txt")
    error = obs.get("last_action_error") or ""
    if error:
        return f"{body}\n\nLast action error:\n{error}"
    return body
