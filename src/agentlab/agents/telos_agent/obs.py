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
    # goal_object = obs.get("goal_object") # goal_object type is tuple[dict[str, str]]
    # if isinstance(goal_object, str) and goal_object.strip():
    #     return goal_object
    # if isinstance(goal_object, list):
    #     texts = []
    #     for item in goal_object:
    #         if isinstance(item, str):
    #             texts.append(item)
    #         elif isinstance(item, dict) and item.get("content"):
    #             texts.append(str(item["content"]))
    #     if texts:
    #         return "\n".join(texts)
    # return str(goal or goal_object or "")


def observation_to_text(obs: dict, use_html: bool = False) -> str:
    """Build the observation string passed to ``Telos.step``.

    Goal is a Telos session constant, not part of this string.
    """
    if use_html:
        body = obs.get("pruned_html") or obs.get("axtree_txt") or ""
    else:
        body = obs.get("axtree_txt") or obs.get("pruned_html") or ""
    error = obs.get("last_action_error") or ""
    if error:
        return f"{body}\n\nLast action error:\n{error}"
    return body
