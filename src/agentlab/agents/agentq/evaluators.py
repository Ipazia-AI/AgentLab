import json
import logging
import re
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional

from agentlab.agents import dynamic_prompting as dp
from agentlab.agents.agentq.prompts import CritiquePrompt

logger = logging.getLogger(__name__)

class BaseCritic(ABC):
    @abstractmethod
    def evaluate(self, goal: str, action: str, current_obs: Dict[str, Any], obs_flags: dp.ObsFlags, pre_obs_summary: Optional[str] = None, action_error: Optional[str] = None) -> float:
        pass

class AbsoluteCritic(BaseCritic):
    def __init__(self, llm):
        self.llm = llm

    def evaluate(self, goal: str, action: str, current_obs: Dict[str, Any], obs_flags: dp.ObsFlags, pre_obs_summary: Optional[str] = None, action_error: Optional[str] = None) -> float:
        obs_summary = dp.Observation(current_obs, obs_flags).prompt
        cp = CritiquePrompt(
            goal=goal,
            obs_summary=obs_summary,
            action=action,
            pre_obs_summary=pre_obs_summary,
            action_error=action_error,
            screenshot=current_obs.get("screenshot")
        )
        response = self.llm(cp.to_messages())
        result = cp.parse_answer(str(response))
        return result.get('score', 0.5)

class TournamentCritic(BaseCritic):
    """
    Ranks multiple candidates using iterative tournament selection.
    Matches the AgentQ paper approach: repeatedly pick best, remove, assign decreasing ranks.
    """
    def __init__(self, llm):
        self.llm = llm

    def evaluate(self, goal: str, action: str, current_obs: Dict[str, Any], obs_flags: dp.ObsFlags, pre_obs_summary: Optional[str] = None, action_error: Optional[str] = None) -> float:
        # Fallback to absolute if called for a single action
        return AbsoluteCritic(self.llm).evaluate(goal, action, current_obs, obs_flags, pre_obs_summary, action_error)

    def rank_actions_tournament(self, goal: str, state_obs: Dict[str, Any], actions: List[str], obs_flags: dp.ObsFlags) -> List[tuple[str, float]]:
        """
        Rank multiple actions. 
        Uses batch ranking (single LLM call) for speed.
        """
        if not actions:
            return []
        if len(actions) == 1:
            return [(actions[0], 1.0)]
        
        obs_summary = dp.Observation(state_obs, obs_flags).prompt
        
        # Batch ranking: ask LLM to rank all candidates in one call
        candidates_text = "\n".join(
            f"[{i}] {action}" for i, action in enumerate(actions)
        )
        prompt = f"""You are a web automation critic. Rank the following candidate actions based on how likely they are to make progress toward the goal.

Goal: {goal}

Current page state:
{obs_summary}

Candidate actions:
{candidates_text}

Rank the actions from best to worst. 
Return the indices (0, 1, 2, ...) separated by commas, starting with the best. 
Example: "1, 0, 2" if [1] is best and [2] is worst.
Return ONLY the comma-separated indices on the last line."""

        messages = [{"role": "user", "content": prompt}]
        try:
            response = str(self.llm(messages))
            # Extract numbers from the last line
            last_line = response.strip().split('\n')[-1]
            indices = [int(n) for n in re.findall(r'\d+', last_line)]
            
            # Map indices back to actions and assign scores (1.0, 0.5, 0.33...)
            ranked = []
            seen = set()
            for i, idx in enumerate(indices):
                if 0 <= idx < len(actions) and idx not in seen:
                    rank = 1.0 / (i + 1)
                    ranked.append((actions[idx], rank))
                    seen.add(idx)
            
            # Add any missing actions
            for idx, action in enumerate(actions):
                if idx not in seen:
                    rank = 0.1  # Low score for unranked
                    ranked.append((action, rank))
            
            logger.debug(f"TournamentCritic | Batch ranking completed | Ranked {len(ranked)} actions | Time: {datetime.now().strftime('%H:%M:%S')}")
            return ranked
        except Exception as e:
            logger.warning(f"TournamentCritic | Batch ranking failed: {e}. Falling back to sequential tournament. | Time: {datetime.now().strftime('%H:%M:%S')}")
            return self._rank_actions_sequential(goal, state_obs, actions, obs_flags)

    def _rank_actions_sequential(self, goal: str, state_obs: Dict[str, Any], actions: List[str], obs_flags: dp.ObsFlags) -> List[tuple[str, float]]:
        """Original sequential tournament ranking fallback."""
        obs_summary = dp.Observation(state_obs, obs_flags).prompt
        ranked = []
        remaining = list(enumerate(actions))
        
        for iteration in range(len(actions)):
            if not remaining:
                break
            if len(remaining) == 1:
                idx, action = remaining[0]
                rank = 1.0 / (iteration + 1)
                ranked.append((action, rank))
                break
            
            remaining_actions = [action for _, action in remaining]
            best_local_idx = self._ask_critic_for_best(goal, obs_summary, remaining_actions)
            
            rank = 1.0 / (iteration + 1)
            _, best_action = remaining[best_local_idx]
            ranked.append((best_action, rank))
            remaining.pop(best_local_idx)
        
        return ranked

    def _ask_critic_for_best(self, goal: str, obs_summary: str, candidates: List[str]) -> int:
        """Ask LLM to pick the single best action from candidates. Returns index."""
        candidates_text = "\n".join(
            f"[{i}] {action}" for i, action in enumerate(candidates)
        )
        prompt = f"""You are a web automation critic. Pick the SINGLE BEST action.

Goal: {goal}

Current page state:
{obs_summary}

Candidate actions:
{candidates_text}

Which action is most likely to make progress toward the goal?
Think briefly, then return ONLY the index number (0, 1, 2, ...) of the best action on the last line."""

        messages = [{"role": "user", "content": prompt}]
        try:
            response = str(self.llm(messages))
            # Extract the last number from response
            numbers = re.findall(r'\b(\d+)\b', response)
            if numbers:
                idx = int(numbers[-1])
                if 0 <= idx < len(candidates):
                    return idx
            # Fallback: try to find any digit
            for char in reversed(response):
                if char.isdigit():
                    idx = int(char)
                    if 0 <= idx < len(candidates):
                        return idx
        except Exception as e:
            logger.warning(f"Critic pick best failed: {e}")
        
        # Default to first action if parsing fails
        return 0

    def evaluate_batch(self, goal: str, candidates: List[Dict[str, Any]], obs_flags: dp.ObsFlags) -> List[float]:
        """
        Legacy batch evaluation - kept for backward compatibility.
        candidates: List of {'action': str, 'obs': dict, 'pre_obs_summary': str, 'error': str}
        """
        if not candidates:
            return []
        if len(candidates) == 1:
            c = candidates[0]
            return [self.evaluate(goal, c['action'], c['obs'], obs_flags, c.get('pre_obs_summary'), c.get('error'))]

        # Use tournament ranking on the actions, then map back to scores
        actions = [c['action'] for c in candidates]
        
        # Use the first candidate's obs for tournament ranking
        state_obs = candidates[0].get('obs', {})
        
        ranked = self.rank_actions_tournament(goal, state_obs, actions, obs_flags)
        
        # Convert ranked list back to scores aligned with original order
        action_to_rank = {action: rank for action, rank in ranked}
        return [action_to_rank.get(c['action'], 0.5) for c in candidates]
