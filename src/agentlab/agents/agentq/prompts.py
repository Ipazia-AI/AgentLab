from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class CritiquePrompt:
    goal: str
    obs_summary: str
    action: str
    pre_obs_summary: Optional[str] = None
    action_error: Optional[str] = None
    screenshot: Optional[Any] = None # numpy array (height, width, rgb)
    
    @property
    def prompt_text(self) -> str:
        pre_state = f"\nState BEFORE Action:\n{self.pre_obs_summary}\n" if self.pre_obs_summary else ""
        error_info = f"\nACTION EXECUTION ERROR:\n{self.action_error}\n" if self.action_error else ""
        
        return f"""
You are an expert web agent critic.
Your goal is to evaluate the success of an action on a given web page to achieve a user's goal.

User Goal: {self.goal}
{pre_state}
State AFTER Action:
{self.obs_summary}

Action Taken:
{self.action}

{error_info}
Critique the action's effect. Consider:
1. Does the transition from the previous state to the current state make progress towards the goal?
2. Is the resulting state valid and safe?
3. Did the action have the intended effect? (If an error occurred, the action likely failed).

Output your critique and a score from 0.0 (terrible, failure likely) to 1.0 (perfect, success likely) in JSON format.
Example:
{{
  "reasoning": "The action successfully types the search query but does not press enter, which is a necessary step.",
  "score": 0.65
}}

ONLY output the JSON object.
"""

    def to_messages(self) -> list[dict[str, Any]]:
        from agentlab.llm.llm_utils import image_to_jpg_base64_url
        
        content = [{"type": "text", "text": self.prompt_text}]
        
        if self.screenshot is not None:
            import numpy as np
            import PIL.Image
            # Convert numpy to PIL
            if isinstance(self.screenshot, np.ndarray):
                img = PIL.Image.fromarray(self.screenshot)
                content.append({
                    "type": "image_url",
                    "image_url": {"url": image_to_jpg_base64_url(img)}
                })
        
        return [{"role": "user", "content": content}]

    def parse_answer(self, answer: str) -> dict[str, Any]:
        import json
        import re
        try:
            # Find JSON block if it exists
            match = re.search(r"(\{.*\})", answer, re.DOTALL)
            if match:
                json_str = match.group(1)
                data = json.loads(json_str)
                return {
                    'score': float(data.get('score', 0.5)),
                    'reasoning': data.get('reasoning', answer)
                }
            
            # Fallback to old regex scoring if JSON fails
            match = re.search(r"\"score\":\s*(\d+(\.\d+)?)", answer)
            if not match:
                match = re.search(r"score:\s*(\d+(\.\d+)?)", answer, re.IGNORECASE)
            
            if match:
                score = float(match.group(1))
            else:
                score = 0.5
            return {'score': score, 'reasoning': answer}
        except Exception:
            # Last resort
            return {'score': 0.5, 'reasoning': answer}

@dataclass
class RefinementPrompt:
    action: str
    critique: str
    score: float
    
    @property
    def prompt(self) -> str:
        return f"""
The previous action attempt was critiqued as follows:
Action: {self.action}
Critique: {self.critique}
Score: {self.score}

Please suggest a refined action that addresses the critique and improves the score.
"""

@dataclass
class TerminalJudgePrompt:
    goal: str
    obs_summary: str
    screenshot: Optional[Any] = None

    @property
    def prompt_text(self) -> str:
        return f"""
You are an expert web judge. Determine if the user's goal has been COMPLETELY achieved based on the current state of the web page.

User Goal: {self.goal}

Current Page State:
{self.obs_summary}

Has the goal been achieved?
Return a JSON object with:
"is_terminal": true/false
"reasoning": "Brief explanation"

ONLY output the JSON object.
"""

    def to_messages(self) -> list[dict[str, Any]]:
        from agentlab.llm.llm_utils import image_to_jpg_base64_url
        content = [{"type": "text", "text": self.prompt_text}]
        if self.screenshot is not None:
            import numpy as np
            import PIL.Image
            if isinstance(self.screenshot, np.ndarray):
                img = PIL.Image.fromarray(self.screenshot)
                content.append({
                    "type": "image_url",
                    "image_url": {"url": image_to_jpg_base64_url(img)}
                })
        return [{"role": "user", "content": content}]

    def parse_answer(self, answer: str) -> bool:
        import json
        import re
        try:
            match = re.search(r"(\{.*\})", answer, re.DOTALL)
            if match:
                data = json.loads(match.group(1))
                return bool(data.get('is_terminal', False))
        except Exception:
            pass
        return "true" in answer.lower()


@dataclass
class ActorPrompt:
    """
    Generate K distinct candidate actions for MCTS expansion.
    Matches the AgentQ paper's actor component.
    """
    goal: str
    completed_actions: list[str]
    current_obs_summary: str
    action_space_description: str
    n_candidates: int = 3
    dpo_context: str = ""
    extra_instructions: str = ""

    @property
    def prompt_text(self) -> str:
        history = "\n".join(f"- {a}" for a in self.completed_actions[-5:]) if self.completed_actions else "None"
        extra = f"\n{self.extra_instructions}\n" if self.extra_instructions else ""
        
        return f"""You are a web automation actor. Generate {self.n_candidates} DISTINCT candidate actions.

## Goal
{self.goal}
{extra}
## Completed Actions (recent)
{history}

## Current Page State
{self.current_obs_summary}

{self.dpo_context}

## Action Space
{self.action_space_description}

## Task
Generate exactly {self.n_candidates} different valid actions to make progress toward the goal.
Each action should represent a DIFFERENT approach or strategy.
Think step-by-step about what the user wants to achieve.

Format each action as:
<action id="1">
... action code ...
</action>
<action id="2">
... action code ...
</action>
<action id="3">
... action code ...
</action>
"""

    def to_messages(self) -> list[dict[str, Any]]:
        return [{"role": "user", "content": self.prompt_text}]

    def parse_actions(self, response: str) -> list[str]:
        """Parse multiple action blocks from response."""
        import re
        
        actions = []
        
        # Try to find <action> blocks
        pattern = r'<action[^>]*>(.*?)</action>'
        matches = re.findall(pattern, response, re.DOTALL | re.IGNORECASE)
        
        if matches:
            for match in matches:
                action = match.strip()
                if action:
                    actions.append(action)
        
        # Fallback: look for numbered actions
        if not actions:
            lines = response.split('\n')
            for line in lines:
                line = line.strip()
                # Remove numbering like "1.", "2.", etc.
                line = re.sub(r'^\d+\.\s*', '', line)
                # Check if it looks like a function call
                if line and '(' in line and ')' in line:
                    actions.append(line)
        
        return actions[:self.n_candidates]


@dataclass
class CriticPickBestPrompt:
    """
    Ask the critic to pick the single best action from candidates.
    Used for iterative tournament ranking in TournamentCritic.
    """
    goal: str
    current_state_summary: str
    candidates: list[str]
    
    @property
    def prompt_text(self) -> str:
        candidates_text = "\n".join(
            f"[{i}] {action}" for i, action in enumerate(self.candidates)
        )
        return f"""You are a web automation critic. Pick the SINGLE BEST action.

## Goal
{self.goal}

## Current Page State
{self.current_state_summary}

## Candidate Actions
{candidates_text}

## Task
Which action is most likely to make progress toward the goal?
Consider:
1. Does the action interact with relevant page elements?
2. Is the action a logical next step given the goal?
3. Will the action move us closer to completing the task?

Think briefly about each option, then return ONLY the index number (0, 1, 2, ...) of the best action on the final line.
"""

    def to_messages(self) -> list[dict[str, Any]]:
        return [{"role": "user", "content": self.prompt_text}]

    def parse_answer(self, response: str) -> int:
        """Parse the selected index from the response."""
        import re
        
        # Look for numbers in the response
        numbers = re.findall(r'\b(\d+)\b', response)
        if numbers:
            # Take the last number (most likely the final answer)
            idx = int(numbers[-1])
            if 0 <= idx < len(self.candidates):
                return idx
        
        # Fallback: look for any single digit
        for char in reversed(response):
            if char.isdigit():
                idx = int(char)
                if 0 <= idx < len(self.candidates):
                    return idx
        
        # Default to first action
        return 0
