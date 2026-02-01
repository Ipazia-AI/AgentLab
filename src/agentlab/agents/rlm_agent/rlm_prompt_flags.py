"""
Prompt flags for RLMGenericAgent.

Based on GenericPromptFlags but adapted for RLM-style interaction.
"""

from dataclasses import dataclass

from agentlab.agents import dynamic_prompting as dp


@dataclass
class RLMPromptFlags(dp.Flags):
    """
    Flags to control RLMGenericAgent behavior.

    Attributes:
        obs: Observation flags (what parts of obs to include)
        action: Action flags (action set configuration)
        max_prompt_tokens: Maximum tokens in prompt (for recursive calls)
        max_trunc_itr: Maximum truncation iterations
    """

    obs: dp.ObsFlags = None
    action: dp.ActionFlags = None
    max_prompt_tokens: int = None
    max_trunc_itr: int = 20

    def __post_init__(self):
        if self.obs is None:
            self.obs = dp.ObsFlags()
        if self.action is None:
            self.action = dp.ActionFlags()
