from dataclasses import dataclass

from agentlab.agents.generic_agent.generic_agent import GenericAgent, GenericAgentArgs
from agentlab.agents.generic_agent.generic_agent_prompt import GenericPromptFlags
from agentlab.llm.base_api import BaseModelArgs


@dataclass
class StructuredAgentArgs(GenericAgentArgs):
    def make_agent(self):
        return StructuredAgent(self.chat_model_args, self.flags, self.max_retry)


class StructuredAgent(GenericAgent):
    def __init__(
        self,
        chat_model_args: BaseModelArgs,
        flags: GenericPromptFlags,
        max_retry: int = 4,
    ):
        super().__init__(chat_model_args, flags, max_retry)
