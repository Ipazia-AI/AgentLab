from dataclasses import dataclass

from agentlab.agents.generic_agent.generic_agent import GenericAgent, GenericAgentArgs
from agentlab.agents.generic_agent.generic_agent_prompt import GenericPromptFlags
from agentlab.llm.base_api import BaseModelArgs


@dataclass
class StructuredAgentArgs(GenericAgentArgs):
    def __post_init__(self):
        try:
            self.agent_name = f"StructuredAgent-{self.chat_model_args.model_name}".replace("/", "_")
        except AttributeError:
            pass

    def make_agent(self):
        return StructuredAgent(self.chat_model_args, self.flags, self.max_retry)
    
    def prepare(self):
        self.chat_model_args.prepare_server()

    def close(self):
        self.chat_model_args.close_server()


class StructuredAgent(GenericAgent):
    def __init__(
        self,
        chat_model_args: BaseModelArgs,
        flags: GenericPromptFlags,
        max_retry: int = 4,
    ):
        super().__init__(chat_model_args, flags, max_retry)
        
        

        

