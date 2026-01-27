import logging
from dataclasses import dataclass, asdict
from typing import Any, Optional

import bgym
from agentlab.agents.generic_agent.generic_agent import GenericAgent, GenericAgentArgs
from agentlab.agents.generic_agent.generic_agent_prompt import GenericPromptFlags, MainPrompt
from agentlab.agents import dynamic_prompting as dp
from agentlab.llm.chat_api import BaseModelArgs
from agentlab.llm.llm_utils import Discussion, SystemMessage, HumanMessage, retry, ParseError
from agentlab.llm.tracking import cost_tracker_decorator
from agentlab.llm.response_api import APIPayload

from .tools import HACKATHON_TOOLS
from .mcts import MCTSPlanner

try:
    from .planner_graph import create_planner_graph
except ImportError:
    create_planner_graph = None

# Optional LangChain imports
try:
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.messages import (
        SystemMessage as LCSystemMessage,
        HumanMessage as LCHumanMessage,
    )

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False


@dataclass
class HackathonAgentArgs(GenericAgentArgs):
    """
    Arguments for the HackathonAgent.
    Inherits from GenericAgentArgs to keep standard flags (obs, action, etc.)
    but allows for custom extensions.
    """

    # Add any custom arguments here
    use_langchain: bool = False
    use_mcts: bool = False
    mcts_max_simulations: int = 6
    mcts_expansion_width: int = 4
    mcts_max_depth: int = 1

    def make_agent(self):
        return HackathonAgent(
            chat_model_args=self.chat_model_args,
            flags=self.flags,
            max_retry=self.max_retry,
            use_langchain=self.use_langchain,
            use_mcts=self.use_mcts,
            mcts_max_simulations=self.mcts_max_simulations,
            mcts_expansion_width=self.mcts_expansion_width,
            mcts_max_depth=self.mcts_max_depth,
        )


class HackathonAgent(GenericAgent):
    """
    A hackathon-friendly agent that extends GenericAgent.
    It preserves the robust observation processing but exposes a cleaner
    interface for custom prompting and logic (e.g., LangChain).
    """

    def __init__(
        self,
        chat_model_args: BaseModelArgs,
        flags: GenericPromptFlags,
        max_retry: int = 4,
        use_langchain: bool = False,
        use_mcts: bool = False,
        mcts_max_simulations: int = 6,
        mcts_expansion_width: int = 4,
        mcts_max_depth: int = 1,
    ):
        super().__init__(chat_model_args, flags, max_retry)
        self.use_langchain = use_langchain
        self.use_mcts = use_mcts
        self.env = None
        self.mcts = (
            MCTSPlanner(
                self.chat_llm,
                self.action_set,
                max_simulations=mcts_max_simulations,
                expansion_width=mcts_expansion_width,
                max_depth=mcts_max_depth,
            )
            if self.use_mcts
            else None
        )

        # Initialize tools
        self.tools = HACKATHON_TOOLS
        # For native tool use, we might need a different format depending on the LLM API,
        # but for now we'll pass them if the LLM supports it.
        # This is a simplified integration.

        # Initialize Planner Graph
        self.planner_runnable = None
        self.planner_state = {}

        if self.use_langchain:
            if not LANGCHAIN_AVAILABLE:
                logging.warning(
                    "LangChain requested but not installed. Falling back to standard execution."
                )
                self.use_langchain = False
            else:
                self._setup_langchain()
                # Initialize the graph
                if create_planner_graph:
                    self.planner_runnable = create_planner_graph(
                        model_name=self.chat_model_args.model_name,
                        temperature=self.chat_model_args.temperature,
                        base_url=getattr(self, "_lc_base_url", None),
                        api_key=getattr(self, "_lc_api_key", None),
                    )
                else:
                    logging.warning("Could not create planner graph (check imports).")

    def _setup_langchain(self):
        """Initialize LangChain components here."""
        # Example setup using the provided model args
        # This is a basic setup; users can customize this heavily.
        if LANGCHAIN_AVAILABLE:
            # Check if we should use OpenRouter based on model name or API Key
            # This logic mimics what AgentLab does internally or what the user requested.
            model_name = self.chat_model_args.model_name
            import os

            # Simple heuristic: if OPENROUTER_API_KEY is present and not OPENAI_API_KEY,
            # or if the model name suggests it (e.g., has a provider prefix), use OpenRouter.
            # But simpler: just check env vars or explicit config.
            # We'll use the environment variable OPENROUTER_API_KEY if available and passed.

            api_key = os.environ.get("OPENROUTER_API_KEY")
            base_url = None

            if api_key:
                base_url = "https://openrouter.ai/api/v1"
            else:
                # Fallback to OpenAI default
                api_key = os.environ.get("OPENAI_API_KEY")

            self.lc_llm = ChatOpenAI(
                model=model_name,
                temperature=self.chat_model_args.temperature,
                api_key=api_key,
                base_url=base_url,
            )

            # Store connection details for graph creation
            self._lc_api_key = api_key
            self._lc_base_url = base_url

    @cost_tracker_decorator
    def get_action(self, obs):
        """
        Main entry point for the agent loop.
        """
        # 1. Preprocess observation (GenericAgent logic)
        processed_obs = self.obs_preprocessor(obs)
        self.obs_history.append(processed_obs)

        # 2. Invoke Planner Graph (if enabled)
        if self.use_langchain and self.planner_runnable:
            try:
                # Prepare input state
                # We need to construct the initial state if it's empty or update it
                current_state_input = {
                    "goal": obs.get("goal", ""),
                    "observation": processed_obs.get("axtree_txt", "")[
                        :5000
                    ],  # Truncate for safety/speed
                    "plan": self.plan,
                }

                # If we have previous messages, we might want to keep them,
                # but for this simple version, let's just re-run reasoning on current state.
                # In a full agent, you'd persist the whole graph state.
                if not self.planner_state:
                    self.planner_state = {"messages": []}

                # Merge input
                graph_input = {**self.planner_state, **current_state_input}

                # Invoke graph
                final_state = self.planner_runnable.invoke(graph_input)

                # Update agent's plan from graph output
                if final_state.get("plan"):
                    self.plan = final_state["plan"]
                    logging.info(f"Planner Graph updated plan: {self.plan}")

                # Persist state (optional, if you want to keep chat history inside the planner)
                self.planner_state = final_state

            except Exception as e:
                logging.error(f"Planner Graph execution failed: {e}")

        if self.use_mcts:
            if self.env is None:
                logging.warning("MCTS enabled but no environment is attached.")
            else:
                action, mcts_info = self.mcts.search(
                    processed_obs,
                    self.env,
                    obs_preprocessor=self.obs_preprocessor,
                    goal=obs.get("goal", ""),
                )
                if action is not None:
                    self.actions.append(action)
                    agent_info = bgym.AgentInfo(
                        think=f"MCTS selected action: {action}",
                        chat_messages=Discussion(
                            [
                                SystemMessage("MCTS decision"),
                                HumanMessage(f"Selected action: {action}\nInfo: {mcts_info}"),
                            ]
                        ),
                        stats=self.chat_llm.get_stats(),
                        extra_info={
                            "chat_model_args": asdict(self.chat_model_args),
                            "mcts_info": mcts_info,
                        },
                    )
                    return action, agent_info

        # 3. Build Prompts using MainPrompt (Reuse Infrastructure)
        main_prompt = MainPrompt(
            action_set=self.action_set,
            obs_history=self.obs_history,
            actions=self.actions,
            memories=self.memories,
            thoughts=self.thoughts,
            previous_plan=self.plan,
            step=self.plan_step,
            flags=self.flags,
        )

        max_prompt_tokens, max_trunc_itr = self._get_maxes()

        # We can still add custom system prompts if needed
        system_prompt = SystemMessage(dp.SystemPrompt().prompt)

        # Fit tokens using dynamic prompting infrastructure
        human_prompt = dp.fit_tokens(
            shrinkable=main_prompt,
            max_prompt_tokens=max_prompt_tokens,
            model_name=self.chat_model_args.model_name,
            max_iterations=max_trunc_itr,
            additional_prompts=system_prompt,
        )

        messages = Discussion([system_prompt, human_prompt])

        # 3. Execution (LangChain or Standard)
        if self.use_langchain:
            return self._get_action_langchain(messages, main_prompt)
        else:
            return self._get_action_standard(messages, main_prompt)

    def set_env(self, env) -> None:
        """Attach the environment for active exploration with MCTS."""
        self.env = env

    def _get_action_standard(self, messages: Discussion, main_prompt: MainPrompt):
        """Standard execution using AgentLab's LLM wrappers with Tool Support."""

        # NOTE: ToolUseAgent logic for native tool calling is complex (requires wrapping tools in API structs).
        # For this hackathon agent, we will implement a simplified ReAct-like loop or direct call.
        # If the model supports native tools, we should pass them.

        # Current implementation of `retry` in `GenericAgent` doesn't natively support tools in the loop easily
        # without changing the `retry` logic or `ChatModel` call.
        # We will use the standard `retry` but potentially we could inject tool descriptions into the prompt
        # if we wanted text-based tools.
        # However, the user asked for Native Tools.

        # To support Native Tools properly, we need to bypass `retry` or use `chat_llm` directly if it supports it.
        # `GenericAgent` uses `retry` which wraps `chat_llm`.

        # For now, let's stick to the `GenericAgent` flow (text-based) for the main action,
        # but if we want to use tools *before* the action, we should do it here.

        # Example: Simple Text-Based Tool Loop (Simulated Native)
        # 1. Check if we need to plan (handled by prompt instructions)
        # 2. Call LLM

        try:
            # We reuse MainPrompt's parser which handles <plan>, <think>, etc.
            ans_dict = retry(
                self.chat_llm,
                messages,
                n_retry=self.max_retry,
                parser=main_prompt._parse_answer,
            )
            ans_dict["busted_retry"] = 0
            ans_dict["n_retry"] = (len(messages) - 3) / 2
        except ParseError:
            ans_dict = dict(
                action=None,
                n_retry=self.max_retry + 1,
                busted_retry=1,
            )

        # Update state
        self.plan = ans_dict.get("plan", self.plan)
        self.plan_step = ans_dict.get("step", self.plan_step)
        self.actions.append(ans_dict["action"])
        self.memories.append(ans_dict.get("memory", None))
        self.thoughts.append(ans_dict.get("think", None))

        agent_info = bgym.AgentInfo(
            think=ans_dict.get("think", None),
            chat_messages=messages,
            stats=self.chat_llm.get_stats(),
            extra_info={"chat_model_args": asdict(self.chat_model_args)},
        )
        return ans_dict["action"], agent_info

    def _get_action_langchain(self, messages: Discussion, main_prompt: MainPrompt):
        """LangChain execution using MainPrompt structure."""

        # Convert AgentLab Discussion to LangChain messages
        lc_messages = []
        for msg in messages:
            if msg["role"] == "system":
                lc_messages.append(LCSystemMessage(content=msg["content"]))
            elif msg["role"] == "user":
                lc_messages.append(LCHumanMessage(content=msg["content"]))

        # We keep the original messages for the loop
        original_lc_messages = list(lc_messages)

        # Retry loop for parsing errors
        for attempt in range(3):
            try:
                # Invoke LangChain model
                response = self.lc_llm.invoke(lc_messages)
                text_answer = response.content

                # Reuse MainPrompt parser
                ans_dict = main_prompt._parse_answer(text_answer)

                # If parsing succeeds, break the loop
                break

            except Exception as e:
                logging.warning(f"LangChain execution/parsing failed (attempt {attempt+1}/3): {e}")

                # If we've exhausted retries, return error action
                if attempt == 2:
                    ans_dict = {"action": None, "think": f"Error: {e}"}
                    break

                # Add error message to conversation to prompt correction
                error_msg = f"Error: {str(e)}\nPlease ensure you provide the action inside <action>...</action> tags."

                # Append assistant response (the malformed one) and the error message
                lc_messages.append(response)  # Append the AIMessage
                lc_messages.append(LCHumanMessage(content=error_msg))

        # Update state
        self.plan = ans_dict.get("plan", self.plan)
        self.plan_step = ans_dict.get("step", self.plan_step)
        self.actions.append(ans_dict.get("action"))
        self.memories.append(ans_dict.get("memory", None))
        self.thoughts.append(ans_dict.get("think", None))

        agent_info = bgym.AgentInfo(
            think=ans_dict.get("think"),
            chat_messages=messages,
            extra_info={"source": "langchain"},
        )
        return ans_dict["action"], agent_info


# -------------------------------------------------------------------------
# Usage Example Configuration
# Copy this into your experiment script (e.g., experiments/run_osworld.py)
# -------------------------------------------------------------------------
# from agentlab.agents.hackathon import HackathonAgentArgs
# from agentlab.llm.llm_configs import CHAT_MODEL_ARGS_DICT
# from agentlab.agents.generic_agent.generic_agent_prompt import GenericPromptFlags
# from agentlab.agents import dynamic_prompting as dp
#
# # Default flags (adjust as needed)
# flags = GenericPromptFlags(
#     obs=dp.ObsFlags(
#         use_html=False,
#         use_axtree=True,
#         use_screenshot=False
#     ),
#     action=dp.ActionFlags(),
#     use_plan=True,  # Enable Planning
#     use_thinking=True
# )
#
# HACKATHON_GPT4 = HackathonAgentArgs(
#     chat_model_args=CHAT_MODEL_ARGS_DICT["openai/gpt-4o-2024-05-13"],
#     flags=flags,
#     use_langchain=False
# )
