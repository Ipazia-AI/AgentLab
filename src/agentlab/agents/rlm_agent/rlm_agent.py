"""
RLMGenericAgent implementation for AgentLab.

This module implements an agent that uses the Recursive Language Model (RLM) paradigm:
instead of stuffing the full DOM/AXTree into the prompt, it externalizes the observation
to a REPL environment where the model can programmatically explore it.

Based on the RLM paper: https://arxiv.org/abs/2512.24601
"""

import logging
import re as regex_module
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any

from bgym import Benchmark
from browsergym.experiments.agent import Agent, AgentInfo

from agentlab.agents import dynamic_prompting as dp
from agentlab.agents.agent_args import AgentArgs
from agentlab.llm.chat_api import BaseModelArgs
from agentlab.llm.llm_utils import Discussion, ParseError, SystemMessage, HumanMessage
from agentlab.llm.tracking import cost_tracker_decorator

from .rlm_repl import REPLExecutor, REPLError
from .rlm_parser import is_final, parse_response
from .rlm_prompts import build_system_prompt, build_user_prompt, build_iteration_context
from .rlm_prompt_flags import RLMPromptFlags


logger = logging.getLogger(__name__)


class RLMError(Exception):
    """Base error for RLM agent."""
    pass


class MaxIterationsError(RLMError):
    """Max iterations exceeded without FINAL."""
    pass


@dataclass
class RLMGenericAgentArgs(AgentArgs):
    """Arguments for RLMGenericAgent."""
    
    chat_model_args: BaseModelArgs = None
    recursive_model_args: BaseModelArgs = None  # Optional cheaper model for recursive calls
    flags: RLMPromptFlags = None
    max_retry: int = 4
    max_iterations: int = 10  # Max REPL iterations per step
    max_depth: int = 2  # Max recursion depth for recursive_llm

    def __post_init__(self):
        try:
            self.agent_name = f"RLMGenericAgent-{self.chat_model_args.model_name}".replace("/", "_")
        except AttributeError:
            pass
        
        # Use main model for recursive calls if not specified
        if self.recursive_model_args is None:
            self.recursive_model_args = self.chat_model_args

    def set_benchmark(self, benchmark: Benchmark, demo_mode):
        """Override some flags based on the benchmark."""
        if benchmark.name.startswith("miniwob"):
            self.flags.obs.use_html = True

        self.flags.obs.use_tabs = benchmark.is_multi_tab
        self.flags.action.action_set = deepcopy(benchmark.high_level_action_set_args)

        if self.flags.action.multi_actions is not None:
            self.flags.action.action_set.multiaction = self.flags.action.multi_actions
        if self.flags.action.is_strict is not None:
            self.flags.action.action_set.strict = self.flags.action.is_strict

        if demo_mode:
            self.flags.action.action_set.demo_mode = "all_blue"

    def set_reproducibility_mode(self):
        self.chat_model_args.temperature = 0
        if self.recursive_model_args:
            self.recursive_model_args.temperature = 0

    def prepare(self):
        return self.chat_model_args.prepare_server()

    def close(self):
        return self.chat_model_args.close_server()

    def make_agent(self):
        return RLMGenericAgent(
            chat_model_args=self.chat_model_args,
            recursive_model_args=self.recursive_model_args,
            flags=self.flags,
            max_retry=self.max_retry,
            max_iterations=self.max_iterations,
            max_depth=self.max_depth,
        )


class RLMGenericAgent(Agent):
    """
    RLM-based agent for web automation.
    
    Instead of putting the full observation in the prompt, this agent:
    1. Externalizes the observation to a REPL environment
    2. Lets the model write code to explore the observation
    3. Iterates until the model produces a FINAL() action
    """

    def __init__(
        self,
        chat_model_args: BaseModelArgs,
        flags: RLMPromptFlags,
        recursive_model_args: BaseModelArgs = None,
        max_retry: int = 4,
        max_iterations: int = 10,
        max_depth: int = 2,
    ):
        self.chat_llm = chat_model_args.make_model()
        self.chat_model_args = chat_model_args
        self.recursive_model_args = recursive_model_args or chat_model_args
        self.max_retry = max_retry
        self.max_iterations = max_iterations
        self.max_depth = max_depth

        self.flags = flags
        self.action_set = self.flags.action.action_set.make_action_set()
        self._obs_preprocessor = dp.make_obs_preprocessor(flags.obs)

        # REPL executor for code execution
        self.repl = REPLExecutor(max_output_chars=4000)

        self.reset(seed=None)

    def obs_preprocessor(self, obs: dict) -> dict:
        return self._obs_preprocessor(obs)

    @cost_tracker_decorator
    def get_action(self, obs: dict) -> tuple[str | None, AgentInfo]:
        """
        Get the next action using the RLM loop.
        
        Args:
            obs: Observation dict from the browser environment
            
        Returns:
            Tuple of (action, AgentInfo)
        """
        self.obs_history.append(obs)
        
        # Build REPL context from observation
        repl_context = self._build_repl_context(obs)
        
        # Build REPL environment
        repl_env = self._build_repl_env(repl_context)
        
        # Get action set description for the prompt
        action_set_description = self.action_set.describe(
            with_long_description=self.flags.action.long_description,
            with_examples=self.flags.action.individual_examples,
        )
        
        # Build context info for system prompt
        context_info = {
            'axtree': len(repl_context.get('axtree', '')),
            'html': len(repl_context.get('html', '')),
        }
        
        # Build system prompt
        system_prompt_text = build_system_prompt(
            context_info=context_info,
            action_set_description=action_set_description,
            depth=0,
        )
        system_prompt = SystemMessage(system_prompt_text)
        
        # Build initial user prompt
        goal = self._extract_goal(obs)
        user_prompt_text = build_user_prompt(goal, iteration=0)
        
        # Initialize conversation
        messages = Discussion([system_prompt, HumanMessage(user_prompt_text)])
        
        # RLM iteration loop
        final_action = None
        iteration_count = 0
        
        for iteration in range(self.max_iterations):
            iteration_count = iteration + 1
            
            # Call LLM
            try:
                llm_response = self.chat_llm(messages)
                # Extract content from response dict (LLM returns {"role": "assistant", "content": "..."})
                response_text = llm_response["content"] if isinstance(llm_response, dict) else str(llm_response)
            except Exception as e:
                logger.error(f"LLM call failed: {e}")
                break
            
            # Check for FINAL statement
            if is_final(response_text):
                final_text = parse_response(response_text, repl_env)
                if final_text is not None:
                    # Extract and validate action
                    try:
                        final_action = self._extract_action(final_text)
                        break
                    except ParseError as e:
                        # FINAL contained invalid action, add error and continue
                        error_msg = f"Invalid action format: {e}. Please fix the action and try FINAL() again."
                        messages.append({"role": "assistant", "content": response_text})
                        messages.append({"role": "user", "content": error_msg})
                        continue
            
            # Execute code in REPL
            try:
                exec_result = self.repl.execute(response_text, repl_env)
            except REPLError as e:
                exec_result = f"Error: {str(e)}"
            except Exception as e:
                exec_result = f"Unexpected error: {str(e)}"
            
            # Add to conversation
            messages.append({"role": "assistant", "content": response_text})
            
            # Build iteration context
            code_executed = self.repl._extract_code(response_text)
            iteration_msg = build_iteration_context(code_executed, exec_result)
            
            # Add user message for next iteration
            next_user_prompt = build_user_prompt(goal, iteration=iteration + 1)
            messages.append({"role": "user", "content": f"{iteration_msg}\n\n{next_user_prompt}"})
        
        # Handle case where no FINAL was produced
        if final_action is None:
            logger.warning(f"No FINAL() produced after {iteration_count} iterations")
            # Try one more time asking explicitly for the action
            messages.append({"role": "user", "content": 
                "You must provide an action now. Based on what you've learned, use FINAL(\"<action>...</action>\") with your best action."
            })
            try:
                llm_response = self.chat_llm(messages)
                response_text = llm_response["content"] if isinstance(llm_response, dict) else str(llm_response)
                if is_final(response_text):
                    final_text = parse_response(response_text, repl_env)
                    if final_text:
                        try:
                            final_action = self._extract_action(final_text)
                        except ParseError:
                            pass
            except Exception:
                pass
        
        # Build stats
        stats = self.chat_llm.get_stats()
        stats["rlm_iterations"] = iteration_count
        stats["n_retry"] = 0
        stats["busted_retry"] = 0 if final_action else 1
        
        # Update agent state
        self.actions.append(final_action)
        
        # Build AgentInfo
        agent_info = AgentInfo(
            think=f"RLM loop ran for {iteration_count} iterations",
            chat_messages=messages,
            stats=stats,
            extra_info={
                "chat_model_args": asdict(self.chat_model_args),
                "rlm_iterations": iteration_count,
            },
        )
        
        return final_action, agent_info

    def reset(self, seed=None):
        """Reset agent state."""
        self.seed = seed
        self.actions = []
        self.obs_history = []

    def _build_repl_context(self, obs: dict) -> dict[str, Any]:
        """
        Build the context dict that will be available in the REPL.
        
        Args:
            obs: Observation dict
            
        Returns:
            Context dict for REPL
        """
        context = {
            'axtree': obs.get('axtree_txt', ''),
            'html': obs.get('pruned_html', obs.get('dom_txt', '')),
            'url': obs.get('url', ''),
            'tabs': self._format_tabs(obs),
            'error': obs.get('last_action_error', ''),
            'goal': self._extract_goal(obs),
            'action_history': [str(a) for a in self.actions if a],
        }
        return context

    def _build_repl_env(self, context: dict) -> dict[str, Any]:
        """
        Build the REPL environment with context and helper functions.
        
        Args:
            context: The context dict
            
        Returns:
            Environment dict for REPL execution
        """
        env = {
            'context': context,
            'recursive_llm': self._make_recursive_fn(),
            're': regex_module,
        }
        return env

    def _make_recursive_fn(self):
        """
        Create the recursive_llm function for the REPL.
        
        Returns:
            Function that can be called from REPL code
        """
        recursive_llm = self.recursive_model_args.make_model()
        
        def recursive_llm_fn(query: str, sub_context: str) -> str:
            """
            Query a sub-LLM to analyze a portion of context.
            
            Args:
                query: The question to ask
                sub_context: The context chunk to analyze
                
            Returns:
                LLM response
            """
            prompt = f"""Analyze the following context and answer the question.

Context:
{sub_context[:10000]}  # Limit context size

Question: {query}

Provide a concise answer."""
            
            messages = Discussion([
                SystemMessage("You are a helpful assistant analyzing web page content."),
                HumanMessage(prompt),
            ])
            
            try:
                response = recursive_llm(messages)
                return response
            except Exception as e:
                return f"Error in recursive_llm: {str(e)}"
        
        return recursive_llm_fn

    def _extract_goal(self, obs: dict) -> str:
        """Extract the goal from observation."""
        goal_object = obs.get('goal_object', [])
        if isinstance(goal_object, list):
            # Extract text from goal object
            goal_parts = []
            for item in goal_object:
                if isinstance(item, dict) and 'text' in item:
                    goal_parts.append(item['text'])
                elif isinstance(item, str):
                    goal_parts.append(item)
            return ' '.join(goal_parts)
        return str(goal_object)

    def _format_tabs(self, obs: dict) -> str:
        """Format open tabs info."""
        urls = obs.get('open_pages_urls', [])
        titles = obs.get('open_pages_titles', [])
        active = obs.get('active_page_index', 0)
        
        if not urls:
            return "No tabs info"
        
        parts = []
        for i, (url, title) in enumerate(zip(urls, titles)):
            marker = " (active)" if i == active else ""
            parts.append(f"Tab {i}{marker}: {title} - {url}")
        
        return "\n".join(parts)

    def _extract_action(self, final_text: str) -> str:
        """
        Extract and validate action from FINAL text.
        
        Args:
            final_text: The text inside FINAL()
            
        Returns:
            Validated action string
            
        Raises:
            ParseError: If action format is invalid
        """
        # Try to extract action from <action> tags
        action_match = regex_module.search(r'<action>(.*?)</action>', final_text, regex_module.DOTALL)
        if action_match:
            action = action_match.group(1).strip()
        else:
            # Assume the whole thing is the action
            action = final_text.strip()
        
        # Validate action against action set
        try:
            self.action_set.to_python_code(action)
        except Exception as e:
            raise ParseError(f"Invalid action: {e}")
        
        return action
