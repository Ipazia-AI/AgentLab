"""
RLM (Recursive Language Model) as a composable LLM decorator.

This module provides RLMChatModel, a decorator that wraps any AbstractChatModel
to add long-context handling through iterative REPL exploration and recursive sub-calls.

Based on the RLM paper (arXiv:2512.24601): "Recursive Language Models"
by Alex L. Zhang, Tim Kraska, and Omar Khattab.

RLM is NOT a provider - it composes over existing providers (OpenAI, LiteLLM,
Anthropic, etc.) to enhance their ability to handle long contexts.

Example usage:
    # Wrap any existing provider
    inner = OpenAIModelArgs(model_name="gpt-4o").make_model()
    rlm = RLMChatModel(inner_model=inner, max_iterations=20)

    # Or use RLMModelArgs for configuration
    args = RLMModelArgs(
        inner_model_args=OpenAIModelArgs(model_name="gpt-4o"),
        recursive_model_args=OpenAIModelArgs(model_name="gpt-4o-mini"),
        max_depth=5,
    )
    rlm = args.make_model()
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from .base_api import AbstractChatModel, BaseModelArgs
from .llm_utils import AIMessage
from .rlm_parser import check_for_final_answer, is_final
from .rlm_prompts import REPL_SYSTEM_PROMPT, USER_PROMPT
from .rlm_repl import REPLError, REPLExecutor

logger = logging.getLogger(__name__)

class RLMError(Exception):
    """Base error for RLM operations."""

    pass


class MaxIterationsError(RLMError):
    """Max iterations exceeded without FINAL() statement."""

    pass


class MaxDepthError(RLMError):
    """Max recursion depth exceeded."""

    pass


class RLMChatModel(AbstractChatModel):
    """
    Decorator that wraps any AbstractChatModel with RLM long-context handling.

    This is NOT a provider - it composes over existing providers (OpenAI, LiteLLM,
    Anthropic, OpenRouter, etc.) to add iterative REPL-based context exploration.

    The RLM approach externalizes long context to a REPL environment variable,
    allowing the model to programmatically explore and process it through code.
    """

    def __init__(
        self,
        inner_model: AbstractChatModel,
        recursive_model: AbstractChatModel | None = None,
        max_depth: int = 5,
        max_iterations: int = 30,
        max_output_chars: int = 2000,
        _current_depth: int = 0,
    ):
        """
        Initialize RLM decorator.

        Args:
            inner_model: The model to wrap (any AbstractChatModel - OpenAI, LiteLLM, etc.)
            recursive_model: Optional cheaper model for recursive sub-calls.
                             If None, uses inner_model for all calls.
            max_depth: Maximum recursion depth for nested sub-calls
            max_iterations: Maximum REPL iterations per call
            max_output_chars: Maximum characters in REPL output (truncated if longer)
            _current_depth: Internal tracker for recursion depth
        """
        self.inner_model = inner_model
        self.recursive_model = recursive_model or inner_model
        self.max_depth = max_depth
        self.max_iterations = max_iterations
        self.max_output_chars = max_output_chars
        self._current_depth = _current_depth

        self.repl = REPLExecutor(max_output_chars=max_output_chars)

        # Observation dict and action_set set by GenericAgent before each call
        self.obs: dict | None = None
        self.action_set = None
        self.current_task_name: str | None = None
        self._last_task_name: str | None = None

        # Stats tracking
        self._llm_calls = 0
        self._iterations = 0
        self._n_retry = 0

    def __call__(self, messages: list[dict], **kwargs) -> dict:
        """
        Process messages using RLM approach.

        Converts the incoming messages to (query, context) format, then runs
        the RLM loop: model generates code -> REPL executes -> repeat until
        FINAL() is detected.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
            **kwargs: Additional parameters (passed to inner model)

        Returns:
            AIMessage dict with the final answer

        Raises:
            MaxIterationsError: If no FINAL() after max_iterations
            MaxDepthError: If recursion exceeds max_depth
        """
        if self._current_depth >= self.max_depth:
            raise MaxDepthError(f"Max recursion depth ({self.max_depth}) exceeded")

        # Track stats per get_action call (not cumulatively across the episode)
        self._llm_calls = 0
        self._iterations = 0
        self._n_retry = 0

        if self.current_task_name is not None and self._last_task_name is not None:
            if self.current_task_name != self._last_task_name:
                logger.info(
                    "RLM model instance switched task context from '%s' to '%s'",
                    self._last_task_name,
                    self.current_task_name,
                )
        if self.current_task_name is not None:
            self._last_task_name = self.current_task_name

        # Extract query and context from self.obs (set by GenericAgent)
        query, context, images = self._extract_query_and_context(messages)

        # Initialize REPL environment
        repl_env = self._build_repl_env(query, context)
        logger.debug(
            "RLM init: query_len=%s axtree_len=%s html_len=%s has_error=%s images=%s",
            len(query),
            len(context.get("axtree", "")),
            len(context.get("html", "")),
            bool(context.get("error")),
            len(images),
        )

        # Build RLM conversation (system prompt is stable, no arguments)
        # This task_info is needed as information that the model needs to know about the existence of bids in the AXTree contained in the context variable.
        task_info = "Note: [bid] is the unique alpha-numeric identifier at the beginning of lines for each element in the AXTree. Always use bid to refer to elements in your actions. The axtree and other important information are provided in the context variable which is a dictionary with the following keys: context['axtree'] (accessibility tree), context['html'], context['goal'], context['error'], you MUST look through it at least once before answering your query."
        rlm_messages: list[dict] = [
            {"role": "system", "content": REPL_SYSTEM_PROMPT},
            {"role": "user", "content": task_info + "\n\n" + query + "\n\n" + USER_PROMPT.format(query=query)},
        ]

        # If there are images, add them to the first user message
        if images:
            rlm_messages[1] = self._add_images_to_message(rlm_messages[1], images)

        # Main RLM loop
        for iteration in range(self.max_iterations):
            self._iterations = iteration + 1

            # Call the wrapped inner model
            response_dict = self.inner_model(rlm_messages, **kwargs)
            self._llm_calls += 1

            response_text = response_dict.get("content", "")
            self._log_iteration_response(iteration + 1, response_text)

            # Check for FINAL() or FINAL_VAR() at start of line
            if is_final(response_text):
                answer = check_for_final_answer(response_text, repl_env)
                if answer is not None:
                    # Wrap in <action> tags if not already present
                    # (GenericAgent's parser expects <action>...</action>)
                    if "<action>" not in answer:
                        answer = f"<action>\n{answer}\n</action>"
                    return AIMessage(answer)
                logger.warning(
                    "RLM FINAL detected but could not resolve answer; "
                    "iteration=%s response_snippet=%s",
                    iteration + 1,
                    self._truncate_text(response_text),
                )

            # Execute code in REPL
            try:
                exec_result = self.repl.execute(response_text, repl_env)
            except REPLError as e:
                exec_result = f"Error: {str(e)}"
            except Exception as e:
                exec_result = f"Unexpected error: {str(e)}"
                logging.warning(f"RLM REPL error: {e}")

            self._log_repl_result(iteration + 1, exec_result)

            # Add to conversation
            rlm_messages.extend(
                [
                    {"role": "assistant", "content": response_text},
                    {"role": "user", "content": exec_result}
                ]
            )

        raise MaxIterationsError(
            f"Max iterations ({self.max_iterations}) exceeded without FINAL()"
        )

    def get_stats(self) -> dict:
        """
        Get execution statistics.

        Returns:
            Dict with RLM stats combined with inner model stats
        """
        stats = {
            "n_retry_llm": self._n_retry,
            "rlm_llm_calls": self._llm_calls,
            "rlm_iterations": self._iterations,
            "rlm_depth": self._current_depth,
        }

        # Combine with inner model stats
        inner_stats = self.inner_model.get_stats()
        stats.update({f"inner_{k}": v for k, v in inner_stats.items()})

        return stats

    def _extract_query_and_context(
        self, messages: list[dict]
    ) -> tuple[str, dict[str, Any], list[dict]]:
        """Build query and context from self.obs (set by GenericAgent)."""
        obs = self.obs

        # Goal: assume standard structure from GenericAgent
        goal = obs["goal_object"][0]["text"]

        # Action format from action_set (set by GenericAgent)
        action_format = ""
        if self.action_set is not None:
            action_format = f"# Action space:\n{self.action_set.describe()}"

        # Get human_content for image extraction
        human_content = messages[-1].get("content", "")

        # Context dict for REPL (large items for exploration)
        context = {
            "axtree": obs.get("axtree_txt", ""),
            "html": obs.get("pruned_html", ""),
            "goal": goal,
            "error": obs.get("last_action_error", ""),
        }

        # Images: assume list content format if present
        images = []
        if isinstance(human_content, list):
            images = [item for item in human_content if item.get("type") == "image_url"]

        query = f"Task: {goal}\n\n{action_format}"
        return query, context, images

    def _add_images_to_message(
        self, message: dict, images: list[dict]
    ) -> dict:
        """Add images to a message (converting to multimodal format if needed)."""
        content = message.get("content", "")

        if isinstance(content, str):
            # Convert to multimodal format
            new_content = [{"type": "text", "text": content}]
        else:
            new_content = list(content)

        # Add images
        new_content.extend(images)

        return {"role": message["role"], "content": new_content}

    def _build_repl_env(self, query: str, context: dict[str, Any]) -> dict[str, Any]:
        """
        Build REPL environment with context dict, query, and llm_query function.

        Args:
            query: The user query (short task description)
            context: Dict with observations (axtree, html, history, etc.)

        Returns:
            Environment dict for REPL execution
        """
        return {
            "context": context,  # Dict with axtree, html, history, etc.
            # Backward-compatible aliases used frequently by model-generated code
            "axtree_str": context.get("axtree", ""),
            "html_str": context.get("html", ""),
            "goal": context.get("goal", ""),
            "last_action_error": context.get("error", ""),
            "query": query,
            "llm_query": self._make_llm_query_fn(),
            "re": re,  # Pre-import re module
        }

    def _truncate_text(self, text: str, limit: int = 500) -> str:
        if len(text) <= limit:
            return text
        return f"{text[:limit]}...[truncated {len(text) - limit} chars]"

    def _log_iteration_response(self, iteration: int, response_text: str) -> None:
        logger.debug(
            "RLM iteration %s/%s response_len=%s",
            iteration,
            self.max_iterations,
            len(response_text),
        )
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "RLM iteration %s response_snippet=%s",
                iteration,
                self._truncate_text(response_text),
            )

    def _log_repl_result(self, iteration: int, exec_result: str) -> None:
        if exec_result.startswith("No code block found"):
            logger.warning("RLM iteration %s: no repl code block found", iteration)
        elif exec_result.startswith("Empty code block"):
            logger.warning("RLM iteration %s: empty repl code block", iteration)
        elif exec_result.startswith("Error:"):
            logger.warning(
                "RLM iteration %s repl error: %s",
                iteration,
                self._truncate_text(exec_result),
            )
    def _make_llm_query_fn(self):
        """
        Create the llm_query function for REPL environment.

        This provides a simple interface to query the recursive LLM directly,
        similar to the original RLM implementation's Sub_RLM.

        Returns:
            A function that takes a prompt string and returns the LLM response
        """

        def llm_query(prompt: Any) -> str:
            """
            Query the LLM with the given prompt.

            This is a direct LLM call without REPL capabilities - useful for
            summarizing chunks, answering questions about sub-contexts, etc.

            Args:
                prompt: The prompt to send to the LLM

            Returns:
                The LLM's response as a string
            """
            if self._current_depth + 1 >= self.max_depth:
                return f"Max recursion depth ({self.max_depth}) reached"

            # Build simple messages for direct LLM call
            if isinstance(prompt, str):
                prompt_text = prompt
            else:
                # Some generated REPL code passes dict/list; coerce instead of crashing.
                try:
                    prompt_text = json.dumps(prompt, ensure_ascii=True)
                except TypeError:
                    prompt_text = str(prompt)
            messages = [{"role": "user", "content": prompt_text}]

            try:
                # Call the recursive model directly (not wrapped in RLM)
                response = self.recursive_model(messages)
                return response.get("content", "")
            except Exception as e:
                return f"LLM query error: {str(e)}"

        return llm_query


@dataclass
class RLMModelArgs(BaseModelArgs):
    """
    Configuration for RLM decorator. NOT a provider itself - wraps existing providers.

    The inner_model_args can be ANY BaseModelArgs subclass:
    - OpenAIModelArgs, AnthropicModelArgs, LiteLLMModelArgs, etc.

    Example:
        args = RLMModelArgs(
            inner_model_args=OpenAIModelArgs(model_name="gpt-4o"),
            recursive_model_args=OpenAIModelArgs(model_name="gpt-4o-mini"),
            max_depth=5,
            max_iterations=30,
        )
        model = args.make_model()
    """

    # Required: the provider to wrap
    inner_model_args: BaseModelArgs = None

    # Optional: cheaper model for recursive sub-calls (defaults to inner_model)
    recursive_model_args: BaseModelArgs | None = None

    # RLM-specific parameters
    max_depth: int = 5
    max_iterations: int = 30
    max_output_chars: int = 2000

    # Override BaseModelArgs defaults (will be set from inner_model_args)
    model_name: str = field(default="rlm")

    def __post_init__(self):
        """Set derived fields from inner_model_args."""
        if self.inner_model_args is None:
            raise ValueError("inner_model_args is required for RLMModelArgs")

        # Inherit properties from inner model for compatibility
        self.model_name = f"rlm({self.inner_model_args.model_name})"
        self.vision_support = self.inner_model_args.vision_support
        self.max_total_tokens = self.inner_model_args.max_total_tokens
        self.max_input_tokens = self.inner_model_args.max_input_tokens
        self.max_new_tokens = self.inner_model_args.max_new_tokens
        self.temperature = self.inner_model_args.temperature

    def make_model(self) -> RLMChatModel:
        """
        Create an RLMChatModel instance.

        Returns:
            RLMChatModel wrapping the configured inner model(s)
        """
        # Create the inner model from provider args
        inner = self.inner_model_args.make_model()

        # Create recursive model (or use inner if not specified)
        if self.recursive_model_args is not None:
            recursive = self.recursive_model_args.make_model()
        else:
            recursive = inner

        return RLMChatModel(
            inner_model=inner,
            recursive_model=recursive,
            max_depth=self.max_depth,
            max_iterations=self.max_iterations,
            max_output_chars=self.max_output_chars,
        )

    def prepare_server(self):
        """Prepare server for inner model if needed."""
        self.inner_model_args.prepare_server()
        if self.recursive_model_args is not None:
            self.recursive_model_args.prepare_server()

    def close_server(self):
        """Close server for inner model if needed."""
        self.inner_model_args.close_server()
        if self.recursive_model_args is not None:
            self.recursive_model_args.close_server()
