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

import concurrent.futures
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from .base_api import AbstractChatModel, BaseModelArgs
from .llm_utils import AIMessage
from .rlm_parser import is_final, parse_response
from .rlm_prompts import build_system_prompt
from .rlm_repl import REPLError, REPLExecutor


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

        # Extract query and context from messages
        query, context, images = self._extract_query_and_context(messages)

        # Initialize REPL environment
        repl_env = self._build_repl_env(query, context)

        # Build RLM conversation
        system_prompt = build_system_prompt(len(context), self._current_depth)
        rlm_messages: list[dict] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
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

            # Check for FINAL() or FINAL_VAR()
            if is_final(response_text):
                answer = parse_response(response_text, repl_env)
                if answer is not None:
                    return AIMessage(answer)

            # Execute code in REPL
            try:
                exec_result = self.repl.execute(response_text, repl_env)
            except REPLError as e:
                exec_result = f"Error: {str(e)}"
            except Exception as e:
                exec_result = f"Unexpected error: {str(e)}"
                logging.warning(f"RLM REPL error: {e}")

            # Add to conversation
            rlm_messages.append({"role": "assistant", "content": response_text})
            rlm_messages.append({"role": "user", "content": exec_result})

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
    ) -> tuple[str, str, list[dict]]:
        """
        Extract query and context from message list.

        Strategy:
        - System messages become part of context (instructions)
        - User text content becomes context
        - Last user message (text part) is the query
        - Images are collected separately to keep in prompt

        Args:
            messages: List of message dicts

        Returns:
            Tuple of (query, text_context, list_of_images)
        """
        context_parts = []
        images = []
        last_user_text = ""

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")

            if role == "system":
                # System messages become context
                text = self._extract_text_from_content(content)
                if text:
                    context_parts.append(f"[System Instructions]\n{text}")

            elif role == "user":
                # User messages: extract text and images
                text = self._extract_text_from_content(content)
                msg_images = self._extract_images_from_content(content)

                if text:
                    context_parts.append(f"[User]\n{text}")
                    last_user_text = text

                images.extend(msg_images)

            elif role == "assistant":
                # Assistant history can provide context
                text = self._extract_text_from_content(content)
                if text:
                    context_parts.append(f"[Assistant]\n{text}")

        # Combine context
        context = "\n\n".join(context_parts)

        # Use last user message as the query
        query = last_user_text or "Process the context and provide an answer."

        return query, context, images

    def _extract_text_from_content(self, content: str | list[dict]) -> str:
        """Extract text from message content (handles multimodal format)."""
        if isinstance(content, str):
            return content

        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        text_parts.append(item.get("text", ""))
                    elif item.get("type") == "input_text":
                        text_parts.append(item.get("input_text", ""))
            return "\n".join(text_parts)

        return ""

    def _extract_images_from_content(self, content: str | list[dict]) -> list[dict]:
        """Extract image items from message content."""
        if isinstance(content, str):
            return []

        if isinstance(content, list):
            images = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image_url":
                    images.append(item)
            return images

        return []

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

    def _build_repl_env(self, query: str, context: str) -> dict[str, Any]:
        """
        Build REPL environment with context, query, and recursive_llm function.

        Args:
            query: The user query
            context: The externalized context string

        Returns:
            Environment dict for REPL execution
        """
        return {
            "context": context,
            "query": query,
            "recursive_llm": self._make_recursive_fn(),
            "re": re,  # Pre-import re module
        }

    def _make_recursive_fn(self):
        """
        Create the recursive_llm function for REPL environment.

        Returns:
            A sync function that calls the recursive_model
        """

        def recursive_llm(sub_query: str, sub_context: str) -> str:
            """
            Recursively process sub-context with another LLM call.

            Args:
                sub_query: Query for the sub-context
                sub_context: The sub-context to process

            Returns:
                Answer from the recursive call
            """
            if self._current_depth + 1 >= self.max_depth:
                return f"Max recursion depth ({self.max_depth}) reached"

            # Create a sub-RLM with increased depth
            sub_rlm = RLMChatModel(
                inner_model=self.recursive_model,
                recursive_model=self.recursive_model,
                max_depth=self.max_depth,
                max_iterations=self.max_iterations,
                max_output_chars=self.max_output_chars,
                _current_depth=self._current_depth + 1,
            )

            # Build simple messages for the sub-call
            messages = [{"role": "user", "content": f"{sub_query}\n\nContext:\n{sub_context}"}]

            try:
                result = sub_rlm(messages)
                return result.get("content", "")
            except Exception as e:
                return f"Recursive call error: {str(e)}"

        return recursive_llm


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
