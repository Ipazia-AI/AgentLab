"""
Tests for RLM (Recursive Language Model) chat model decorator.

This module tests the RLMChatModel wrapper that adds long-context handling
to any AbstractChatModel through iterative REPL exploration.
"""

import os

import pytest

from agentlab.llm.base_api import AbstractChatModel, BaseModelArgs
from agentlab.llm.rlm_chat_model import (
    MaxDepthError,
    MaxIterationsError,
    RLMChatModel,
    RLMError,
    RLMModelArgs,
)
from agentlab.llm.rlm_parser import extract_final, extract_final_var, is_final, parse_response
from agentlab.llm.rlm_prompt_parser import (
    ParsedPrompt,
    build_context_dict,
    get_context_info,
    parse_agent_prompt,
)
from agentlab.llm.rlm_prompts import build_system_prompt
from agentlab.llm.rlm_repl import REPLError, REPLExecutor


# ==============================================================================
# Test Fixtures and Mock Classes
# ==============================================================================


class MockChatModel(AbstractChatModel):
    """Mock chat model for testing that returns predefined responses."""

    def __init__(self, responses: list[str] | None = None):
        """
        Initialize mock model.

        Args:
            responses: List of responses to return in sequence.
                       If None, returns a simple FINAL response.
        """
        self.responses = responses or ['FINAL("mock answer")']
        self.call_count = 0
        self.received_messages = []

    def __call__(self, messages: list[dict], **kwargs) -> dict:
        self.received_messages.append(messages)
        response = self.responses[min(self.call_count, len(self.responses) - 1)]
        self.call_count += 1
        return {"role": "assistant", "content": response}

    def get_stats(self) -> dict:
        return {"call_count": self.call_count}


from dataclasses import dataclass, field


@dataclass
class MockModelArgs(BaseModelArgs):
    """Mock model args for testing."""

    model_name: str = "mock-model"
    responses: list[str] | None = field(default=None)

    def make_model(self) -> MockChatModel:
        return MockChatModel(responses=self.responses)


# ==============================================================================
# REPL Executor Tests
# ==============================================================================


class TestREPLExecutor:
    """Tests for the REPL executor."""

    def test_simple_print(self):
        """Test basic print statement execution."""
        repl = REPLExecutor()
        env = {"context": {"axtree": "hello world"}, "query": "test"}
        result = repl.execute("```python\nprint(context['axtree'])\n```", env)
        assert "hello world" in result

    def test_context_slicing(self):
        """Test slicing context variable."""
        repl = REPLExecutor()
        env = {"context": {"axtree": "hello world"}, "query": "test"}
        result = repl.execute("```python\nprint(context['axtree'][:5])\n```", env)
        assert "hello" in result

    def test_regex_search(self):
        """Test regex operations on context."""
        import re as re_module
        repl = REPLExecutor()
        env = {"context": {"axtree": "The answer is 42."}, "query": "find number", "re": re_module}
        # Note: 're' is pre-imported in the REPL environment, no need to import it
        result = repl.execute("```python\nprint(re.findall(r'\\d+', context['axtree']))\n```", env)
        assert "42" in result

    def test_expression_evaluation(self):
        """Test that last expression value is returned."""
        repl = REPLExecutor()
        env = {"context": {"axtree": "test"}, "query": "test"}
        result = repl.execute("```python\nlen(context['axtree'])\n```", env)
        assert "4" in result

    def test_code_block_extraction(self):
        """Test extraction of code from markdown blocks."""
        repl = REPLExecutor()
        env = {"context": {"axtree": "test"}, "query": "test"}

        code_with_block = """
```python
print("hello")
```
"""
        result = repl.execute(code_with_block, env)
        assert "hello" in result

    def test_output_truncation(self):
        """Test that long output is truncated."""
        repl = REPLExecutor(max_output_chars=50)
        env = {"context": {"axtree": "x" * 1000}, "query": "test"}
        result = repl.execute("```python\nprint(context['axtree'])\n```", env)
        assert "truncated" in result.lower()

    def test_compilation_error(self):
        """Test that syntax errors are caught."""
        repl = REPLExecutor()
        env = {"context": {"axtree": "test"}, "query": "test"}
        with pytest.raises(REPLError) as exc_info:
            repl.execute("```python\ndef incomplete(\n```", env)
        assert "error" in str(exc_info.value).lower()

    def test_no_file_access(self):
        """Test that file operations are blocked."""
        repl = REPLExecutor()
        env = {"context": "test", "query": "test"}
        with pytest.raises(REPLError):
            repl.execute("```python\nopen('/etc/passwd')\n```", env)

    def test_raw_text_not_executed(self):
        """Test that raw text without code blocks is NOT executed."""
        repl = REPLExecutor()
        env = {"context": {"axtree": "test", "html": "test"}, "query": "test"}

        # This raw text should NOT be executed as code
        result = repl.execute("click('20')", env)
        assert "No code block found" in result

        # Also test partial action-like text
        result = repl.execute("I'll click the button with bid 20", env)
        assert "No code block found" in result

    def test_dict_context_access(self):
        """Test that dict context can be accessed in code."""
        repl = REPLExecutor()
        env = {
            "context": {
                "axtree": "RootWebArea\n  [20] button 'Close'",
                "html": "<button bid='20'>Close</button>",
                "goal": "Close the dialog",
            },
            "query": "test",
        }

        result = repl.execute("```python\nprint(context['axtree'])\n```", env)
        assert "button" in result
        assert "Close" in result

    def test_dict_context_keys(self):
        """Test listing dict context keys."""
        repl = REPLExecutor()
        env = {
            "context": {
                "axtree": "tree",
                "html": "html",
                "goal": "goal",
            },
            "query": "test",
        }

        result = repl.execute("```python\nprint(list(context.keys()))\n```", env)
        assert "axtree" in result
        assert "html" in result


# ==============================================================================
# Parser Tests
# ==============================================================================


class TestParser:
    """Tests for the FINAL() parser."""

    def test_is_final_positive(self):
        """Test detection of FINAL statements."""
        assert is_final('FINAL("answer")')
        assert is_final("FINAL('answer')")
        assert is_final("FINAL_VAR(my_var)")

    def test_is_final_negative(self):
        """Test non-FINAL statements."""
        assert not is_final("No final here")
        assert not is_final("FINALE()")
        assert not is_final("final()")  # Case sensitive

    def test_extract_final_double_quotes(self):
        """Test extraction with double quotes."""
        result = extract_final('FINAL("hello world")')
        assert result == "hello world"

    def test_extract_final_single_quotes(self):
        """Test extraction with single quotes."""
        result = extract_final("FINAL('hello world')")
        assert result == "hello world"

    def test_extract_final_triple_quotes(self):
        """Test extraction with triple quotes."""
        result = extract_final('FINAL("""multi\nline""")')
        assert "multi" in result and "line" in result

    def test_extract_final_var(self):
        """Test extraction of variable reference."""
        env = {"my_answer": "the answer is 42"}
        result = extract_final_var("FINAL_VAR(my_answer)", env)
        assert result == "the answer is 42"

    def test_extract_final_var_missing(self):
        """Test extraction with missing variable."""
        env = {"other_var": "value"}
        result = extract_final_var("FINAL_VAR(my_answer)", env)
        assert result is None

    def test_parse_response_final(self):
        """Test parse_response with FINAL."""
        result = parse_response('FINAL("test")', {})
        assert result == "test"

    def test_parse_response_final_var(self):
        """Test parse_response with FINAL_VAR."""
        env = {"result": "computed value"}
        result = parse_response("FINAL_VAR(result)", env)
        assert result == "computed value"


# ==============================================================================
# Prompts Tests
# ==============================================================================


class TestPrompts:
    """Tests for prompt building functions."""

    def test_system_prompt_contains_essentials(self):
        """Test that system prompt contains required elements."""
        context_info = {"axtree": 5000, "html": 10000}
        action_format = "click(bid: str) - Click an element"
        prompt = build_system_prompt(context_info=context_info, action_format=action_format, depth=0)

        assert "context" in prompt.lower()
        assert "FINAL" in prompt
        assert "python" in prompt.lower()
        assert "axtree" in prompt.lower()
        assert "<action>" in prompt  # Action tags are explained
        assert "click" in prompt  # Action format is included

    def test_system_prompt_depth(self):
        """Test that depth is included in prompt."""
        prompt = build_system_prompt(context_info={"axtree": 100}, action_format="", depth=2)
        assert "2" in prompt

    def test_system_prompt_context_keys(self):
        """Test that context keys and sizes are listed."""
        context_info = {"axtree": 1234, "html": 5678, "history": 999}
        prompt = build_system_prompt(context_info=context_info, action_format="", depth=0)

        assert "axtree" in prompt
        assert "html" in prompt
        assert "history" in prompt
        assert "1,234" in prompt  # Formatted size


# ==============================================================================
# Prompt Parser Tests
# ==============================================================================


class TestPromptParser:
    """Tests for the GenericAgent prompt parser."""

    def test_parse_simple_task(self):
        """Test parsing a simple task from chat messages."""
        messages = [
            {"role": "system", "content": "You are a web assistant."},
            {
                "role": "user",
                "content": """## Chat messages:

 - [user] UTC Time: Mon Feb 2 14:00:00 2026 - Close the dialog by clicking the "x".

# Observation of current step:

## AXTree:
RootWebArea 'Test'
    [20] button 'Close'

## HTML:
<button bid="20">Close</button>

# Action space:
click(bid: str) - Click an element.
""",
            },
        ]
        parsed = parse_agent_prompt(messages)

        assert 'Close the dialog by clicking the "x"' in parsed.task
        assert "button" in parsed.axtree
        assert "<button" in parsed.html
        assert "click" in parsed.action_format

    def test_parse_with_history(self):
        """Test parsing prompt with history section."""
        messages = [
            {
                "role": "user",
                "content": """## Chat messages:
 - [user] 2026-02-02 - Do the task.

# History of interaction with the task:
Step 1: Clicked button 1
Step 2: Filled form

# Observation of current step:

## AXTree:
[10] input 'Name'

# Action space:
fill(bid, text) - Fill a form field.
""",
            },
        ]
        parsed = parse_agent_prompt(messages)

        assert "Clicked button" in parsed.history
        assert "Filled form" in parsed.history

    def test_parse_multimodal(self):
        """Test parsing messages with images."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "## Chat messages:\n - [user] Click the red button.\n\n## AXTree:\n[1] button 'Red'"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc123"}},
                ],
            },
        ]
        parsed = parse_agent_prompt(messages)

        assert "Click the red button" in parsed.task
        assert len(parsed.images) == 1
        assert parsed.images[0]["type"] == "image_url"

    def test_build_context_dict(self):
        """Test building context dict from parsed prompt."""
        parsed = ParsedPrompt(
            task="Click the button",
            axtree="[20] button 'Submit'",
            html="<button>Submit</button>",
            history="Step 1: Opened page",
            error="Previous click failed",
        )

        context = build_context_dict(parsed)

        assert context["axtree"] == "[20] button 'Submit'"
        assert context["html"] == "<button>Submit</button>"
        assert context["goal"] == "Click the button"
        assert context["history"] == "Step 1: Opened page"
        assert context["error"] == "Previous click failed"

    def test_get_context_info(self):
        """Test getting context size information."""
        parsed = ParsedPrompt(
            axtree="x" * 1000,
            html="y" * 500,
            history="",  # Empty should not appear
        )

        info = get_context_info(parsed)

        assert info["axtree"] == 1000
        assert info["html"] == 500
        assert "history" not in info  # Empty strings excluded

    def test_parse_goal_section(self):
        """Test parsing non-chat mode with Goal section."""
        messages = [
            {
                "role": "user",
                "content": """## Goal:
Navigate to the settings page.

# Observation of current step:

## AXTree:
[5] link 'Settings'
""",
            },
        ]
        parsed = parse_agent_prompt(messages)

        assert "Navigate to the settings page" in parsed.task


# ==============================================================================
# RLMChatModel Tests
# ==============================================================================


class TestRLMChatModel:
    """Tests for the RLMChatModel decorator."""

    def test_simple_final_response(self):
        """Test that a simple FINAL response is returned."""
        mock = MockChatModel(['FINAL("the answer")'])
        rlm = RLMChatModel(inner_model=mock, max_iterations=5)

        messages = [{"role": "user", "content": "What is the answer?"}]
        result = rlm(messages)

        assert result["content"] == "the answer"
        assert mock.call_count == 1

    def test_iterative_exploration(self):
        """Test that RLM iterates through REPL execution."""
        responses = [
            "```python\nprint(context['axtree'][:10])\n```",  # First: explore context
            'FINAL("<action>click(\'20\')</action>")',  # Second: provide answer
        ]
        mock = MockChatModel(responses)
        rlm = RLMChatModel(inner_model=mock, max_iterations=5)

        messages = [
            {
                "role": "user",
                "content": """## Chat messages:
 - [user] Find something

## AXTree:
hello world
""",
            }
        ]
        result = rlm(messages)

        assert "<action>click('20')</action>" in result["content"]
        assert mock.call_count == 2

    def test_max_iterations_error(self):
        """Test that max iterations raises error."""
        mock = MockChatModel(["```python\nprint('still searching...')\n```"])  # Never returns FINAL
        rlm = RLMChatModel(inner_model=mock, max_iterations=3)

        messages = [{"role": "user", "content": "## Chat messages:\n - [user] test"}]
        with pytest.raises(MaxIterationsError):
            rlm(messages)

        assert mock.call_count == 3

    def test_max_depth_error(self):
        """Test that max depth is enforced."""
        mock = MockChatModel()
        rlm = RLMChatModel(inner_model=mock, max_depth=2, _current_depth=2)

        messages = [{"role": "user", "content": "test"}]
        with pytest.raises(MaxDepthError):
            rlm(messages)

    def test_stats_tracking(self):
        """Test that stats are properly tracked."""
        responses = [
            "```python\nprint(len(context['axtree']))\n```",
            'FINAL("done")',
        ]
        mock = MockChatModel(responses)
        rlm = RLMChatModel(inner_model=mock)

        messages = [{"role": "user", "content": "## Chat messages:\n - [user] test\n\n## AXTree:\ntest"}]
        rlm(messages)

        stats = rlm.get_stats()
        assert stats["rlm_llm_calls"] == 2
        assert stats["rlm_iterations"] == 2
        assert "inner_call_count" in stats

    def test_multimodal_messages(self):
        """Test handling of messages with images."""
        mock = MockChatModel(['FINAL("processed")'])
        rlm = RLMChatModel(inner_model=mock)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc123"}},
                ],
            }
        ]
        result = rlm(messages)

        assert result["content"] == "processed"
        # Check that images were preserved
        received = mock.received_messages[0]
        user_msg = [m for m in received if m.get("role") == "user"][0]
        assert isinstance(user_msg["content"], list)

    def test_context_extraction(self):
        """Test that context is properly extracted from messages."""
        mock = MockChatModel(['FINAL("ok")'])
        rlm = RLMChatModel(inner_model=mock)

        messages = [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Process this: important data"},
        ]
        rlm(messages)

        # Check that the model received the RLM system prompt
        received = mock.received_messages[0]
        system_msg = received[0]
        assert "REPL" in system_msg["content"] or "context" in system_msg["content"]


# ==============================================================================
# RLMModelArgs Tests
# ==============================================================================


class TestRLMModelArgs:
    """Tests for the RLMModelArgs configuration class."""

    def test_requires_inner_model_args(self):
        """Test that inner_model_args is required."""
        with pytest.raises(ValueError):
            RLMModelArgs(inner_model_args=None)

    def test_model_name_derivation(self):
        """Test that model_name is derived from inner model."""
        inner_args = MockModelArgs(model_name="test-model")
        args = RLMModelArgs(inner_model_args=inner_args)
        assert "rlm(" in args.model_name
        assert "test-model" in args.model_name

    def test_make_model(self):
        """Test that make_model creates RLMChatModel."""
        inner_args = MockModelArgs()
        args = RLMModelArgs(inner_model_args=inner_args, max_iterations=10)

        model = args.make_model()

        assert isinstance(model, RLMChatModel)
        assert model.max_iterations == 10

    def test_recursive_model_args(self):
        """Test separate recursive model configuration."""
        inner_args = MockModelArgs(model_name="main-model")
        recursive_args = MockModelArgs(model_name="recursive-model")
        args = RLMModelArgs(
            inner_model_args=inner_args,
            recursive_model_args=recursive_args,
        )

        model = args.make_model()

        # The models should be different instances
        assert model.inner_model is not model.recursive_model


# ==============================================================================
# Integration Tests (require API keys)
# ==============================================================================


@pytest.mark.pricy
@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="Skipping - no OpenAI API key")
def test_rlm_with_openai():
    """Integration test with real OpenAI model."""
    from agentlab.llm.chat_api import OpenAIModelArgs

    args = RLMModelArgs(
        inner_model_args=OpenAIModelArgs(
            model_name="gpt-4o-mini",
            max_new_tokens=512,
            temperature=0.1,
        ),
        max_iterations=5,
    )
    model = args.make_model()

    # A simple task that requires exploring context
    context = "The secret number hidden in this text is 42."
    messages = [
        {"role": "user", "content": f"Find the secret number in this context:\n\n{context}"}
    ]

    result = model(messages)
    assert "42" in result["content"]


if __name__ == "__main__":
    # Run a quick smoke test
    test_repl = TestREPLExecutor()
    test_repl.test_simple_print()
    print("REPL tests passed!")

    test_parser = TestParser()
    test_parser.test_is_final_positive()
    test_parser.test_extract_final_double_quotes()
    print("Parser tests passed!")

    test_rlm = TestRLMChatModel()
    test_rlm.test_simple_final_response()
    print("RLM tests passed!")

    print("\nAll smoke tests passed!")
