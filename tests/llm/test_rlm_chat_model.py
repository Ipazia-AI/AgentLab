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
        env = {"context": "hello world", "query": "test"}
        result = repl.execute("print(context)", env)
        assert "hello world" in result

    def test_context_slicing(self):
        """Test slicing context variable."""
        repl = REPLExecutor()
        env = {"context": "hello world", "query": "test"}
        result = repl.execute("print(context[:5])", env)
        assert "hello" in result

    def test_regex_search(self):
        """Test regex operations on context."""
        import re as re_module
        repl = REPLExecutor()
        env = {"context": "The answer is 42.", "query": "find number", "re": re_module}
        # Note: 're' is pre-imported in the REPL environment, no need to import it
        result = repl.execute("print(re.findall(r'\\d+', context))", env)
        assert "42" in result

    def test_expression_evaluation(self):
        """Test that last expression value is returned."""
        repl = REPLExecutor()
        env = {"context": "test", "query": "test"}
        result = repl.execute("len(context)", env)
        assert "4" in result

    def test_code_block_extraction(self):
        """Test extraction of code from markdown blocks."""
        repl = REPLExecutor()
        env = {"context": "test", "query": "test"}

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
        env = {"context": "x" * 1000, "query": "test"}
        result = repl.execute("print(context)", env)
        assert "truncated" in result.lower()

    def test_compilation_error(self):
        """Test that syntax errors are caught."""
        repl = REPLExecutor()
        env = {"context": "test", "query": "test"}
        with pytest.raises(REPLError) as exc_info:
            repl.execute("def incomplete(", env)
        assert "error" in str(exc_info.value).lower()

    def test_no_file_access(self):
        """Test that file operations are blocked."""
        repl = REPLExecutor()
        env = {"context": "test", "query": "test"}
        with pytest.raises(REPLError):
            repl.execute("open('/etc/passwd')", env)


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
        prompt = build_system_prompt(context_size=10000, depth=0)
        assert "context" in prompt.lower()
        assert "FINAL" in prompt
        assert "REPL" in prompt or "Python" in prompt
        assert "10,000" in prompt  # Formatted context size

    def test_system_prompt_depth(self):
        """Test that depth is included in prompt."""
        prompt = build_system_prompt(context_size=100, depth=2)
        assert "2" in prompt


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
            "print(context[:10])",  # First: explore context
            'FINAL("found it")',  # Second: provide answer
        ]
        mock = MockChatModel(responses)
        rlm = RLMChatModel(inner_model=mock, max_iterations=5)

        messages = [{"role": "user", "content": "Find something in: hello world"}]
        result = rlm(messages)

        assert result["content"] == "found it"
        assert mock.call_count == 2

    def test_max_iterations_error(self):
        """Test that max iterations raises error."""
        mock = MockChatModel(["print('still searching...')"])  # Never returns FINAL
        rlm = RLMChatModel(inner_model=mock, max_iterations=3)

        messages = [{"role": "user", "content": "test"}]
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
            "print(len(context))",
            'FINAL("done")',
        ]
        mock = MockChatModel(responses)
        rlm = RLMChatModel(inner_model=mock)

        messages = [{"role": "user", "content": "test"}]
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
