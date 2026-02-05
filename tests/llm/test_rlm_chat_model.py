"""
Tests for RLM (Recursive Language Model) chat model.
Focused on core functionality: REPL execution, FINAL parsing, and integration.
"""

import os

import pytest
from dotenv import find_dotenv, load_dotenv

from agentlab.llm.rlm_chat_model import MaxDepthError, RLMChatModel, RLMModelArgs
from agentlab.llm.rlm_parser import (
    extract_final,
    extract_final_var,
    is_final,
    parse_response,
)
from agentlab.llm.rlm_prompts import build_system_prompt
from agentlab.llm.rlm_repl import REPLError, REPLExecutor

load_dotenv(find_dotenv())

HAS_OPENROUTER_KEY = bool(os.getenv("OPENROUTER_API_KEY"))


class TestREPL:
    """Core REPL functionality tests."""

    def test_execute_and_print(self):
        """REPL executes code and captures print output."""
        repl = REPLExecutor()
        env = {"context": {"axtree": "hello", "html": "<div>test</div>"}, "query": "test"}
        result = repl.execute("```repl\nprint(context['axtree'])\n```", env)
        assert "hello" in result

    def test_truncation(self):
        """Long output gets truncated."""
        repl = REPLExecutor(max_output_chars=50)
        env = {"context": {"axtree": "x" * 1000}, "query": "test"}
        result = repl.execute("```repl\nprint(context['axtree'])\n```", env)
        assert "truncated" in result.lower()

    def test_syntax_error(self):
        """Syntax errors raise REPLError."""
        repl = REPLExecutor()
        with pytest.raises(REPLError):
            repl.execute("```repl\ndef broken(\n```", {})


class TestFinalParsing:
    """FINAL/FINAL_VAR parsing tests."""

    def test_final_detection_and_extraction(self):
        """Detect and extract FINAL statements."""
        assert is_final('FINAL("answer")')
        assert is_final("FINAL_VAR(var)")
        assert not is_final("No final here")
        assert extract_final('FINAL("click(20)")') == "click(20)"

    def test_final_var(self):
        """Extract value from FINAL_VAR."""
        env = {"result": "computed"}
        assert extract_final_var("FINAL_VAR(result)", env) == "computed"
        assert extract_final_var("FINAL_VAR(missing)", env) is None

    def test_parse_response(self):
        """parse_response handles both FINAL and FINAL_VAR."""
        assert parse_response('FINAL("test")', {}) == "test"
        assert parse_response("FINAL_VAR(x)", {"x": "value"}) == "value"


class TestSystemPrompt:
    """System prompt validation."""

    def test_contains_essentials(self):
        """System prompt has required RLM elements."""
        prompt = build_system_prompt()
        assert "context" in prompt.lower()
        assert "FINAL" in prompt
        assert "llm_query" in prompt
        assert "```repl" in prompt


# Integration tests (require API key)

def make_mock_obs(goal: str, axtree: str) -> dict:
    """Create mock obs dict."""
    return {
        "goal_object": [{"type": "text", "text": goal}],
        "axtree_txt": axtree,
        "pruned_html": "",
        "last_action_error": "",
    }


@pytest.mark.pricy
@pytest.mark.skipif(not HAS_OPENROUTER_KEY, reason="No OPENROUTER_API_KEY")
class TestRLMIntegration:
    """Integration tests with real LLM."""

    def test_simple_task(self):
        """RLM finds element and returns action."""
        from agentlab.llm.chat_api import OpenRouterModelArgs

        rlm = RLMModelArgs(
            inner_model_args=OpenRouterModelArgs(model_name="openai/gpt-4o-mini", max_new_tokens=512),
            max_iterations=10,
        ).make_model()

        rlm.obs = make_mock_obs(
            goal="Click the OK button.",
            axtree="[10] button 'Cancel'\n[20] button 'OK'",
        )

        messages = [{"role": "user", "content": "# Action space:\nclick(bid: str)"}]
        result = rlm(messages)

        assert result["role"] == "assistant"
        assert "20" in result["content"] or "OK" in result["content"]

    def test_max_depth_enforced(self):
        """MaxDepthError raised when depth exceeded."""
        from agentlab.llm.chat_api import OpenRouterModelArgs

        inner = OpenRouterModelArgs(model_name="openai/gpt-4o-mini").make_model()
        rlm = RLMChatModel(inner_model=inner, max_depth=2, _current_depth=2)
        rlm.obs = make_mock_obs("Test", "[1] button")

        with pytest.raises(MaxDepthError):
            rlm([{"role": "user", "content": "# Action space:\nclick(bid)"}])
