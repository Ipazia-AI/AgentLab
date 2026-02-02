"""Tests for RLMGenericAgent."""

import re
from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest
from bgym import HighLevelActionSetArgs

from agentlab.agents.dynamic_prompting import ActionFlags, ObsFlags
from agentlab.agents.rlm_agent import (
    REPLError,
    REPLExecutor,
    RLMGenericAgent,
    RLMGenericAgentArgs,
    RLMPromptFlags,
    extract_final,
    is_final,
    parse_response,
)
from agentlab.llm.chat_api import BaseModelArgs
from agentlab.llm.llm_utils import Discussion

# ============================================================================
# REPL Tests
# ============================================================================

class TestREPLExecutor:
    """Tests for REPLExecutor."""
    
    def test_simple_expression(self):
        repl = REPLExecutor()
        env = {}
        repl.execute("x = 5 + 3", env)
        assert env['x'] == 8
    
    def test_context_access(self):
        repl = REPLExecutor()
        env = {'context': {'axtree': '[1] button "Submit"'}}
        result = repl.execute("print(context['axtree'])", env)
        assert 'Submit' in result
    
    def test_regex_operations(self):
        repl = REPLExecutor()
        env = {'context': {'html': 'bid="123" bid="456"'}}
        repl.execute('matches = re.findall(r\'bid="(\\d+)"\', context["html"])', env)
        assert env['matches'] == ['123', '456']
    
    def test_code_block_extraction(self):
        text = '```python\nx = 5\n```'
        repl = REPLExecutor()
        env = {}
        repl.execute(text, env)
        assert env['x'] == 5
    
    def test_empty_code(self):
        repl = REPLExecutor()
        result = repl.execute("", {})
        assert "No code" in result
    
    def test_output_truncation(self):
        repl = REPLExecutor(max_output_chars=50)
        result = repl.execute("print('x' * 200)", {})
        assert 'truncated' in result.lower()


# ============================================================================
# Parser Tests
# ============================================================================

class TestParser:
    """Tests for RLM parser functions."""
    
    def test_is_final_true(self):
        assert is_final('FINAL("answer")')
        assert is_final("FINAL('answer')")
        assert is_final('FINAL_VAR(result)')
    
    def test_is_final_false(self):
        assert not is_final('No final here')
        assert not is_final('FINALLY done')
    
    def test_extract_final_double_quotes(self):
        assert extract_final('FINAL("click(\'1\')")') == "click('1')"
    
    def test_extract_final_single_quotes(self):
        assert extract_final("FINAL('click(\"1\")')") == 'click("1")'
    
    def test_extract_final_triple_quotes(self):
        result = extract_final('FINAL("""<action>click(\'1\')</action>""")')
        assert '<action>' in result
    
    def test_parse_response_final(self):
        assert parse_response('FINAL("answer")', {}) == "answer"
    
    def test_parse_response_final_var(self):
        env = {'result': 'computed'}
        assert parse_response('FINAL_VAR(result)', env) == 'computed'
    
    def test_parse_response_none(self):
        assert parse_response("No final", {}) is None


# ============================================================================
# Agent Tests
# ============================================================================

@dataclass
class MockModelArgs(BaseModelArgs):
    """Mock model args for testing."""
    model_name: str = "mock-model"
    
    def make_model(self):
        return MockLLM()


class MockLLM:
    """Mock LLM for testing. Returns dict format like real LLMs."""
    
    def __init__(self):
        self.call_count = 0
        self.responses = ['FINAL("<action>noop()</action>")']
        self._stats = {"prompt_tokens": 100, "completion_tokens": 50}
    
    def set_responses(self, responses):
        self.responses = responses
        self.call_count = 0
    
    def __call__(self, messages):
        if self.call_count < len(self.responses):
            response = self.responses[self.call_count]
        else:
            response = 'FINAL("<action>noop()</action>")'
        self.call_count += 1
        # Return dict format like real LLMs do
        return {"role": "assistant", "content": response}
    
    def get_stats(self):
        return self._stats.copy()


def make_test_obs():
    """Create test observation."""
    return {
        'axtree_txt': '[1] button "Submit" [clickable]',
        'pruned_html': '<button>Submit</button>',
        'url': 'https://example.com',
        'open_pages_urls': ['https://example.com'],
        'open_pages_titles': ['Test'],
        'active_page_index': 0,
        'last_action_error': '',
        'goal_object': [{'text': 'Click Submit'}],
    }


def make_test_agent():
    """Create test agent."""
    flags = RLMPromptFlags(
        obs=ObsFlags(use_html=True, use_ax_tree=True),
        action=ActionFlags(
            action_set=HighLevelActionSetArgs(subsets=["bid"], multiaction=False, strict=False),
        ),
    )
    return RLMGenericAgent(
        chat_model_args=MockModelArgs(),
        flags=flags,
        max_iterations=5,
    )


class TestRLMGenericAgent:
    """Tests for RLMGenericAgent."""
    
    def test_agent_creation(self):
        agent = make_test_agent()
        assert agent is not None
        assert agent.max_iterations == 5
    
    def test_agent_reset(self):
        agent = make_test_agent()
        agent.actions = ["click('1')"]
        agent.reset(seed=42)
        assert agent.actions == []
        assert agent.seed == 42
    
    def test_build_repl_context(self):
        agent = make_test_agent()
        obs = make_test_obs()
        context = agent._build_repl_context(obs)
        assert 'axtree' in context
        assert 'Submit' in context['axtree']
    
    def test_extract_goal(self):
        agent = make_test_agent()
        obs = make_test_obs()
        goal = agent._extract_goal(obs)
        assert 'Click Submit' in goal
    
    def test_extract_action_with_tags(self):
        agent = make_test_agent()
        action = agent._extract_action("<action>click('1')</action>")
        assert action == "click('1')"
    
    def test_get_action_immediate_final(self):
        agent = make_test_agent()
        obs = make_test_obs()
        agent.chat_llm.set_responses(['FINAL("<action>click(\'1\')</action>")'])
        
        action, info = agent.get_action(obs)
        
        assert action == "click('1')"
        assert info.stats['rlm_iterations'] == 1
    
    def test_get_action_with_exploration(self):
        agent = make_test_agent()
        obs = make_test_obs()
        agent.chat_llm.set_responses([
            'print(context["axtree"])',
            'FINAL("<action>click(\'1\')</action>")',
        ])
        
        action, info = agent.get_action(obs)
        
        assert action == "click('1')"
        assert info.stats['rlm_iterations'] == 2
