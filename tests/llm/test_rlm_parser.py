import pytest

from agentlab.llm import rlm_parser


def test_find_final_answer_basic():
    text = "FINAL(click('a1'))"
    assert rlm_parser.find_final_answer(text) == ("FINAL", "click('a1')")


def test_find_final_answer_allows_whitespace():
    text = "  FINAL   (  click('a1')  )  "
    assert rlm_parser.find_final_answer(text) == ("FINAL", "click('a1')")


def test_find_final_answer_multiline():
    text = "intro\nFINAL(click('a1'))\nmore"
    assert rlm_parser.find_final_answer(text) == ("FINAL", "click('a1')")


def test_check_for_final_var_resolves_env():
    text = "FINAL_VAR('result')"
    env = {"result": "<action>click('a1')</action>"}
    assert rlm_parser.check_for_final_answer(text, env) == "<action>click('a1')</action>"


def test_check_for_final_var_missing_returns_none():
    text = "FINAL_VAR('missing')"
    env = {}
    assert rlm_parser.check_for_final_answer(text, env) is None
