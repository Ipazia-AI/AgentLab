"""Ground a Telos NL action description into a BrowserGym action string."""

from dataclasses import dataclass
from typing import Any

from browsergym.core.action.base import AbstractActionSet

from agentlab.llm.llm_utils import (
    Discussion,
    HumanMessage,
    ParseError,
    SystemMessage,
    parse_html_tags_raise,
    retry,
)

_SYSTEM = (
    "You convert a natural-language web instruction into a single BrowserGym action. "
    "Use only the allowed actions and the element bids from the accessibility tree. "
    "Output exactly one action."
)


@dataclass
class GroundedAction:
    action: str | None
    raw: str = ""
    n_retry: int = 0
    busted_retry: int = 0
    chat_messages: Any = None


class Actor:
    def __init__(self, chat_model, action_set: AbstractActionSet, max_retry: int = 4):
        self._chat = chat_model
        self.action_set = action_set
        self.max_retry = max_retry

    def ground(self, instruction: str, axtree_txt: str) -> GroundedAction:
        action_space = self.action_set.describe(
            with_long_description=True, with_examples=True
        )
        messages = Discussion(
            [
                SystemMessage(_SYSTEM),
                HumanMessage(
                    f"# Instruction\n{instruction}\n\n"
                    f"# Action space\n{action_space}\n\n"
                    f"# Current page (accessibility tree)\n{axtree_txt}\n\n"
                    "Return exactly one action. The following is an example of the expected output format:\n"
                    "<action>\nclick('12')\n</action>"
                ),
            ]
        )

        def parser(response: str) -> dict:
            ans = parse_html_tags_raise(response, keys=["action"], merge_multiple=True)
            self.action_set.to_python_code(ans["action"])
            ans["raw"] = response
            return ans

        try:
            ans = retry(self._chat, messages, n_retry=self.max_retry, parser=parser)
            n_retry = max(0, (len(messages) - 3) / 2)
            return GroundedAction(
                action=ans["action"],
                raw=ans.get("raw", ""),
                n_retry=n_retry,
                busted_retry=0,
                chat_messages=messages,
            )
        except ParseError:
            raw = messages[-1]["content"] if messages else ""
            return GroundedAction(
                action=None,
                raw=raw if isinstance(raw, str) else str(raw),
                n_retry=self.max_retry,
                busted_retry=1,
                chat_messages=messages,
            )
