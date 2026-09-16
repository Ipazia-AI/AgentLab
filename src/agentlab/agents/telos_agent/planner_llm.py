"""Adapt an AgentLab chat model to Telos's PlannerLLM protocol."""

from collections.abc import Sequence

from telos.core.interfaces import ChatMessage


def _content_to_text(content: str | list) -> str:
    if isinstance(content, str):
        return content
    return "\n".join(part for part in content if isinstance(part, str))


def _message_to_dict(msg: ChatMessage) -> dict:
    return {"role": msg.role, "content": _content_to_text(msg.content)}


class AgentLabPlannerLLM:
    """Wrap an AgentLab ``AbstractChatModel`` as a Telos ``PlannerLLM``.

    Text-only: image blocks in ``ChatMessage.content`` are dropped.
    """

    def __init__(self, chat_model):
        self._chat = chat_model

    def complete(self, messages: Sequence[ChatMessage]) -> str:
        payload = [_message_to_dict(msg) for msg in messages]
        reply = self._chat(payload)
        if isinstance(reply, dict):
            return reply.get("content") or ""
        return getattr(reply, "content", "") or ""

    def get_stats(self) -> dict:
        getter = getattr(self._chat, "get_stats", None)
        return getter() if getter else {}
