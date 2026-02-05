"""
Parse FINAL() and FINAL_VAR() statements from LLM responses.

Based on the RLM paper (arXiv:2512.24601) termination signal patterns.
FINAL/FINAL_VAR must be at the start of a line (with optional whitespace).
"""

import re
from typing import Any


def _extract_balanced(text: str, start: int) -> str | None:
    """Extract content with balanced parens, respecting quotes."""
    depth, i, n = 1, start, len(text)
    in_quote = None
    
    while i < n:
        c = text[i]
        if in_quote:
            if c == in_quote and text[i-1:i] != "\\":
                in_quote = None
        elif c in "\"'":
            in_quote = c
        elif c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                return text[start:i]
        i += 1
    return None


def find_final_answer(text: str) -> tuple[str, str] | None:
    """
    Find FINAL(...) or FINAL_VAR(...) at start of line.
    Returns (type, content) or None.
    """
    # Single regex for both patterns
    match = re.search(r"^\s*(FINAL(?:_VAR)?)\(", text, re.MULTILINE)
    if match:
        content = _extract_balanced(text, match.end())
        if content is not None:
            return (match.group(1), content.strip())
    return None


def check_for_final_answer(response: str, repl_env: dict[str, Any]) -> str | None:
    """Extract final answer, resolving FINAL_VAR from repl_env if needed."""
    result = find_final_answer(response)
    if not result:
        return None

    answer_type, content = result
    
    if answer_type == "FINAL":
        return content
    
    # FINAL_VAR: lookup variable in environment
    var_name = content.strip().strip("\"'")
    return str(repl_env[var_name]) if var_name in repl_env else None


def is_final(response: str) -> bool:
    """Check if response contains FINAL() or FINAL_VAR() at start of line."""
    return find_final_answer(response) is not None
