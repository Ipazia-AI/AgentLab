"""
REPL executor for RLM.

This module provides a Python execution environment for the RLM
to explore and manipulate externalized context through code execution.

Based on the RLM paper (arXiv:2512.24601) approach of treating prompts as
external environment variables that can be explored programmatically.

NOTE: Currently running in DEBUG mode with no restrictions.
TODO: Add RestrictedPython sandboxing before production use.
"""

import io
import re
import sys
from typing import Any


class REPLError(Exception):
    """Error during REPL execution."""

    pass


class REPLExecutor:
    """
    Python code executor for RLM context exploration.

    Currently runs code without restrictions for debugging.
    """

    def __init__(self, timeout: int = 5, max_output_chars: int = 2000):
        self.timeout = timeout
        self.max_output_chars = max_output_chars

    def execute(self, response: str, env: dict[str, Any]) -> str:
        """
        Execute Python code from LLM response.

        Args:
            response: LLM response containing ```repl or ```python blocks
            env: Environment dict with context, query, etc.

        Returns:
            Execution output as string

        Raises:
            REPLError: If code execution fails
        """
        code, num_blocks = self._extract_code(response)

        if code is None:
            return (
                "No code block found. Write code in ```repl blocks.\n"
                "Example:\n```repl\nprint(context['axtree'])\n```"
            )

        if not code.strip():
            return "Empty code block."

        # Capture stdout
        old_stdout = sys.stdout
        sys.stdout = captured = io.StringIO()

        try:
            # Simple exec with env as globals
            exec(code, env)
            output = captured.getvalue()

            if not output:
                return "Code executed (no output)"

            if len(output) > self.max_output_chars:
                return (
                    f"{output[:self.max_output_chars]}\n\n"
                    f"[Truncated: {len(output)} chars, showing {self.max_output_chars}]"
                )

            return output.strip()

        except Exception as e:
            raise REPLError(f"Execution error: {e}") from e

        finally:
            sys.stdout = old_stdout

    def _extract_code(self, text: str) -> tuple[str | None, int]:
        """Extract and concatenate all ```repl or ```python blocks."""
        pattern = r"```(?:repl|python)\s*\n(.*?)```"
        matches = re.findall(pattern, text, re.DOTALL)

        if not matches:
            return None, 0

        code = "\n".join(match.strip() for match in matches)
        return code, len(matches)
