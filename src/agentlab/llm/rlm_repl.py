"""
Safe REPL executor for RLM using RestrictedPython.

This module provides a sandboxed Python execution environment for the RLM
to explore and manipulate externalized context through code execution.

Based on the RLM paper (arXiv:2512.24601) approach of treating prompts as
external environment variables that can be explored programmatically.
"""

import io
import sys
from typing import Any

from RestrictedPython import (
    compile_restricted_exec,
    limited_builtins,
    safe_globals,
    utility_builtins,
)
from RestrictedPython.Guards import guarded_iter_unpack_sequence, safer_getattr
from RestrictedPython.PrintCollector import PrintCollector


class REPLError(Exception):
    """Error during REPL execution."""

    pass


class REPLExecutor:
    """
    Safe Python code executor for RLM context exploration.

    Uses RestrictedPython to provide a sandboxed environment where the LLM
    can write code to explore context without accessing the file system or network.
    """

    def __init__(self, timeout: int = 5, max_output_chars: int = 2000):
        """
        Initialize REPL executor.

        Args:
            timeout: Execution timeout in seconds (not currently enforced)
            max_output_chars: Maximum characters to return (truncate if longer)
        """
        self.timeout = timeout
        self.max_output_chars = max_output_chars

    def execute(self, code: str, env: dict[str, Any]) -> str:
        """
        Execute Python code in restricted environment.

        Args:
            code: Python code to execute
            env: Environment with context, query, recursive_llm, etc.

        Returns:
            String result of execution (stdout or last expression)

        Raises:
            REPLError: If code execution fails
        """
        # Import here to avoid import errors if RestrictedPython not installed



        # Filter out code blocks if present (LLM might wrap code)
        code = self._extract_code(code)

        if not code.strip():
            return "No code to execute"

        # Build restricted globals
        restricted_globals = self._build_globals(
            env,
            safe_globals,
            limited_builtins,
            utility_builtins,
            guarded_iter_unpack_sequence,
            safer_getattr,
            PrintCollector,
        )

        # Capture stdout
        old_stdout = sys.stdout
        sys.stdout = captured_output = io.StringIO()

        try:
            # Compile with RestrictedPython
            byte_code = compile_restricted_exec(code)

            if byte_code.errors:
                raise REPLError(f"Compilation error: {', '.join(byte_code.errors)}")

            # Execute
            exec(byte_code.code, restricted_globals, env)

            # Get output from stdout
            output = captured_output.getvalue()

            # Get output from PrintCollector if available
            if "_print" in env and hasattr(env["_print"], "__call__"):
                print_collector = env["_print"]
                if hasattr(print_collector, "txt"):
                    output += "".join(print_collector.txt)

            # Check if last line was an expression (try to get its value)
            lines = code.strip().split("\n")
            if lines:
                last_line = lines[-1].strip()
                # If last line is a simple expression (no assignment, no keyword)
                keywords = ["=", "import", "def", "class", "if", "for", "while", "with"]
                if last_line and not any(kw in last_line for kw in keywords):
                    try:
                        result = eval(last_line, restricted_globals, env)
                        if result is not None:
                            output += str(result) + "\n"
                    except Exception:
                        pass  # Not an expression, ignore

            if not output:
                return "Code executed successfully (no output)"

            # Truncate output if too long (as per RLM paper: "truncated version of output")
            if len(output) > self.max_output_chars:
                truncated = output[: self.max_output_chars]
                return (
                    f"{truncated}\n\n[Output truncated: {len(output)} chars total, "
                    f"showing first {self.max_output_chars}]"
                )

            return output.strip()

        except REPLError:
            raise
        except Exception as e:
            raise REPLError(f"Execution error: {str(e)}") from e

        finally:
            sys.stdout = old_stdout

    def _extract_code(self, text: str) -> str:
        """
        Extract code from markdown code blocks if present.

        Args:
            text: Raw text that might contain code

        Returns:
            Extracted code
        """
        # Check for markdown code blocks with python tag
        if "```python" in text:
            start = text.find("```python") + len("```python")
            end = text.find("```", start)
            if end != -1:
                return text[start:end].strip()

        # Check for generic code blocks
        if "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            if end != -1:
                return text[start:end].strip()

        return text

    def _build_globals(
        self,
        env: dict[str, Any],
        safe_globals,
        limited_builtins,
        utility_builtins,
        guarded_iter_unpack_sequence,
        safer_getattr,
        PrintCollector,
    ) -> dict[str, Any]:
        """
        Build restricted globals for safe execution.

        Args:
            env: User environment
            safe_globals: RestrictedPython safe_globals
            limited_builtins: RestrictedPython limited_builtins
            utility_builtins: RestrictedPython utility_builtins
            guarded_iter_unpack_sequence: Guard function
            safer_getattr: Guard function
            PrintCollector: Print collector class

        Returns:
            Safe globals dict
        """
        import json
        import math
        import re
        from collections import Counter, defaultdict
        from datetime import datetime, timedelta

        restricted_globals = safe_globals.copy()
        restricted_globals.update(limited_builtins)
        restricted_globals.update(utility_builtins)

        # Add guards
        restricted_globals["_iter_unpack_sequence_"] = guarded_iter_unpack_sequence
        restricted_globals["_getattr_"] = safer_getattr
        restricted_globals["_getitem_"] = lambda obj, index: obj[index]
        restricted_globals["_getiter_"] = iter
        restricted_globals["_print_"] = PrintCollector

        # Add additional safe builtins
        restricted_globals.update(
            {
                # Types
                "len": len,
                "str": str,
                "int": int,
                "float": float,
                "bool": bool,
                "list": list,
                "dict": dict,
                "tuple": tuple,
                "set": set,
                "frozenset": frozenset,
                "bytes": bytes,
                "bytearray": bytearray,
                # Iteration
                "range": range,
                "enumerate": enumerate,
                "zip": zip,
                "map": map,
                "filter": filter,
                "reversed": reversed,
                "iter": iter,
                "next": next,
                # Aggregation
                "sorted": sorted,
                "sum": sum,
                "min": min,
                "max": max,
                "any": any,
                "all": all,
                # Math
                "abs": abs,
                "round": round,
                "pow": pow,
                "divmod": divmod,
                # String/repr
                "chr": chr,
                "ord": ord,
                "hex": hex,
                "oct": oct,
                "bin": bin,
                "repr": repr,
                "ascii": ascii,
                "format": format,
                # Type checking
                "isinstance": isinstance,
                "issubclass": issubclass,
                "callable": callable,
                "type": type,
                "hasattr": hasattr,
                # Constants
                "True": True,
                "False": False,
                "None": None,
                # Safe standard library modules
                "re": re,
                "json": json,
                "math": math,
                "datetime": datetime,
                "timedelta": timedelta,
                "Counter": Counter,
                "defaultdict": defaultdict,
            }
        )

        return restricted_globals
