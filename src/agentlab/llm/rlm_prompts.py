"""
System prompt templates for RLM.

This module provides prompt building functions for the RLM approach.
The prompts instruct the model to explore context through a Python REPL
environment rather than having the context directly in the prompt.

The prompts are agent-aware - they understand that the final output
should be an action in the correct format for the downstream agent.

Based on the RLM paper (arXiv:2512.24601).
"""


def build_system_prompt(
    context_info: dict[str, int], action_format: str, depth: int = 0
) -> str:
    """
    Build system prompt for RLM with agent context awareness.

    Args:
        context_info: Dict mapping context keys to their character counts
        action_format: Action space description from the agent
        depth: Current recursion depth (0 = root)

    Returns:
        System prompt string
    """
    # Format context info as readable list
    context_desc = "\n".join(
        f"  - context['{key}']: {size:,} characters"
        for key, size in context_info.items()
        if size > 0
    )

    if not context_desc:
        context_desc = "  - context: (empty)"

    prompt = f"""You are a Recursive Language Model helping a web automation agent.

## Context Structure

The web page observations are stored in a `context` dictionary (NOT in this prompt).
You MUST write Python code to explore and search this context.

Available context keys:
{context_desc}

## Environment Variables

Available in the Python REPL:
- context: dict with keys 'axtree', 'html', 'goal', 'error', 'focused_element', 'tabs', 'history'
- query: str (the task to perform)
- recursive_llm(sub_query, sub_context) -> str (recursively process with another LLM)
- re: the regex module (already imported)

## How to Explore

Write Python code in ```python blocks to explore the context:

```python
# See available keys
print(context.keys())
```

```python
# View the accessibility tree (most useful for finding elements)
print(context['axtree'][:1000])
```

```python
# Search for an element
matches = re.findall(r'\\[\\d+\\].*button.*', context['axtree'], re.IGNORECASE)
print(matches[:5])
```

```python
# View HTML for detailed structure
print(context['html'][:500])
```

## CRITICAL RULES

1. ALWAYS write code in ```python blocks - raw text is NOT executed
2. ALWAYS search the context BEFORE deciding on an action
3. NEVER guess element IDs - find them in the axtree or html
4. The `bid` attribute identifies clickable elements

## Output Format

When you have found the correct element and are ready to act, call FINAL() with the action:

FINAL("<action>click('20')</action>")

Or for other actions:
FINAL("<action>fill('15', 'search text')</action>")
FINAL("<action>keyboard_press('Enter')</action>")

The action MUST be wrapped in <action></action> tags inside FINAL().

You can also use FINAL_VAR(variable_name) to return a variable from the REPL.

{_build_action_section(action_format)}

Depth: {depth}"""

    return prompt


def _build_action_section(action_format: str) -> str:
    """Build the action format section if available."""
    if not action_format:
        return ""

    return f"""## Action Space

{action_format}

Remember: Wrap your chosen action in <action></action> tags inside FINAL().
Example: FINAL("<action>click('bid_value')</action>")
"""


def build_user_prompt(query: str) -> str:
    """
    Build user prompt.

    Args:
        query: User's question/task

    Returns:
        User prompt string
    """
    return query


def build_iteration_context(exec_result: str) -> str:
    """
    Build the context message showing code execution results.

    Args:
        exec_result: The output from REPL execution

    Returns:
        Formatted message for the conversation
    """
    return exec_result
