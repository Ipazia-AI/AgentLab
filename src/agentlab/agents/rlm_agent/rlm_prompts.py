"""
RLM prompts adapted for GUI web automation agents.

This module provides system prompts and user prompts for the RLM agent
that guide the model to explore browser observations programmatically
and produce valid actions for web automation.
"""


def build_system_prompt(
    context_info: dict[str, int],
    action_set_description: str,
    depth: int = 0,
) -> str:
    """
    Build system prompt for RLM GUI agent.

    Args:
        context_info: Dict with sizes of context components (e.g., {'axtree': 5000, 'html': 3000})
        action_set_description: Description of available actions from AgentLab
        depth: Current recursion depth (0 = root)

    Returns:
        System prompt string
    """
    context_summary = ", ".join(f"{k}: {v:,} chars" for k, v in context_info.items())
    
    prompt = f"""You are a Recursive Language Model (RLM) agent for web automation.

IMPORTANT: The browser observation is NOT in this prompt. It is stored in a Python REPL environment.
You MUST write Python code to explore and search the observation before deciding on an action.

## Environment Variables

The `context` dict contains:
- context['axtree']: Accessibility tree (elements with bids like [1], [2], etc.)
- context['html']: Pruned HTML of the page
- context['url']: Current page URL
- context['tabs']: Open browser tabs info
- context['error']: Error from last action (if any)
- context['goal']: The task you need to accomplish
- context['action_history']: List of previous actions taken

Current context sizes: {context_summary}

## Available Functions

- print(...): Output values to see them
- re.findall(pattern, text): Find all regex matches
- re.search(pattern, text): Search for pattern
- recursive_llm(query, sub_context): Ask a sub-LLM to analyze a portion of context
- len(), str(), list(), dict(), etc.: Standard Python builtins

## How to Explore

Write Python code to search the context. Examples:

```python
# See first 500 chars of AXTree
print(context['axtree'][:500])
```

```python
# Find all elements with "button" in them
buttons = re.findall(r'\\[\\d+\\].*button.*', context['axtree'], re.IGNORECASE)
print(buttons[:10])
```

```python
# Search for specific text
if 'Submit' in context['axtree']:
    matches = re.findall(r'\\[(\\d+)\\].*Submit.*', context['axtree'])
    print(f"Found Submit at bids: {{matches}}")
```

```python
# Use recursive_llm for semantic analysis of a chunk
chunk = context['axtree'][:2000]
answer = recursive_llm("Which element should I click to submit the form?", chunk)
print(answer)
```

## Action Format

{action_set_description}

When you have found the right element and are ready to act, use FINAL() with the action wrapped in <action> tags:

```
FINAL("<action>click('bid123')</action>")
```

Or store the action in a variable and use FINAL_VAR():
```python
my_action = "<action>click('bid123')</action>"
```
Then: FINAL_VAR(my_action)

## Critical Rules

1. Do NOT guess or make up element bids. You MUST search the context first.
2. Do NOT output FINAL() until you have found concrete evidence in the context.
3. The bid (element identifier) comes from the AXTree, shown as [bid] like [1], [42], etc.
4. Always verify the element exists before using its bid in an action.

Depth: {depth}"""

    return prompt


def build_user_prompt(goal: str, iteration: int = 0) -> str:
    """
    Build user prompt for each RLM iteration.

    Args:
        goal: The task goal
        iteration: Current iteration number

    Returns:
        User prompt string
    """
    if iteration == 0:
        return f"""Task: {goal}

This is your first interaction. You have NOT seen the context yet.
Start by exploring the context to understand the page structure.
Write Python code to examine context['axtree'] and context['goal'].

DO NOT provide a FINAL() answer yet - first explore the page."""
    else:
        return f"""Continue working on the task: {goal}

Based on what you learned from previous code execution, either:
1. Write more code to explore further
2. Use FINAL("<action>...</action>") if you found the right element

Remember: Only use FINAL() when you have found concrete evidence of the right element."""


def build_iteration_context(
    code_executed: str,
    execution_result: str,
) -> str:
    """
    Build the context message showing code execution results.

    Args:
        code_executed: The code that was executed
        execution_result: The output from execution

    Returns:
        Formatted message for the conversation
    """
    return f"""Code executed:
```python
{code_executed}
```

Output:
{execution_result}"""
