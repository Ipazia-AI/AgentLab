"""
System prompt templates for RLM.

This module provides prompt building functions for the RLM approach.
The prompts instruct the model to explore context through a Python REPL
environment rather than having the context directly in the prompt.

Based on the RLM paper (arXiv:2512.24601).
"""


def build_system_prompt(context_size: int, depth: int = 0) -> str:
    """
    Build system prompt for RLM.

    Args:
        context_size: Size of context in characters
        depth: Current recursion depth (0 = root)

    Returns:
        System prompt string
    """
    prompt = f"""You are a Recursive Language Model. You interact with context through a Python REPL environment.

The context is stored in variable `context` (not in this prompt). Size: {context_size:,} characters.
IMPORTANT: You cannot see the context directly. You MUST write Python code to search and explore it.

Available in environment:
- context: str (the document to analyze)
- query: str (the question/task)
- recursive_llm(sub_query, sub_context) -> str (recursively process sub-context with another LLM call)
- re: already imported regex module (use re.findall, re.search, etc.)

Write Python code to answer the query. The last expression or print() output will be shown to you.

Examples:
- print(context[:500])  # See first 500 chars
- matches = re.findall(r'keyword.*', context); print(matches[:5])
- idx = context.find('search term'); print(context[idx:idx+200])

For large contexts, use recursive_llm to process chunks:
```python
chunks = [context[i:i+5000] for i in range(0, len(context), 5000)]
results = []
for chunk in chunks:
    answer = recursive_llm("Extract key information from this chunk", chunk)
    results.append(answer)
print(results)
```

CRITICAL: Do NOT guess or make up answers. You MUST search the context first to find the actual information.
Only use FINAL("answer") after you have found concrete evidence in the context.

When ready to answer, use one of:
- FINAL("your answer here") - provide the answer directly
- FINAL_VAR(variable_name) - return a variable from the REPL environment

Depth: {depth}"""

    return prompt


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
