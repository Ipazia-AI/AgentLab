# Dynamic Prompting Reuse Guide

This guide explains how to leverage the existing `dynamic_prompting` infrastructure in AgentLab to build custom agents efficiently.

## Core Components

The dynamic prompting system (`src/agentlab/agents/dynamic_prompting.py`) is designed to handle the constraints of browser automation: large contexts (HTML/AXTree), fixed token limits, and variable history.

### 1. `MainPrompt`
Located in `src/agentlab/agents/generic_agent/generic_agent_prompt.py`, this class acts as the conductor. It assembles:
- **Instructions**: The goal and persona.
- **Observation**: The current state of the page (HTML, AXTree, Screenshot).
- **History**: Past actions, thoughts, and errors.
- **Action Space**: Description of available browser actions.
- **Reasoning**: Plan, Think, Criticise blocks.

**Reuse Strategy**:
Instead of building your own prompt string, instantiate `MainPrompt` inside your agent's `get_action` loop. It automatically handles flag-based visibility (e.g., `use_html=False`) and shrinking.

### 2. `fit_tokens`
This function (`dp.fit_tokens`) is the engine that ensures your prompt fits within the LLM's context window. It iteratively calls `shrink()` on the `MainPrompt` (and its children) until the token count is safe.

**Usage**:
```python
human_prompt = dp.fit_tokens(
    shrinkable=main_prompt,
    max_prompt_tokens=max_prompt_tokens,
    model_name=model_name,
    additional_prompts=[system_prompt]
)
```

### 3. `PromptElement` & `Shrinkable`
These are the building blocks. You can subclass them to create custom sections (e.g., a specific "Hint" block or "Tool Output" block).

- `PromptElement`: Basic block with `_prompt`, `_abstract_ex` (for few-shot), and `_parse_answer`.
- `Shrinkable`: Adds a `shrink()` method for reducing size (e.g., truncating logs).

## Example: Customizing `MainPrompt`

If you need a slightly different prompt structure, you can subclass `MainPrompt`:

```python
class MyCustomPrompt(MainPrompt):
    def __init__(self, ...):
        super().__init__(...)
        # Add your custom element
        self.my_tool_output = MyToolOutput(visible=True)

    @property
    def _prompt(self):
        # Override composition order
        return f"""
{self.instructions.prompt}
{self.my_tool_output.prompt}  <-- Custom element injected
{self.obs.prompt}
...
"""
```

## Flags System

The `GenericPromptFlags` (`src/agentlab/agents/generic_agent/generic_agent_prompt.py`) control what gets included.
- `obs.use_html`: Include DOM.
- `obs.use_ax_tree`: Include Accessibility Tree.
- `obs.use_screenshot`: Include visual screenshot.
- `use_plan`: Enable the `<plan>` block.
- `use_thinking`: Enable the `<think>` block.

By reusing `GenericAgentArgs` and passing these flags, you get all this toggling logic for free.
