import logging


def planner_tool(goal: str, context: str) -> str:
    """
    A planning tool that simulates consulting multiple sub-agents to generate a strategy.

    Args:
        goal (str): The user's high-level goal.
        context (str): A summary of the current situation or obstacle.

    Returns:
        str: A reasoned advice or updated plan snippet from the "expert agents".
    """
    # In a real implementation, this could call other LLMs, agents, or APIs.
    # For the hackathon, we simulate a helpful "Expert Panel".

    logging.info(f"Planner Tool invoked with goal: {goal}")

    return f"""
[Expert Panel Advice]
Based on the goal '{goal}' and context '{context}', we recommend:
1. Verify if the element is inside an iframe.
2. If previous clicks failed, try using coordinate-based clicks or JavaScript execution.
3. Don't forget to scroll into view if the element is hidden.
"""


# Dictionary mapping tool names to functions for easy registration
HACKATHON_TOOLS = {"planner_tool": planner_tool}
