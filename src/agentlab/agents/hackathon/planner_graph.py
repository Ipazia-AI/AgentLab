import logging
from typing import TypedDict, Annotated, List, Union
import operator

try:
    from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
    from langchain_openai import ChatOpenAI
    from langgraph.graph import StateGraph, END

    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False


class PlannerState(TypedDict):
    """
    The state of the planner graph.
    """

    # Messages in the conversation
    messages: Annotated[List[BaseMessage], operator.add]
    # The high-level goal
    goal: str
    # The current observation (text representation)
    observation: str
    # The output plan
    plan: str


def create_planner_graph(
    model_name: str = "gpt-4-turbo",
    temperature: float = 0,
    base_url: str = None,
    api_key: str = None,
):
    """
    Creates and returns a compiled LangGraph runnable for planning.
    """
    if not LANGGRAPH_AVAILABLE:
        logging.warning("LangGraph not installed. Planner graph cannot be created.")
        return None

    llm = ChatOpenAI(model=model_name, temperature=temperature, base_url=base_url, api_key=api_key)

    def reasoning_node(state: PlannerState):
        """
        Analyzes the current situation and decides if the plan needs updating.
        """
        prompt = f"""
You are a strategic planner for a browser automation agent.
Goal: {state['goal']}

Current Observation:
{state['observation']}

Current Plan:
{state.get('plan', 'No plan yet.')}

Analyze the situation. Has the previous step succeeded? Is the current plan still valid?
Provide a brief reasoning.
"""
        response = llm.invoke([HumanMessage(content=prompt)])
        return {"messages": [response]}

    def planning_node(state: PlannerState):
        """
        Updates the plan based on the reasoning.
        """
        # In a real implementation, we might pass the reasoning output here.
        # For now, we just ask for a plan update based on the last message (reasoning).

        prompt = f"""
Based on the previous analysis, update the plan.
Output ONLY the plan as a numbered list.
If the current plan is still valid, output it as is.
"""
        response = llm.invoke(state["messages"] + [HumanMessage(content=prompt)])
        return {"plan": response.content}

    # Define the graph
    workflow = StateGraph(PlannerState)

    workflow.add_node("reasoning", reasoning_node)
    workflow.add_node("planning", planning_node)

    workflow.set_entry_point("reasoning")
    workflow.add_edge("reasoning", "planning")
    workflow.add_edge("planning", END)

    return workflow.compile()
