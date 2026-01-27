#!/usr/bin/env python3
"""
RDD Planner Example with LLM Agent

Demonstrates recursive task decomposition using an LLM agent.
"""

import sys
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

sys.path.insert(0, str(Path(__file__).parent))

from rdd_planner import SimplifiedRDDPlanner, RDDConfig
from agentlab.llm.chat_api import OpenRouterModelArgs

load_dotenv(find_dotenv())

# Configuration
MODEL_NAME = "openai/gpt-4o-mini"


def run_workarena_example():
    """Execute WorkArena incident creation task decomposition."""
    print("RDD Planner - WorkArena Task Example")
    
    llm_args = OpenRouterModelArgs(model_name=MODEL_NAME)
    llm = llm_args.make_model()
    
    config = RDDConfig(
        max_depth=2,      # Decompose up to 2 levels deep
        max_width=3,      # Max 3 subtasks per task
        max_nodes=10      # Limit total number of tasks
    )
    planner = SimplifiedRDDPlanner(llm, config)
    
    # Define the task
    task = "Create a new incident in ServiceNow with priority High and category Network"
    state = "On ServiceNow homepage, logged in as admin"
    
    print(f"Task: {task}")
    print(f"State: {state}\n")
    print("="*80)
    print("Starting decomposition...")
    print("="*80)
    
    # Generate the plan
    result = planner.plan(task, state)
    
    print("\nGenerated Plan:\n")
    print(result["plan"])
    print("\n" + "="*80)
    print(f"Total tasks: {len(result['graph']['nodes'])}")
    print("="*80 + "\n")
    
    return result


def run_simple_example():
    """Execute simple e-commerce task decomposition."""
    print("RDD PLANNER - Simple E-commerce Example")
    
    llm_args = OpenRouterModelArgs(model_name=MODEL_NAME)
    llm = llm_args.make_model()
    
    config = RDDConfig(max_depth=2, max_width=3, max_nodes=8)
    planner = SimplifiedRDDPlanner(llm, config)
    
    task = "Add a blue t-shirt size M to cart on amazon.com"
    state = "On amazon.com homepage, logged in"
    
    print(f"Task: {task}")
    print(f"State: {state}")
    
    result = planner.plan(task, state)
    
    print("\nGenerated Plan:\n")
    print(result["plan"])
    print("\n" + "="*80)
    print(f"Total tasks: {len(result['graph']['nodes'])}")
    print("="*80 + "\n")
    
    return result


def run_cooking_example():
    """Execute cooking task decomposition example."""
    print("RDD PLANNER - Cooking Example")
    
    llm_args = OpenRouterModelArgs(model_name=MODEL_NAME)
    llm = llm_args.make_model()
    
    config = RDDConfig(max_depth=2, max_width=4, max_nodes=12)
    planner = SimplifiedRDDPlanner(llm, config)
    
    task = "Prepare a good Pakistani biryani for 4 persons"
    state = "In a kitchen with all basic ingredients and equipment available"
    
    print(f"Task: {task}")
    print(f"State: {state}")
    
    result = planner.plan(task, state)
    
    print("\nGenerated Plan:\n")
    print(result["plan"])
    print("\n" + "="*80)
    print(f"Total tasks: {len(result['graph']['nodes'])}")
    print("="*80 + "\n")
    
    return result


def main():
    """Run selected examples."""
    
    # print("\nRDD Planner - Examples with LLM Agent")
    # print("\nThis will demonstrate:")
    # print("  1. How BFS decomposes tasks level-by-level")
    # print("  2. How DFS solves dependencies bottom-up")
    # print("  3. How the planner generates hierarchical plans")
    # print("  4. LLM Agent responses")
    # print("="*80 + "\n")

    # Run WorkArena example
    ans = input("Press Enter or Esc to skip WorkArena example, or type anything else to run it: ")
    if ans.strip() and ans != "\x1b":
        run_workarena_example()
    
    # Run simple example
    ans = input("\nPress Enter or Esc to skip simple e-commerce example, or type anything else to run it: ")
    if ans.strip() and ans != "\x1b":
        run_simple_example()
    
    # Run cooking example
    ans = input("\nPress Enter or Esc to skip cooking example, or type anything else to run it: ")
    if ans.strip() and ans != "\x1b":
        run_cooking_example()


if __name__ == "__main__":
    main()
