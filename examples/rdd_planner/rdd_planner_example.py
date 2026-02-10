#!/usr/bin/env python3
"""
RDD Planner Example with LLM Agent

Demonstrates recursive task decomposition using an LLM agent.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

from dotenv import find_dotenv, load_dotenv  # type: ignore[import-untyped]

sys.path.insert(0, str(Path(__file__).parent))

from rdd_planner import SimplifiedRDDPlanner, RDDConfig
from agentlab.llm.chat_api import OpenRouterModelArgs

load_dotenv(find_dotenv())

# Configuration
MODEL_NAME = "openai/gpt-4o-mini"


def _save_result(result: dict, example_name: str) -> None:
    """Save planner result as JSON in a per-run file."""
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = results_dir / f"{example_name}_{timestamp}.json"

    with file_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"Saved planner result to {file_path}")


def run_workarena_example():
    """Execute WorkArena incident creation task decomposition."""
    print("RDD Planner - WorkArena Task Example")
    
    llm_args = OpenRouterModelArgs(model_name=MODEL_NAME)
    llm = llm_args.make_model()
    
    config = RDDConfig(
        max_depth=2,      # Decompose up to 2 levels deep
        max_nodes=20,     # Limit total number of tasks
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

    _save_result(result, "workarena")
    return result


def run_simple_example():
    """Execute simple e-commerce task decomposition."""
    print("RDD PLANNER - Simple E-commerce Example")
    
    llm_args = OpenRouterModelArgs(model_name=MODEL_NAME)
    llm = llm_args.make_model()
    
    config = RDDConfig(
        max_depth=2,      # Decompose up to 2 levels deep
        max_nodes=20,     # Limit total number of tasks
    )
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

    _save_result(result, "simple_ecommerce")
    return result


def run_cooking_example():
    """Execute cooking task decomposition example."""
    print("RDD PLANNER - Cooking Example")
    
    llm_args = OpenRouterModelArgs(model_name=MODEL_NAME)
    llm = llm_args.make_model()
    
    config = RDDConfig(
        max_depth=2,      # Decompose up to 2 levels deep
        max_nodes=20,     # Limit total number of tasks
    )
    planner = SimplifiedRDDPlanner(llm, config)
    
    task = "Prepare a neapolitan pizza for 10 persons"
    state = "In a kitchen with all basic ingredients and equipment available"
    
    print(f"Task: {task}")
    print(f"State: {state}")
    
    result = planner.plan(task, state)
    
    print("\nGenerated Plan:\n")
    print(result["plan"])
    print("\n" + "="*80)
    print(f"Total tasks: {len(result['graph']['nodes'])}")
    print("="*80 + "\n")

    _save_result(result, "cooking")
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

    def _should_run(prompt: str, *, default_run: bool = True) -> bool:
        """
        Return True if user wants to run the example.

        - Enter: run (by default)
        - Esc or 's'/'skip': skip
        - Non-interactive stdin (EOF): follow default_run
        """
        try:
            ans = input(prompt)
        except EOFError:
            # e.g. when stdin isn't interactive (piped/CI)
            return default_run

        if ans == "\x1b":
            return False
        if ans.strip().lower() in {"s", "skip", "n", "no"}:
            return False
        # Empty input => default action (run by default)
        return default_run

    # Run WorkArena example
    if _should_run("Press Enter to run WorkArena example (Esc or 's' to skip): "):
        run_workarena_example()
    
    # Run simple example
    if _should_run("\nPress Enter to run simple e-commerce example (Esc or 's' to skip): "):
        run_simple_example()
    
    # Run cooking example
    if _should_run("\nPress Enter to run cooking example (Esc or 's' to skip): "):
        run_cooking_example()


if __name__ == "__main__":
    main()
