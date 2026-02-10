"""
Example: Using obs_utils with RDD Planner

This script demonstrates how to use the observation utilities
to extract context and feed it to the RDD planner.
"""

import json
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import gymnasium as gym
from dotenv import find_dotenv, load_dotenv

from obs_utils import (
    extract_goal_text,
    extract_axtree_flat,
    extract_html,
    extract_rdd_context,
    format_state_description,
    export_context_json,
)


def _lazy_register_browsergym_task(task_name: str) -> None:
    """Register BrowserGym tasks via lazy imports."""
    if task_name.startswith("miniwob"):
        import browsergym.miniwob  # noqa: F401
    elif task_name.startswith("workarena"):
        import browsergym.workarena  # noqa: F401
    elif task_name.startswith("webarena"):
        import browsergym.webarena  # noqa: F401
        import browsergym.webarenalite  # noqa: F401
        try:
            import browsergym.webarena_verified  # noqa: F401
        except ImportError:
            pass
    elif task_name.startswith("visualwebarena"):
        import browsergym.visualwebarena  # noqa: F401
    elif task_name.startswith("assistantbench"):
        import browsergym.assistantbench  # noqa: F401
    elif task_name.startswith("weblinx"):
        import weblinx_browsergym  # noqa: F401


def main():
    """Demonstrate observation utilities usage."""
    
    # Example task - change this to test different tasks
    task_name = "workarena.servicenow.sort-change-request-list"
    
    print(f"=" * 80)
    print(f"RDD Planner Observation Utilities Demo")
    print(f"=" * 80)
    print(f"\nTask: {task_name}\n")
    
    # Register and create environment
    _lazy_register_browsergym_task(task_name)
    env = gym.make(
        f"browsergym/{task_name}",
        disable_env_checker=True,
        max_episode_steps=1,
        headless=True,
        use_raw_page_output=False,
    )
    
    try:
        # Reset environment to get initial observation
        obs, _info = env.reset(seed=0)
        
        print("=" * 80)
        print("EXTRACTED CONTEXT")
        print("=" * 80)
        
        # Example 1: Extract goal text only
        print("\n1. Goal Text (extract_goal_text):")
        print("-" * 80)
        goal_text = extract_goal_text(obs)
        if goal_text:
            print(f"✓ Goal: {goal_text}")
        else:
            print("✗ No goal text found")
        
        # Example 2: Extract accessibility tree only
        print("\n2. Accessibility Tree (extract_axtree_flat):")
        print("-" * 80)
        axtree_flat = extract_axtree_flat(obs)
        if axtree_flat:
            # Show first 500 characters
            preview = axtree_flat[:500]
            print(f"✓ AXTree (first 500 chars):\n{preview}")
            if len(axtree_flat) > 500:
                print(f"... ({len(axtree_flat) - 500} more characters)")
        else:
            print("✗ No accessibility tree found")
        
        # Example 3: Extract HTML content
        print("\n3. HTML Content (extract_html):")
        print("-" * 80)
        html = extract_html(obs)
        if html:
            # Show first 500 characters
            preview = html[:500]
            print(f"✓ HTML (first 500 chars):\n{preview}")
            if len(html) > 500:
                print(f"... ({len(html) - 500} more characters)")
        else:
            print("✗ No HTML found")
        
        # Example 4: Format state description
        print("\n4. State Description (format_state_description):")
        print("-" * 80)
        state_desc = format_state_description(obs)
        print(f"✓ State: {state_desc}")
        
        # Example 5: Extract complete RDD context
        print("\n5. Complete RDD Context (extract_rdd_context):")
        print("-" * 80)
        rdd_context = extract_rdd_context(obs)
        print(f"✓ Has goal: {rdd_context['has_goal']}")
        print(f"✓ Has AXTree: {rdd_context['has_axtree']}")
        print(f"✓ Has HTML: {rdd_context['has_html']}")
        print(f"✓ Page URL: {rdd_context['page_state']['url']}")
        print(f"✓ Page Title: {rdd_context['page_state']['page_title']}")
        
        # Example 6: Export as JSON
        print("\n6. JSON Export (export_context_json):")
        print("-" * 80)
        # Export without the full axtree and html for readability
        context_for_export = {
            "goal_text": rdd_context["goal_text"],
            "has_axtree": rdd_context["has_axtree"],
            "axtree_length": len(rdd_context["axtree_flat"]) if rdd_context["axtree_flat"] else 0,
            "has_html": rdd_context["has_html"],
            "html_length": len(rdd_context["html"]) if rdd_context["html"] else 0,
            "page_state": rdd_context["page_state"],
        }
        print(json.dumps(context_for_export, indent=2, default=str))
        
        # Example 7: How to use with RDD Planner
        print("\n" + "=" * 80)
        print("USAGE WITH RDD PLANNER")
        print("=" * 80)
        print("\nExample code:")
        print("-" * 80)
        print("""
from rdd_planner.obs_utils import extract_goal_text, extract_html, format_state_description
from rdd_planner.rdd_planner import SimplifiedRDDPlanner, RDDConfig

# Get observation from environment
obs, info = env.reset()

# Extract context for planning
goal_text = extract_goal_text(obs)
html = extract_html(obs)
state_desc = format_state_description(obs)

# Create planner and generate plan
config = RDDConfig(max_depth=2, max_nodes=20)
planner = SimplifiedRDDPlanner(llm, config)
result = planner.plan(task=goal_text, state=state_desc, html=html)

print(result["plan"])
        """)
        
        print("\n" + "=" * 80)
        print("DEMO COMPLETE")
        print("=" * 80)
        
    finally:
        env.close()


if __name__ == "__main__":
    main()
