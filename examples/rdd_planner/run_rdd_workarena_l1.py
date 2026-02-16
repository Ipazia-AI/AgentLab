"""
Simple RDD Planner Integration with MiniWoB
"""

import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))

import bgym

from agentlab.agents.generic_agent.tmlr_config import BASE_FLAGS
from agentlab.agents.rdd_agent import RDDAgentArgs
from agentlab.experiments.study import make_study
from agentlab.llm.chat_api import OpenRouterModelArgs

load_dotenv()

# ============================================================================
# CONFIGURATION - Edit these variables
# ============================================================================
TASK_INDEX = 0          # Which task to run (0 = first task)
USE_PLAN = True        # Enable RDD planning
USE_AXTREE = True      # Use AXTree for planning
HEADLESS = True        # Run browser in headless mode (True = no GUI)
USE_PLAN_REFINER = True # Enable plan refinement (default: True)
REFINER_ITERATIONS = 2  # Number of refinement iterations (default: 2)
# ============================================================================

if __name__ == "__main__":
    print("RDD Planner + WorkArena Integration\n")
    
    # Configure model
    model_args = OpenRouterModelArgs(
        model_name="openai/gpt-5-mini",
        max_total_tokens=128000
    )
    
    # Configure agent with planning enabled
    from copy import deepcopy
    flags = deepcopy(BASE_FLAGS)
    flags.use_plan = USE_PLAN
    
    agent_args = RDDAgentArgs(
        chat_model_args=model_args,
        flags=flags,
        use_axtree=USE_AXTREE,
        use_plan_refiner=USE_PLAN_REFINER,
        refiner_iterations=REFINER_ITERATIONS
    )
    
    benchmark_name = "workarena_l1"
    benchmark = bgym.DEFAULT_BENCHMARKS[benchmark_name](n_repeats=10)

# Specifically target a simple task for testing
    benchmark = benchmark.subset_from_regexp("task_name", "workarena.servicenow.create-problem")

    # Create study
    study = make_study(
        agent_args=[agent_args],
        benchmark=benchmark
    )
    
    # Limit to selected task by index
    print(f"Total tasks available: {len(study.exp_args_list)}")
    
    # if TASK_INDEX >= len(study.exp_args_list):
    #     print(f"❌ Error: Task index {TASK_INDEX} out of range (0-{len(study.exp_args_list)-1})")
    #     sys.exit(1)
    
    # study.exp_args_list = [study.exp_args_list[TASK_INDEX]]
    # print(f"Running task at index {TASK_INDEX}\n")
    
    # NOW set task for the experiments that will actually run
    for i, exp_args in enumerate(study.exp_args_list, 1):
        exp_args.env_args.headless = HEADLESS
        exp_args.agent_args.task = exp_args.env_args.task_name
        print(f"  Task {i}: {exp_args.env_args.task_name}")
    print("\n▶️  Running experiments...\n")
    study.run(n_jobs=1, parallel_backend="sequential")
    
    # Results
    results, summary, errors = study.get_results()
    
    # Show available columns for debugging
    print(f"\nAvailable summary columns: {list(summary.columns)}")
    
    # Try to get success rate from available columns
    if 'cum_reward' in summary.columns:
        success_rate = summary['cum_reward'].mean()
        print(f"✅ Success rate: {success_rate:.1%}")
    elif 'reward' in summary.columns:
        success_rate = summary['reward'].mean()
        print(f"✅ Success rate: {success_rate:.1%}")
    elif 'avg_reward' in summary.columns:
        success_rate = summary['avg_reward'].mean()
        print(f"✅ Average reward: {success_rate:.2f}")
    else:
        print("⚠ No reward column found in results")
    
    print(f"📁 Results: {study.dir}")
    print(f"📁 Results: {study.dir}")
