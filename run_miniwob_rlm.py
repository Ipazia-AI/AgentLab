"""
Script to run RLMGenericAgent on MiniWob tasks for benchmarking and debugging.
Based on run_miniwob_agentq.py but specialized for RLMGenericAgent.

The RLM agent externalizes browser observations (AXTree, HTML) to a sandboxed
REPL environment, allowing the LLM to programmatically explore large contexts
instead of stuffing them into the prompt. Screenshots are kept in the prompt
for vision models.
"""

import logging
from pathlib import Path

import bgym
from bgym import HighLevelActionSetArgs
from dotenv import find_dotenv, load_dotenv

from agentlab.agents.rlm_agent import RLMGenericAgentArgs, RLMPromptFlags
from agentlab.agents.dynamic_prompting import ActionFlags, ObsFlags
from agentlab.benchmarks.setup_benchmark import ensure_benchmark
from agentlab.experiments.study import Study
from agentlab.llm.chat_api import OpenRouterModelArgs

# 1. Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 2. Configure RLMGenericAgent
# Ensure MiniWob assets and MINIWOB_URL are set
project_root = Path(__file__).resolve().parent
ensure_benchmark("miniwob", project_root=project_root)
load_dotenv(find_dotenv())

# Primary model for main RLM loop (vision-capable for screenshots)
main_model_args = OpenRouterModelArgs(
    model_name="google/gemini-2.0-flash-001",
    vision_support=True,  # Enable vision for screenshot support
)

# Optional: cheaper/faster model for recursive_llm calls (sub-queries)
recursive_model_args = OpenRouterModelArgs(
    model_name="google/gemini-2.0-flash-001",
    vision_support=False,  # Recursive calls are text-only
)

# Use RLM-specific flags
# RLM externalizes DOM/AXTree to REPL, but keeps screenshots in prompt
flags = RLMPromptFlags(
    obs=ObsFlags(
        use_html=True,  # Include HTML in REPL context
        use_ax_tree=True,  # Include AXTree in REPL context
        use_focused_element=True,
        use_error_logs=True,
        use_history=False,  # RLM handles history in REPL context
        use_past_error_logs=True,
        use_action_history=False,  # RLM handles this in REPL context
        use_think_history=False,
        use_diff=False,
        # Screenshot settings - kept in prompt for vision models
        use_screenshot=True,  # Enable screenshot in prompt
        use_som=False,  # Set to True for annotated screenshots with bids
        openai_vision_detail="auto",  # "low", "high", or "auto"
    ),
    action=ActionFlags(
        action_set=HighLevelActionSetArgs(
            subsets=["bid", "nav"],  # Enable basic interactions and navigation
            multiaction=False,
            strict=False,  # Allow imperfect actions
        )
    ),
)

agent_args = RLMGenericAgentArgs(
    chat_model_args=main_model_args,
    recursive_model_args=recursive_model_args,  # Set to None to use main model
    flags=flags,
    max_retry=4,
    max_iterations=10,  # Max REPL iterations per step
    max_depth=2,  # Max recursion depth for recursive_llm calls
)

# 3. Setup Benchmark
benchmark_name = "miniwob_tiny_test"
benchmark = bgym.DEFAULT_BENCHMARKS[benchmark_name](n_repeats=1)

# Specifically target a simple task to start
try:
    benchmark = benchmark.subset_from_glob("task_name", "miniwob.click-dialog")
except (AttributeError, Exception) as e:
    logger.warning(f"Could not filter benchmark: {e}. Running full benchmark if needed.")

# 4. Run Study
n_jobs = 1  # 1 job for sequential execution
parallel_backend = "sequential"

if __name__ == "__main__":
    study = Study([agent_args], benchmark, logging_level_stdout=logging.INFO)

    print(f"Starting Study on {benchmark_name} with RLMGenericAgent...")
    print(f"Main model: {main_model_args.model_name}")
    print(f"Recursive model: {recursive_model_args.model_name if recursive_model_args else 'same as main'}")
    print(f"Max iterations: {agent_args.max_iterations}")
    print(f"Max depth: {agent_args.max_depth}")
    
    study.run(
        n_jobs=n_jobs, parallel_backend=parallel_backend, strict_reproducibility=False, n_relaunch=1
    )
    print("Study completed.")
