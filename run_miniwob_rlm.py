"""
Script to run GenericAgent with RLM (Recursive Language Model) on MiniWob tasks.

This demonstrates using RLM as a drop-in LLM wrapper that enhances any provider
with long-context handling through iterative REPL exploration.

Based on run_miniwob_agentq.py but using the RLM approach instead of AgentQ.
"""

import logging
from pathlib import Path

import bgym
from bgym import HighLevelActionSetArgs
from dotenv import find_dotenv, load_dotenv
from numpy import True_

from agentlab.agents.dynamic_prompting import ActionFlags, ObsFlags
from agentlab.agents.generic_agent.generic_agent import GenericAgentArgs
from agentlab.agents.generic_agent.generic_agent_prompt import GenericPromptFlags
from agentlab.benchmarks.setup_benchmark import ensure_benchmark
from agentlab.experiments.study import Study
from agentlab.llm.chat_api import OpenRouterModelArgs
from agentlab.llm.rlm_chat_model import RLMModelArgs

# 1. Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 2. Configure the models
# Ensure MiniWob assets and MINIWOB_URL are set
project_root = Path(__file__).resolve().parent
ensure_benchmark("miniwob", project_root=project_root)
load_dotenv(find_dotenv())

# Base model to wrap with RLM
# You can use any provider: OpenAI, Anthropic, LiteLLM, etc.
base_model_args = OpenRouterModelArgs(
    model_name="openai/gpt-5-mini",
    max_new_tokens=None,
    temperature=None,
)

# Optional: Use a cheaper model for recursive sub-calls
recursive_model_args = OpenRouterModelArgs(
    model_name="openai/gpt-5-mini",
    max_new_tokens=None,
    temperature=None,
)

# Wrap the base model with RLM
# RLM adds iterative REPL exploration for handling long contexts
rlm_model_args = RLMModelArgs(
    inner_model_args=base_model_args,
    recursive_model_args=recursive_model_args,  # Optional: cheaper model for sub-calls
    max_depth=5,  # Maximum recursion depth
    max_iterations=30,  # Maximum REPL iterations per call
    max_output_chars=3000,  # Truncate long REPL outputs
)

# Use standard flags for MiniWob
flags = GenericPromptFlags(
    obs=ObsFlags(
        use_html=False,
        use_ax_tree=True,  # Use Accessibility Tree
        use_focused_element=True,
        use_error_logs=False,
        use_history=False,
        use_past_error_logs=False,
        use_action_history=False,
        use_think_history=False,
        use_diff=False,
        use_screenshot=False,
    ),
    use_abstract_example=False,
    use_concrete_example=False,
    enable_chat=True,
    action=ActionFlags(
        action_set=HighLevelActionSetArgs(
            subsets=["bid", "nav"],  # Enable basic interactions and navigation
            multiaction=False,
            strict=False,  # Allow imperfect actions
        )
    ),
)

# 3. Create GenericAgent with RLM-enhanced LLM
agent_args = GenericAgentArgs(
    chat_model_args=rlm_model_args,  # RLM wraps the base model
    flags=flags,
    max_retry=2,
)

# 4. Setup Benchmark
benchmark_name = "miniwob"
benchmark = bgym.DEFAULT_BENCHMARKS[benchmark_name](n_repeats=1)

# Specifically target a simple task for testing
try:
    benchmark = benchmark.subset_from_regexp("task_name", "miniwob.click")
except (AttributeError, Exception) as e:
    logger.warning(f"Could not filter benchmark: {e}. Running full benchmark if needed.")
print(benchmark)
# 5. Run Study
n_jobs = 4  # Sequential execution for debugging
parallel_backend = "ray"

if __name__ == "__main__":
    study = Study([agent_args], benchmark, logging_level_stdout=logging.INFO, logging_level=logging.INFO)

    print(f"Starting Study on {benchmark_name} with GenericAgent + RLM...")
    print(f"  Base model: {base_model_args.model_name}")
    print(f"  RLM max_iterations: {rlm_model_args.max_iterations}")
    print(f"  RLM max_depth: {rlm_model_args.max_depth}")

    study.run(
        n_jobs=n_jobs,
        parallel_backend=parallel_backend,
        strict_reproducibility=False,
        n_relaunch=1,
    )
    print("Study completed.")
