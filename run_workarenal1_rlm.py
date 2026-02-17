"""
Script to run GenericAgent with RLM (Recursive Language Model) on MiniWob tasks.

This demonstrates using RLM as a drop-in LLM wrapper that enhances any provider
with long-context handling through iterative REPL exploration.

Based on run_miniwob_agentq.py but using the RLM approach instead of AgentQ.
"""

import logging
import os
from pathlib import Path

import bgym
from bgym import HighLevelActionSetArgs
from dotenv import find_dotenv, load_dotenv

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
# project_root = Path(__file__).resolve().parent
# ensure_benchmark("workarena_l1", project_root=project_root)
load_dotenv(find_dotenv())

# Base model to wrap with RLM
# You can use any provider: OpenAI, Anthropic, LiteLLM, etc.
base_model_args = OpenRouterModelArgs(
    model_name="openai/gpt-5",
    max_new_tokens=None,
    temperature=None,
)

# Optional: Use a cheaper model for recursive sub-calls
recursive_model_args = OpenRouterModelArgs(
    model_name="openai/gpt-5",
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
        use_ax_tree=False,
        use_focused_element=True,
        use_error_logs=True,
        use_history=False,
        use_past_error_logs=False,
        use_action_history=True,
        use_think_history=True,
        use_diff=False,
        html_type="pruned_html",
        use_screenshot=False,
        use_som=False,
        extract_visible_tag=True,
        extract_clickable_tag=True,
        extract_coords="False",
        filter_visible_elements_only=False,
    ),
    action=ActionFlags(
        multi_actions=False,
        action_set="bid",
        long_description=False,
        individual_examples=False,
    ),
    use_plan=False,
    use_criticise=False,
    use_thinking=True,
    use_memory=False,
    use_concrete_example=True,
    use_abstract_example=True,
    use_hints=True,
    enable_chat=False,
    max_prompt_tokens=40_000,
    be_cautious=True,
    extra_instructions=None,
)

# 3. Create GenericAgent with RLM-enhanced LLM
agent_args = GenericAgentArgs(
    chat_model_args=rlm_model_args,  # RLM wraps the base model
    flags=flags,
    max_retry=2,
)

# 4. Setup Benchmark
benchmark_name = "workarena_l1"
benchmark = bgym.DEFAULT_BENCHMARKS[benchmark_name](n_repeats=10)
task_name_regex = os.getenv(
    "RLM_TASK_REGEX",
    "workarena.servicenow.order-apple-watch|workarena.servicenow.all-menu",
)

# Specifically target a simple task for testing
try:
    benchmark = benchmark.subset_from_regexp("task_name", task_name_regex)
except (AttributeError, Exception) as e:
    logger.warning(f"Could not filter benchmark: {e}. Running full benchmark if needed.")
print(benchmark)
# 5. Run Study
n_jobs = int(os.getenv("RLM_N_JOBS", "5"))
parallel_backend = os.getenv("RLM_PARALLEL_BACKEND", "ray")

if __name__ == "__main__":
    # RLM steps can be substantially slower than generic prompting.
    # Increase timeout budget to avoid premature Ray cancellations.
    study = Study(
        [agent_args],
        benchmark,
        logging_level_stdout=logging.INFO,
        logging_level=logging.INFO,
        avg_step_timeout=120,
    )

    print(f"Starting Study on {benchmark_name} with GenericAgent + RLM...")
    print(f"  Base model: {base_model_args.model_name}")
    print(f"  RLM max_iterations: {rlm_model_args.max_iterations}")
    print(f"  RLM max_depth: {rlm_model_args.max_depth}")
    print(f"  Task regex: {task_name_regex}")
    print(f"  Backend: {parallel_backend}, n_jobs: {n_jobs}")

    study.run(
        n_jobs=n_jobs,
        parallel_backend=parallel_backend,
        strict_reproducibility=False,
        n_relaunch=1,
    )
    print("Study completed.")
