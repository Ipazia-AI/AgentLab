"""
Script to run AgentQ on MiniWob tasks for benchmarking and debugging.
Based on main_workarena_debug.py but specialized for AgentQ.
"""

import logging
from pathlib import Path

import bgym
from bgym import HighLevelActionSetArgs
from dotenv import find_dotenv, load_dotenv

from agentlab.agents.agentq.agentq import AgentQArgs
from agentlab.agents.dynamic_prompting import ActionFlags, ObsFlags
from agentlab.agents.generic_agent.generic_agent_prompt import GenericPromptFlags
from agentlab.benchmarks.setup_benchmark import ensure_benchmark
from agentlab.experiments.study import Study
from agentlab.llm.chat_api import OpenRouterModelArgs

# 1. Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 2. Configure AgentQ
# Ensure MiniWob assets and MINIWOB_URL are set
project_root = Path(__file__).resolve().parent
ensure_benchmark("miniwob", project_root=project_root)
load_dotenv(find_dotenv())

gpt_oss_args = OpenRouterModelArgs(
    model_name="google/gemini-3-flash-preview",
)
chat_args = OpenRouterModelArgs(
    model_name="openai/gpt-5",
)

# chat_args = gpt_oss_args

# Use standard flags for MiniWob
flags = GenericPromptFlags(
    obs=ObsFlags(
        use_html=False,
        use_ax_tree=True,  # Use Accessibility Tree
        use_focused_element=True,
        use_error_logs=True,
        use_history=True,
        use_past_error_logs=True,
        use_action_history=True,
        use_think_history=True,
        use_diff=False,
        use_screenshot=True,
    ),
    use_abstract_example=True,
    use_concrete_example=True,
    enable_chat=False,
    action=ActionFlags(
        action_set=HighLevelActionSetArgs(
            subsets=["bid", "nav"],  # Enable basic interactions and navigation
            multiaction=False,
            strict=False,  # Allow imperfect actions
        )
    ),
)

agent_args = AgentQArgs(
    chat_model_args=chat_args,
    flags=flags,
    mcts_budget=3,
    mcts_max_workers=1,  # Sequential for better debugging
    mcts_rollout_depth=2,
    critic_type="tournament",  # "tournament" | "absolute"
    selection_strategy="max_visit",  # "max_visit" | "ahead_k"
    # ahead_k=1,  # Only used if selection_strategy="ahead_k"
    use_real_rollouts=False,  # False=fast_reward (paper default), True=real browser rollouts
    sync_mcts=False,
    iteration_timeout=None,
    # Debug/logging toggles
    mcts_debug_logging=False,  # Enables DEBUG timing/heartbeat logs
    browser_fork_logging=False,  # Enables browser_forking INFO logs
)

# 3. Setup Benchmark
benchmark_name = "workarena_l1"
benchmark = bgym.DEFAULT_BENCHMARKS[benchmark_name](n_repeats=1)

# Specifically target the first task of workarena_l1
try:
    benchmark.env_args_list = benchmark.env_args_list[:1]
    # benchmark = benchmark.subset_from_glob(
    #     "task_name", "workarena.servicenow.sort-change-request-list"
    # )
except (AttributeError, Exception) as e:
    logger.warning(f"Could not filter benchmark: {e}. Running full benchmark if needed.")

# 4. Run Study
n_jobs = 1  # 1 job for sequential execution
parallel_backend = "sequential"

if __name__ == "__main__":
    study = Study([agent_args], benchmark, logging_level_stdout=logging.INFO)

    print(f"Starting Study on {benchmark_name} with AgentQ...")
    study.run(
        n_jobs=n_jobs, parallel_backend=parallel_backend, strict_reproducibility=False, n_relaunch=1
    )
    print("Study completed.")
