"""
Script to compare GenericAgent with and without RLM on WorkArena L2 tasks,
using Claude Opus 4.6 via OpenRouter.

Runs two studies sequentially:
  1. GenericAgent + RLM (Recursive Language Model) wrapping Opus 4.6
  2. GenericAgent (plain) with Opus 4.6 directly

This allows a direct comparison of the RLM approach vs. standard prompting
on the same L2 task with the same underlying model.
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
from agentlab.experiments.study import Study
from agentlab.llm.chat_api import OpenRouterModelArgs
from agentlab.llm.rlm_chat_model import RLMModelArgs

# ---------------------------------------------------------------------------
# 1. Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 2. Environment
# ---------------------------------------------------------------------------
load_dotenv(find_dotenv())

MODEL_NAME = "openai/gpt-5-mini"
# MODEL_NAME = "openai/gpt-5.2"

# ---------------------------------------------------------------------------
# 3. Model configuration  (Claude Opus 4.6 via OpenRouter)
# ---------------------------------------------------------------------------
opus_model_args = OpenRouterModelArgs(
    model_name=MODEL_NAME,
    max_total_tokens=200_000,
    max_input_tokens=200_000,
    max_new_tokens=16_384,
    temperature=None,
    vision_support=True,
)

# For RLM recursive sub-calls you can use a cheaper model to save cost.
# Set to the same model for now; swap to e.g. "anthropic/claude-sonnet-4" if desired.
recursive_model_args = OpenRouterModelArgs(
    model_name=MODEL_NAME,
    max_total_tokens=200_000,
    max_input_tokens=200_000,
    max_new_tokens=16_384,
    temperature=None,
    vision_support=True,
)

# ---------------------------------------------------------------------------
# 4. RLM-wrapped model
# ---------------------------------------------------------------------------
rlm_model_args = RLMModelArgs(
    inner_model_args=opus_model_args,
    recursive_model_args=recursive_model_args,
    max_depth=5,
    max_iterations=30,
    max_output_chars=3000,
)

# ---------------------------------------------------------------------------
# 5. Shared prompt / observation flags  (same for both agents)
# ---------------------------------------------------------------------------
flags = GenericPromptFlags(
    obs=ObsFlags(
        use_html=True,
        use_ax_tree=True,
        use_focused_element=True,
        use_error_logs=False,
        use_history=True,
        use_past_error_logs=True,
        use_action_history=True,
        use_think_history=True,
        use_diff=True,
        use_screenshot=False,
    ),
    use_abstract_example=False,
    use_concrete_example=False,
    enable_chat=True,
    action=ActionFlags(
        action_set=HighLevelActionSetArgs(
            subsets=["bid", "nav"],
            multiaction=False,
            strict=False,
        )
    ),
)

# ---------------------------------------------------------------------------
# 6. Two agent configurations
# ---------------------------------------------------------------------------
# Agent A: GenericAgent + RLM
agent_rlm = GenericAgentArgs(
    chat_model_args=rlm_model_args,
    flags=flags,
    max_retry=2,
)

# Agent B: plain GenericAgent (no RLM)
agent_plain = GenericAgentArgs(
    chat_model_args=opus_model_args,
    flags=flags,
    max_retry=2,
)

# ---------------------------------------------------------------------------
# 7. Benchmark: WorkArena L2
# ---------------------------------------------------------------------------
# NOTE: The registered name in bgym is "workarena_l2_agent_curriculum_eval"
#       (there is no plain "workarena_l2" key).
benchmark_name = "workarena_l2_agent_curriculum_eval"
benchmark = bgym.DEFAULT_BENCHMARKS[benchmark_name]()

# Pick a single L2 task to start with.
# "navigate-and-order-apple-watch-l2" is a natural step up from the L1 order-apple-watch task.
# Override via env var:  L2_TASK_REGEX="some-other-task" python run_workarenal2_opus.py
task_name_regex = os.getenv(
    "L2_TASK_REGEX",
    "workarena.servicenow.navigate-and-order-apple-watch-l2",
)

try:
    benchmark = benchmark.subset_from_regexp("task_name", task_name_regex)
except (AttributeError, Exception) as e:
    logger.warning(f"Could not filter benchmark: {e}. Running full L2 benchmark.")

logger.info(f"Benchmark after filtering:\n{benchmark}")

# ---------------------------------------------------------------------------
# 8. Run parameters
# ---------------------------------------------------------------------------
n_jobs = int(os.getenv("L2_N_JOBS", "1"))
parallel_backend = os.getenv("L2_PARALLEL_BACKEND", "sequential")

# ---------------------------------------------------------------------------
# 9. Run both studies sequentially
# ---------------------------------------------------------------------------
if __name__ == "__main__":

    # ---- Study 1: GenericAgent + RLM ----
    study_rlm = Study(
        [agent_rlm],
        benchmark,
        logging_level_stdout=logging.INFO,
        logging_level=logging.INFO,
        avg_step_timeout=180,  # RLM steps are slower; generous timeout
    )

    print("=" * 70)
    print(f"STUDY 1: GenericAgent + RLM  ({MODEL_NAME})")
    print("=" * 70)
    print(f"  Model (root):      {opus_model_args.model_name}")
    print(f"  Model (recursive): {recursive_model_args.model_name}")
    print(f"  RLM iterations:    {rlm_model_args.max_iterations}")
    print(f"  RLM max_depth:     {rlm_model_args.max_depth}")
    print(f"  Task regex:        {task_name_regex}")
    print(f"  Backend: {parallel_backend}, n_jobs: {n_jobs}")

    study_rlm.run(
        n_jobs=n_jobs,
        parallel_backend=parallel_backend,
        strict_reproducibility=False,
        n_relaunch=1,
    )
    print("Study 1 (RLM) completed.\n")

    # # ---- Study 2: plain GenericAgent (no RLM) ----
    # study_plain = Study(
    #     [agent_plain],
    #     benchmark,
    #     logging_level_stdout=logging.INFO,
    #     logging_level=logging.INFO,
    #     avg_step_timeout=120,
    # )

    # print("=" * 70)
    # print(f"STUDY 2: GenericAgent  (plain, no RLM)  ({MODEL_NAME})")
    # print("=" * 70)
    # print(f"  Model: {opus_model_args.model_name}")
    # print(f"  Task regex: {task_name_regex}")
    # print(f"  Backend: {parallel_backend}, n_jobs: {n_jobs}")

    # study_plain.run(
    #     n_jobs=n_jobs,
    #     parallel_backend=parallel_backend,
    #     strict_reproducibility=False,
    #     n_relaunch=1,
    # )
    # print("Study 2 (plain) completed.\n")

    # print("=" * 70)
    # print("Both studies finished. Check .tmp_results/ for outputs.")
    # print("=" * 70)
