"""
Note: This script is a convenience script to launch experiments instead of using
the command line.

Copy this script and modify at will, but don't push your changes to the
repository.
"""

import logging
from copy import deepcopy

import bgym

from agentlab.agents.tool_use_agent.tool_use_agent import (
    DEFAULT_PROMPT_CONFIG,
    GPT_4_1,
    GPT4_1_OPENROUTER_MODEL,
    ToolUseAgentArgs,
)
from agentlab.agents.structured_agent.structured_agent import StructuredAgentArgs
from agentlab.agents.generic_agent.agent_configs import FLAGS_GPT_3_5
from agentlab.llm.chat_api import OpenRouterModelArgs as ChatOpenRouterModelArgs
from agentlab.experiments.study import Study

logging.getLogger().setLevel(logging.INFO)

config = deepcopy(DEFAULT_PROMPT_CONFIG)
# config.keep_last_n_obs = 1
config.obs.use_som = True


structured_flags = deepcopy(FLAGS_GPT_3_5)
structured_flags.obs.use_som = True

# Chat-API OpenRouter model (accepts list[dict]); do not use response_api's GPT4_1_OPENROUTER_MODEL here.
STRUCTURED_AGENT_CHAT_MODEL = ChatOpenRouterModelArgs(
    model_name="openai/gpt-4.1",
    max_total_tokens=200_000,
    max_input_tokens=200_000,
    max_new_tokens=2_000,
    temperature=0.0,  # use 0 for reproducibility when model supports it
    vision_support=True,
)

agent_configs = [
    # ToolUseAgentArgs(
    #     model_args=GPT4_1_OPENROUTER_MODEL,
    #     config=config,
    # ),
    StructuredAgentArgs(
        chat_model_args=STRUCTURED_AGENT_CHAT_MODEL,
        flags=structured_flags,
    ),
    # ToolUseAgentArgs(
    #     model_args=GPT_4_1,
    #     config=config,
    # ),
]

for agent_config in agent_configs:
    if hasattr(agent_config, "config"):
        agent_config.config.action_subsets = ("workarena",)  # use the workarena action set
    if hasattr(agent_config, "flags"):
        agent_config.flags.obs.use_som = True


# ## select the benchmark to run on
# benchmark = "miniwob_tiny_test"
benchmark = "workarena_l1"


benchmark = bgym.DEFAULT_BENCHMARKS[benchmark](n_repeats=4)  # type: bgym.Benchmark
benchmark = benchmark.subset_from_glob("task_name", "*create*")
# benchmark.EnvArgs.headless = False
# benchmark = DEFAULT_BENCHMARKS["workarena_l1"]()
benchmark.env_args_list = benchmark.env_args_list[:1]
 

for env_args in benchmark.env_args_list:
    print(env_args.task_name)
    # env_args.max_steps = 15
    env_args.headless = False

relaunch = False

## Number of parallel jobs
n_jobs = 10  # Make sure to use 1 job when debugging in VSCode
# n_jobs = 1  # Make sure to use 1 job when debugging in VSCode
# parallel_backend = "ray"
parallel_backend = "sequential"  # activate sequential backend for debugging in VSCode

if __name__ == "__main__":  # necessary for dask backend

    if relaunch:
        #  relaunch an existing study
        study = Study.load_most_recent(contains=None)
        study.find_incomplete(include_errors=True)

    else:
        study = Study(agent_configs, benchmark, logging_level_stdout=logging.WARNING)

    study.run(
        n_jobs=n_jobs,
        parallel_backend=parallel_backend,  # "ray", "joblib" or "sequential"
        strict_reproducibility=False,
        n_relaunch=3,
    )
