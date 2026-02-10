from pathlib import Path

from bgym import DEFAULT_BENCHMARKS, HighLevelActionSetArgs
from dotenv import load_dotenv

import agentlab.agents.dynamic_prompting as dp
from agentlab.agents.generic_agent import AGENT_4o_MINI
from agentlab.agents.structured_agent.hpa_agent import HPAAgentArgs
from agentlab.experiments.study import Study

# Load environment variables for WorkArena (e.g., ServiceNow credentials/instance).
project_dir = Path(__file__).parents[2]
load_dotenv(project_dir.joinpath(".env"), override=False)


agent_config = HPAAgentArgs(
    multiaction=False,
    chat_model_args=AGENT_4o_MINI.chat_model_args,
    flags=AGENT_4o_MINI.flags,
    max_retry=AGENT_4o_MINI.max_retry,
)
agent_config.flags.action = dp.ActionFlags(
    long_description=True,
    individual_examples=True,
)

agent_configs = [agent_config]
benchmark = DEFAULT_BENCHMARKS["workarena_l1"](n_repeats=1)

metadata = benchmark.task_metadata
tasks_workload = metadata[metadata["task_name"].str.match("workarena.servicenow.all-menu")]
tasks_list = tasks_workload["task_name"].tolist()
benchmark = benchmark.subset_from_regexp("task_name", "workarena.servicenow.all-menu")


# Optionally filter tasks:
# benchmark = benchmark.subset_from_glob(column="task_name", glob="*create*")

n_jobs = 1  # keep 1 for debugging

if __name__ == "__main__":
    study = Study(agent_configs, benchmark)
    study.run(
        n_jobs=n_jobs,
        parallel_backend="sequential",
        n_relaunch=1,
    )
