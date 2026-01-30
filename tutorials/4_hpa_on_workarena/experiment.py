from pathlib import Path

from bgym import DEFAULT_BENCHMARKS
from dotenv import load_dotenv

from agentlab.agents.structured_agent.hpa_agent import HPAAgentArgs
from agentlab.experiments.study import Study


# Load environment variables for WorkArena (e.g., ServiceNow credentials/instance).
project_dir = Path(__file__).parents[2]
load_dotenv(project_dir.joinpath(".env"), override=False)


agent_config = HPAAgentArgs(
    goal_text="Solve the WorkArena task",
    action_subsets=("workarena",),
    multiaction=False,
)

agent_configs = [agent_config]
benchmark = DEFAULT_BENCHMARKS["workarena_l1"]()

# Optionally filter tasks:
# benchmark = benchmark.subset_from_glob(column="task_name", glob="*create*")

n_jobs = 1  # keep 1 for debugging

if __name__ == "__main__":
    study = Study(agent_configs, benchmark)
    study.run(
        n_jobs=n_jobs,
        parallel_backend="ray",
        n_relaunch=1,
    )
