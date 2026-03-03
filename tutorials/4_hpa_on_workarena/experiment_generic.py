from pathlib import Path

from bgym import DEFAULT_BENCHMARKS
from dotenv import load_dotenv

from agentlab.agents.generic_agent.generic_agent import GenericAgentArgs
from agentlab.agents.generic_agent.tmlr_config import BASE_FLAGS
from agentlab.experiments.study import Study
from agentlab.llm.chat_api import OpenRouterModelArgs

# Load environment variables for WorkArena (e.g., ServiceNow credentials/instance).
project_dir = Path(__file__).parents[2]
load_dotenv(project_dir.joinpath(".env"), override=False)

max_total_tokens = 1000 # Change values here!!!!!!
max_new_tokens = 100 # Change values here!!!!!
max_input_tokens = max_total_tokens - max_new_tokens
model_name = ""

chat_model_args = OpenRouterModelArgs(
    model_name=model_name,
    max_total_tokens=max_total_tokens,
    max_input_tokens=max_input_tokens,
    max_new_tokens=max_new_tokens,
)

agent_config = GenericAgentArgs(
    chat_model_args=chat_model_args,
    flags=BASE_FLAGS,
)
agent_config.set_reproducibility_mode()
agent_configs = [agent_config]

benchmark = DEFAULT_BENCHMARKS["workarena_l1"]()
tasks_list = [
    "workarena.servicenow.all-menu",
    "workarena.servicenow.filter-hardware-list",
    "workarena.servicenow.single-chart-value-retrieval",
    "workarena.servicenow.sort-asset-list",
]
benchmark = benchmark.subset_from_list(tasks_list, benchmark_name_suffix="generic_reduced_l1")

n_jobs = 1  # keep 1 for debugging

if __name__ == "__main__":
    study = Study(agent_configs, benchmark)
    study.run(
        n_jobs=n_jobs,
        parallel_backend="sequential",
        n_relaunch=1,
    )
