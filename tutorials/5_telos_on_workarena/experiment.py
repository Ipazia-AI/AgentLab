from pathlib import Path

from bgym import DEFAULT_BENCHMARKS
from dotenv import load_dotenv

from agentlab.agents.telos_agent import TelosAgentArgs
from agentlab.experiments.study import Study
from agentlab.llm.chat_api import OpenRouterModelArgs

project_dir = Path(__file__).parents[2]
load_dotenv(project_dir.joinpath(".env"), override=False)

planner_model_args = OpenRouterModelArgs(
    model_name="google/gemini-3-flash-preview",
    max_total_tokens=1_048_576,
    max_input_tokens=1_048_576 - 40_000,
    max_new_tokens=40_000,
)

actor_model_args = OpenRouterModelArgs(
    model_name="openai/gpt-5",
    max_total_tokens=400_000,
    max_input_tokens=400_000 - 40_000,
    max_new_tokens=40_000,
)

agent_config = TelosAgentArgs(
    planner_model_args=planner_model_args,
    actor_model_args=actor_model_args,
    horizon_T=30,
)
agent_configs = [agent_config]

benchmark = DEFAULT_BENCHMARKS["workarena_l1"]()
# First-run smoke: one template, one seed. Comment these out for the full L1 eval.
benchmark = benchmark.subset_from_glob(column="task_name", glob="*all-menu*")
benchmark.env_args_list = benchmark.env_args_list[:1]
n_jobs = 1

if __name__ == "__main__":
    study = Study(agent_configs, benchmark, avg_step_timeout=120)
    study.run(
        n_jobs=n_jobs,
        parallel_backend="sequential",
        n_relaunch=1,
    )
