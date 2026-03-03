from pathlib import Path

from bgym import DEFAULT_BENCHMARKS
from dotenv import load_dotenv

from agentlab.agents.generic_agent.tmlr_config import BASE_FLAGS
from agentlab.agents.structured_agent.hpa_agent import HPAAgentArgs
from agentlab.agents.structured_agent.hpa_prompt import HPAPromptFlags
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

hpa_prompt_flags = HPAPromptFlags(
    obs=BASE_FLAGS.obs,
    action=BASE_FLAGS.action,
    use_constraints=True,
    use_progress=True,
    use_suggestion=True,
    use_plan=BASE_FLAGS.use_plan,
    use_criticise=BASE_FLAGS.use_criticise,
    use_thinking=BASE_FLAGS.use_thinking,
    use_memory=BASE_FLAGS.use_memory,
    use_concrete_example=BASE_FLAGS.use_concrete_example,
    use_abstract_example=BASE_FLAGS.use_abstract_example,
    use_hints=BASE_FLAGS.use_hints,
    enable_chat=BASE_FLAGS.enable_chat,
    max_prompt_tokens=BASE_FLAGS.max_prompt_tokens,
    be_cautious=BASE_FLAGS.be_cautious,
    extra_instructions=BASE_FLAGS.extra_instructions,
)

agent_config = HPAAgentArgs(
    chat_model_args=chat_model_args,
    flags=hpa_prompt_flags,
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
benchmark = benchmark.subset_from_list(tasks_list, benchmark_name_suffix="hpa_reduced_l1")
for env_args in benchmark.env_args_list:
    env_args.max_steps = 50
n_jobs = 1 

if __name__ == "__main__":
    study = Study(agent_configs, benchmark)
    study.override_max_steps(50)
    study.run(
        n_jobs=n_jobs,
        parallel_backend="sequential",
        n_relaunch=1,
    )
