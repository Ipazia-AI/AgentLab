from pathlib import Path

from bgym import DEFAULT_BENCHMARKS
from dotenv import load_dotenv

import agentlab.agents.dynamic_prompting as dp
from agentlab.agents.generic_agent.generic_agent_prompt import GenericPromptFlags
from agentlab.agents.structured_agent.hpa_agent import HPAAgentArgs
from agentlab.agents.structured_agent.hpa_prompt import HPAPromptFlags
from agentlab.experiments.study import Study
from agentlab.llm.chat_api import OpenRouterModelArgs

# Load environment variables for WorkArena (e.g., ServiceNow credentials/instance).
project_dir = Path(__file__).parents[2]
load_dotenv(project_dir.joinpath(".env"), override=False)

prompt_flags = GenericPromptFlags(
    obs=dp.ObsFlags(
        use_html=False,
        use_ax_tree=True,
        use_focused_element=True,
        use_error_logs=True,
        use_history=True,
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
    action=dp.ActionFlags(
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
    obs=prompt_flags.obs,
    action=prompt_flags.action,
    use_constraints=True,
    use_progress=True,
    use_suggestion=True,
    use_plan=prompt_flags.use_plan,
    use_criticise=prompt_flags.use_criticise,
    use_thinking=prompt_flags.use_thinking,
    use_memory=prompt_flags.use_memory,
    use_concrete_example=prompt_flags.use_concrete_example,
    use_abstract_example=prompt_flags.use_abstract_example,
    use_hints=prompt_flags.use_hints,
    enable_chat=prompt_flags.enable_chat,
    max_prompt_tokens=prompt_flags.max_prompt_tokens,
    be_cautious=prompt_flags.be_cautious,
    extra_instructions=prompt_flags.extra_instructions,
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
n_jobs = 1 

if __name__ == "__main__":
    study = Study(agent_configs, benchmark)
    study.run(
        n_jobs=n_jobs,
        parallel_backend="sequential",
        n_relaunch=1,
    )
