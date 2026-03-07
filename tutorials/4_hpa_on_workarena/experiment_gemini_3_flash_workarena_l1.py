from dataclasses import asdict
from pathlib import Path

from bgym import DEFAULT_BENCHMARKS
from dotenv import load_dotenv

import agentlab.agents.dynamic_prompting as dp
from agentlab.agents.generic_agent import (
    AGENT_37_SONNET,
    AGENT_GPT5_MINI,
    AGENT_4o_MINI,
)
from agentlab.agents.generic_agent.generic_agent_prompt import GenericPromptFlags
from agentlab.agents.generic_agent.tmlr_config import BASE_FLAGS
from agentlab.agents.structured_agent.hpa_agent import HPAAgentArgs
from agentlab.agents.structured_agent.hpa_prompt import HPAPromptFlags
from agentlab.experiments.study import Study
from agentlab.llm.chat_api import OpenRouterModelArgs

# Load environment variables for WorkArena (e.g., ServiceNow credentials/instance).
project_dir = Path(__file__).parents[2]
load_dotenv(project_dir.joinpath(".env"), override=False)

prompt_flags = GenericPromptFlags(
    obs=dp.ObsFlags(
        use_html=BASE_FLAGS.obs.use_html,
        use_ax_tree=BASE_FLAGS.obs.use_ax_tree,
        use_focused_element=BASE_FLAGS.obs.use_focused_element,
        use_error_logs=BASE_FLAGS.obs.use_error_logs,
        use_history=BASE_FLAGS.obs.use_history,
        use_past_error_logs=BASE_FLAGS.obs.use_past_error_logs,
        use_action_history=BASE_FLAGS.obs.use_action_history,
        use_think_history=BASE_FLAGS.obs.use_think_history,
        use_diff=BASE_FLAGS.obs.use_diff,
        html_type=BASE_FLAGS.obs.html_type,
        use_screenshot=BASE_FLAGS.obs.use_screenshot,
        use_som=BASE_FLAGS.obs.use_som,
        extract_visible_tag=BASE_FLAGS.obs.extract_visible_tag,
        extract_clickable_tag=BASE_FLAGS.obs.extract_clickable_tag,
        extract_coords=BASE_FLAGS.obs.extract_coords,
        filter_visible_elements_only=BASE_FLAGS.obs.filter_visible_elements_only,
    ),
    action=dp.ActionFlags(
        multi_actions=BASE_FLAGS.action.multi_actions,
        action_set=BASE_FLAGS.action.action_set,
        long_description=BASE_FLAGS.action.long_description,
        individual_examples=BASE_FLAGS.action.individual_examples,
    ),
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

chat_model_args = OpenRouterModelArgs(
    model_name="google/gemini-3-flash-preview",
    max_total_tokens=131_072,
    max_input_tokens=131_072 - 40_000,
    max_new_tokens=40_000,
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

n_jobs = 5  # keep 1 for debugging

if __name__ == "__main__":
    study = Study(agent_configs, benchmark)
    study.override_max_steps(30)
    study.run(
        n_jobs=n_jobs,
        parallel_backend="ray",
        n_relaunch=1,
        relaunch_errors=True,
    )
