import os
from pathlib import Path

import pandas as pd
from bgym import DEFAULT_BENCHMARKS
from dotenv import load_dotenv

import agentlab.agents.dynamic_prompting as dp
from agentlab.agents.generic_agent import (
    AGENT_37_SONNET,
    AGENT_GPT5_MINI,
    AGENT_4o_MINI,
)
from agentlab.agents.generic_agent.generic_agent import GenericAgentArgs
from agentlab.agents.structured_agent.hpa_agent import HPAAgentArgs
from agentlab.agents.structured_agent.hpa_prompt import HPAPromptFlags
from agentlab.experiments.loop import EnvArgs
from agentlab.experiments.study import Study
from agentlab.llm.chat_api import OpenRouterModelArgs

# Load environment variables for WorkArena (e.g., ServiceNow credentials/instance).
project_dir = Path(__file__).parents[2]
load_dotenv(project_dir.joinpath(".env"), override=False)

agent_args = AGENT_GPT5_MINI
agent_args.flags.use_plan = True
chat_model_args = agent_args.chat_model_args
# agent_args.chat_model_args.temperature = 0.0

chat_model_args = OpenRouterModelArgs(
    model_name="google/gemini-3-flash-preview",
    max_total_tokens=1_050_000,
    max_input_tokens=1_050_000 - 65_536,
    max_new_tokens=65_536,
    temperature=0.1,
)

agent_config = HPAAgentArgs(
    chat_model_args=chat_model_args,
    flags=HPAPromptFlags(
        action=agent_args.flags.action,
        obs=agent_args.flags.obs,
        use_concrete_example=agent_args.flags.use_concrete_example,
        use_abstract_example=agent_args.flags.use_abstract_example,
        use_hints=agent_args.flags.use_hints,
        use_memory=agent_args.flags.use_memory,
        use_plan=agent_args.flags.use_plan,
        use_criticise=agent_args.flags.use_criticise,
        use_thinking=agent_args.flags.use_thinking,
        be_cautious=agent_args.flags.be_cautious,
        max_prompt_tokens=agent_args.flags.max_prompt_tokens,
        max_trunc_itr=agent_args.flags.max_trunc_itr,
        extra_instructions=agent_args.flags.extra_instructions,
        use_constraints=True,
        use_progress=True,
        use_suggestion=True,
    ),
    max_retry=agent_args.max_retry,
)
# agent_config = GenericAgentArgs(
#     chat_model_args=agent_args.chat_model_args,
#     flags=agent_args.flags,
#     max_retry=agent_args.max_retry,
# )

agent_config.flags.action = dp.ActionFlags(
    long_description=False,
    individual_examples=False,
)
agent_config.set_reproducibility_mode()

agent_configs = [agent_config]
benchmark = DEFAULT_BENCHMARKS["workarena_l1"]()

df = pd.read_csv("tutorials/4_hpa_on_workarena/task_errors.csv")
benchmark.env_args_list = []
for index, row in df.iterrows():
    env_args = EnvArgs(
        task_name=row["env.task_name"],
        task_seed=row["env.task_seed"],
    )
    benchmark.env_args_list.append(env_args)

n_jobs = 1  # keep 1 for debugging

if __name__ == "__main__":
    study = Study(agent_configs, benchmark)
    study.run(
        n_jobs=n_jobs,
        parallel_backend="sequential",
        n_relaunch=1,
    )
