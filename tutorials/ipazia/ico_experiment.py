from pathlib import Path
from dotenv import load_dotenv
from agentlab.agents.generic_agent import GenericAgentArgs
from agentlab.agents.generic_agent.tmlr_config import BASE_FLAGS
from agentlab.experiments.study import make_study
from agentlab.llm.chat_api import OpenRouterModelArgs

# 1. Load environment variables (API keys) from .env
project_dir = Path(__file__).parents[2]
load_dotenv(project_dir.joinpath(".env"), override=False)  # load .env variables

# 2. Configure the Agent
# We use a pre-configured OpenRouter model from the library
# You can see available keys in src/agentlab/llm/llm_configs.py
gemini_flash_args = OpenRouterModelArgs(
    model_name="google/gemini-3-flash-preview",
    max_total_tokens=1_000_000,  # 1M context window
    max_input_tokens=900_000,  # Safe buffer
    max_new_tokens=8_192,  # Output limit
    temperature=0.5,  # Adjust as needed
)
model_args = (
    gemini_flash_args  # CHAT_MODEL_ARGS_DICT["openrouter/meta-llama/llama-3.1-70b-instruct"]
)

agent_config = GenericAgentArgs(
    chat_model_args=model_args,
    flags=BASE_FLAGS,  # Standard prompting flags used in the paper
)

from bgym import DEFAULT_BENCHMARKS

benchmark = DEFAULT_BENCHMARKS["workarena_l1"]()
benchmark.env_args_list = benchmark.env_args_list[:2]
# metadata = benchmark.task_metadata
# tasks_workload = metadata[((metadata["level"] == "l2") & (metadata["category"] == "planning_and_problem_solving") & (metadata["task_name"].str.contains("workload")))]
# tasks_list = tasks_workload["task_name"].tolist()
# benchmark = benchmark.subset_from_list(tasks_list, benchmark_name_suffix="custom_l2")
# benchmark = benchmark.subset_from_list(["workarena.servicenow.create-hardware-asset"], benchmark_name_suffix="custom_l1")

study = make_study(
    benchmark=benchmark,
    agent_args=[agent_config],
    comment="Testing Gemini Flash on WorkArena L1",
)

if __name__ == "__main__":
    study.run(n_jobs=1)
 