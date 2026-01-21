from pathlib import Path

from bgym import DEFAULT_BENCHMARKS
from dotenv import load_dotenv

from agentlab.agents.htn_agent.agent import HTNAgentArgs, HTNObsConfig
from agentlab.experiments.study import make_study

# 1. Load environment variables (API keys) from .env
project_dir = Path(__file__).parents[2]
load_dotenv(project_dir.joinpath(".env"), override=False)  # load .env variables

# 2. Configure the HTN Agent
agent_args = HTNAgentArgs(
    obs_config=HTNObsConfig(
        use_dom=True,
        use_axtree=True,
    )
)

# 3. Pick a benchmark and optionally slice tasks
benchmark = DEFAULT_BENCHMARKS["workarena_l1"]()
benchmark.env_args_list = benchmark.env_args_list[:2]

study = make_study(
    benchmark=benchmark,  # or workarena_l2 / workarena_l3
    agent_args=[agent_args],
    comment="HTN agent MVP",
)

if __name__ == "__main__":
    study.run(n_jobs=1)