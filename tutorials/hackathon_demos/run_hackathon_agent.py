import logging
from pathlib import Path
from dotenv import load_dotenv

import bgym
from agentlab.experiments.study import make_study
from agentlab.llm.chat_api import OpenRouterModelArgs

# Import our new HackathonAgent components
from agentlab.agents.hackathon import HackathonAgentArgs
from agentlab.agents.generic_agent.generic_agent_prompt import GenericPromptFlags
from agentlab.agents import dynamic_prompting as dp

# 1. Load environment variables
project_dir = Path(__file__).parents[2]
load_dotenv(project_dir.joinpath(".env"), override=False)

# 2. Configure the LLM
# Using Gemini Flash via OpenRouter as per your previous example,
# but you can switch to GPT-4o if you have the key.
model_args = OpenRouterModelArgs(
    model_name="google/gemini-3-flash-preview",
    max_total_tokens=1_000_000,
    max_input_tokens=900_000,
    max_new_tokens=8_192,
    temperature=0.5,
)

# 3. Configure the HackathonAgent
flags = GenericPromptFlags(
    obs=dp.ObsFlags(
        use_html=False,
        use_ax_tree=True,  # WorkArena usually relies heavily on AXTree
        use_screenshot=True,  # Enable vision
        use_error_logs=True,
        use_history=True,
        use_past_error_logs=True,
        use_action_history=True,
        use_think_history=True,
    ),
    action=dp.ActionFlags(),
    use_plan=True,  # Enable the planning block in the prompt
    use_thinking=True,  # Enable the thinking block
    be_cautious=True,
)

# We enable use_langchain=True to trigger the Planner Graph we built
agent_config = HackathonAgentArgs(
    chat_model_args=model_args,
    flags=flags,
    use_langchain=True,
    # NOTE: Ensure you have 'langgraph' and 'langchain_openai' installed
    # and OPENAI_API_KEY set for the planner graph (which defaults to OpenAI currently)
)

# 4. Configure the Benchmark
# We select a single task from WorkArena L1 for quick testing
# "workarena.servicenow.list-menu-filter" is a good candidate
benchmark = bgym.DEFAULT_BENCHMARKS["workarena_l1"]()
# Keep only the first task for the demo
benchmark.env_args_list = benchmark.env_args_list[:1]

# 5. Create and Run the Study
study = make_study(
    benchmark=benchmark,
    agent_args=[agent_config],
    comment="HackathonAgent with Planner Graph on WorkArena",
    logging_level=logging.INFO,
)

if __name__ == "__main__":
    print(f"Running study with {len(study.exp_args_list)} experiments...")
    # Using sequential backend for debugging visibility
    study.run(n_jobs=1, parallel_backend="sequential")
