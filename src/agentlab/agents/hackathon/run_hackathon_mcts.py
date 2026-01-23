import argparse
import logging
from pathlib import Path

import gymnasium as gym
from dotenv import load_dotenv

from agentlab.agents import dynamic_prompting as dp
from agentlab.agents.generic_agent.generic_agent_prompt import GenericPromptFlags
from agentlab.agents.hackathon import HackathonAgentArgs
from agentlab.llm.chat_api import OpenRouterModelArgs


def build_agent(model_name: str, temperature: float) -> HackathonAgentArgs:
    flags = GenericPromptFlags(
        obs=dp.ObsFlags(
            use_html=False,
            use_ax_tree=True,
            use_screenshot=False,
            use_error_logs=True,
            use_history=True,
            use_past_error_logs=True,
            use_action_history=True,
            use_think_history=True,
        ),
        action=dp.ActionFlags(),
        use_plan=True,
        use_thinking=True,
        be_cautious=True,
    )

    model_args = OpenRouterModelArgs(
        model_name=model_name,
        max_total_tokens=128_000,
        max_input_tokens=120_000,
        max_new_tokens=4_096,
        temperature=temperature,
    )

    return HackathonAgentArgs(
        chat_model_args=model_args,
        flags=flags,
        use_langchain=False,
        use_mcts=True,
    )


def run_episode(task_name: str, model_name: str, headless: bool, max_steps: int, seed: int):
    agent_args = build_agent(model_name, temperature=0.5)
    agent = agent_args.make_agent()

    env = gym.make(
        f"browsergym/{task_name}",
        disable_env_checker=True,
        max_episode_steps=max_steps,
        headless=headless,
        action_mapping=agent.action_set.to_python_code,
        use_raw_page_output=True,
    )
    agent.set_env(env)

    obs, _ = env.reset(seed=seed)
    step = 0

    try:
        done = False
        while not done and (max_steps is None or step < max_steps):
            action, agent_info = agent.get_action(obs.copy())
            logging.info("Step %s action: %s", step, action)
            if action is None:
                break
            obs, reward, terminated, truncated, _ = env.step(action)
            logging.info("Step %s reward: %s", step, reward)
            done = terminated or truncated
            step += 1
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description="Run HackathonAgent with active MCTS.")
    parser.add_argument("--task", required=True, help="BrowserGym task name, e.g. workarena_l1")
    parser.add_argument(
        "--model",
        default="google/gemini-3-flash-preview",
        help="OpenRouter model name",
    )
    parser.add_argument("--headless", action="store_true", default=False)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    project_dir = Path(__file__).parents[4]
    load_dotenv(project_dir.joinpath(".env"), override=False)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s:%(lineno)d - %(message)s",
    )

    run_episode(
        task_name=args.task,
        model_name=args.model,
        headless=args.headless,
        max_steps=args.max_steps,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
