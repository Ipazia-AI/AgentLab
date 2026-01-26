
import logging
import os

import browsergym.core  # Register tasks
import gymnasium as gym
from bgym import HighLevelActionSetArgs
from dotenv import find_dotenv, load_dotenv

from agentlab.agents.agentq.agentq import AgentQ
from agentlab.agents.dynamic_prompting import ActionFlags, ObsFlags
from agentlab.agents.generic_agent.generic_agent_prompt import GenericPromptFlags
from agentlab.llm.chat_api import OpenRouterModelArgs


def run_amazon_task():
    # Setup Logging with timestamps and detailed format
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
   

    # 1. Initialize BrowserGym Environment
    # Use 'openended' task to allow free navigation to amazon
    # headless=False to see it
    env = gym.make(
        "browsergym/openended", 
        task_kwargs={"start_url": "https://www.amazon.it", "goal": "Find the price and the name of the cheapest modern laptop"},
        wait_for_user_message=False,
        headless=False,
        #slow_mo=1000, # Slow down to see what happens
    )

    obs, info = env.reset()
    print(f"DEBUG: Type of obs: {type(obs)}")
    print(f"DEBUG: Obs keys: {obs.keys() if isinstance(obs, dict) else 'Not a dict'}")

    # 2. Initialize AgentQ
    # Use OpenRouter with reduced retry wait time for faster error recovery
    chat_args = OpenRouterModelArgs(
        model_name="openai/gpt-5-mini",
        min_retry_wait_time=5  # Reduced from 60s to 5s for faster recovery on API errors
    )
    
    flags = GenericPromptFlags(
        obs=ObsFlags(
            use_html=False,
            use_ax_tree=True, # Use Accessibility Tree
            use_focused_element=True,
            use_error_logs=True,
            use_history=True,
            use_past_error_logs=True,
            use_action_history=True,
            use_think_history=True,
            use_diff=False,
            use_screenshot=False
        ),
        use_abstract_example=True,
        use_concrete_example=True,
        enable_chat=True,
        action=ActionFlags(
            action_set=HighLevelActionSetArgs(
                subsets=["bid", "nav"], # Enable basic interactions and navigation
                multiaction=False,
                strict=False # Allow imperfect actions
            )
        )
    )

    agent = AgentQ(
        chat_model_args=chat_args,
        flags=flags,
        mcts_budget=3,  # Optimized budget for faster verification
        mcts_rollout_depth=2,
        critic_type="tournament",  # Changed to "tournament" for batch ranking (1 LLM call vs 3)
        selection_strategy="max_visit",  # "max_visit" | "ahead_k"
        use_real_rollouts=False,  # False=fast_reward (paper default)
        action_timeout=5,  # Reduced from 10s default for faster failure detection
    )

    goal = "Get me the price and the name of the cheapest modern laptop"
    
    # 3. Execution Loop
    max_steps = 5
    total_reward = 0.0

    print(f"Starting task: {goal}")

    for step in range(max_steps):
        print(f"\n--- Step {step + 1} ---")
        
        # AgentQ internal logic needs goal in the obs usually, or we pass it via prompt
        # GenericAgent usually extracts goal from obs['goal_object'] or chat.
        # BrowserGym openended task puts goal in obs['goal'] if configured or we assume it.
        # But AgentQ.get_action extracts it.
        # note: AgentQ.get_action calls `goal = self._get_goal(obs)`
        # GenericAgent._get_goal checks: if self.flags.enable_chat -> chat_messages.
        # else -> obs["goal"] (str) or obs["goal_object"] ...
        
        # Let's ensure obs has goal if missing
        if "goal" not in obs:
            obs["goal"] = goal
            
        action_str, agent_info = agent.get_action(obs)
        
        if not action_str:
            print("Agent returned no action. Stopping.")
            break
            
        print(f"Agent Action: {action_str}")
        
        # Execute in Environment
        obs, reward, terminated, truncated, info = env.step(action_str)
        total_reward += reward
        
        if terminated or truncated:
            print(f"Environment terminated. Total Reward: {total_reward}")
            break
            
        # Optional: Check if we successfully found the date logic? 
        # For openended, reward is usually 0 unless defined.
        
    env.close()

if __name__ == "__main__":
    load_dotenv(find_dotenv())
    run_amazon_task()
