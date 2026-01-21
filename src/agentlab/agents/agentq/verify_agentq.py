import logging
import os
from unittest.mock import MagicMock

from bgym import HighLevelActionSetArgs

from agentlab.agents.agentq.agentq import AgentQ, AgentQArgs
from agentlab.agents.dynamic_prompting import ActionFlags, ObsFlags
from agentlab.agents.generic_agent.generic_agent_prompt import GenericPromptFlags
from agentlab.llm.chat_api import OpenRouterModelArgs


def test_agentq():
    logging.basicConfig(level=logging.INFO)
    
    # Read .env for OPENROUTER_API_KEY
    env_path = os.path.expanduser("~/workspace/AgentLab/.env")
    if os.path.exists(env_path):
        with open(env_path, "r") as f:
            for line in f:
                if line.startswith("OPENROUTER_API_KEY="):
                    key = line.strip().split("=", 1)[1]
                    os.environ["OPENROUTER_API_KEY"] = key
                    break
    
    # 1. Setup Mock Components
    chat_args = OpenRouterModelArgs(model_name="openai/gpt-4o") 
    # Mocking llm to avoid real API costs/secrets in this test script if possible,
    # or rely on environment variables if set.
    # For robust verification, we might just instantiate the class and see if it crashes.
    
    flags = GenericPromptFlags(
        obs=ObsFlags(),
        action=ActionFlags(action_set=HighLevelActionSetArgs(subsets=["bid", "nav"]))
    )
    
    agent = AgentQ(
        chat_model_args=chat_args,
        flags=flags,
        mcts_budget=2  # Small budget for test
    )
    
    # Mock LLM calls inside MCTS
    agent.chat_llm = MagicMock()
    # expand response
    agent.chat_llm.query.return_value = "CLICK 42\nTYPE 12 \"test\""
    
    # critique response
    # We might need to mock critique inside MCTS since we didn't inject a separate critique LLM in this test setup properly
    agent.mcts.critique_llm = MagicMock()  # Reuse
    # Mock answer for CritiquePrompt
    # The GenericAgent/LLM structure usually returns a dict or string?
    # HelperPrompt.parse_answer expects string.
    # agent.mcts.critique = ... (method)
    
    # 2. Create Mock Obs
    obs = {
        "url": "http://example.com",
        "cookies": [],
        "storage_state": None,
        "goal_object": {"text": "Click button 42"},
        "dom_object": {"documents": [{"nodes": {"parent": [], "nodeName": [], "nodeValue": [], "attributes": []}}]},
        "axtree_object": {"nodes": {"parent": [], "role": [], "name": [], "description": []}},
        "extra_element_properties": {},
        "screenshot": None,
        "open_pages_urls": ["http://example.com"],
        "open_pages_titles": ["Example"],
        "active_page_index": 0,
        "last_action_error": "",
        "focused_element_bid": None,
        # Even if preprocessor fails, we can pre-populate to be safe
        "dom_txt": "<html><body>mock</body></html>",
        "axtree_txt": "[0] Root"
    }
    
    # 3. Running get_action
    print("Running AgentQ.get_action...")
    try:
        # We need to mock sync_playwright inside mcts.py to avoid real browser launch if we want to be fast,
        # but the USER asked for "Real Browser Forking".
        # So let's NOT mock playwright, but we expect it to fail if logic errors exist.
        # But we need a valid URL. example.com is fine.
        
        action, info = agent.get_action(obs)
        print(f"Agent Action: {action}")
        print("Success!")
        
    except Exception as e:
        print(f"Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_agentq()
