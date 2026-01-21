"""
Unit tests for AgentQ components.

Tests cover:
- MCTSNode dataclass and UCB1 calculation
- Selector components (MaxVisitSelector, AheadKSelector)
- Evaluator/Critic components (mocked LLM)
- AgentQArgs configuration
- DPO pair generation logic
- Browser forking utilities (mocked)
"""

import math
import re
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from agentlab.agents.agentq.agentq import AgentQ, AgentQArgs
from agentlab.agents.agentq.evaluators import (
    AbsoluteCritic,
    BaseCritic,
    TournamentCritic,
)
from agentlab.agents.agentq.mcts import MCTS, MCTSNode
from agentlab.agents.agentq.selectors import AheadKSelector, MaxVisitSelector
from agentlab.llm.chat_api import BaseModelArgs


# =============================================================================
# Timing Utilities
# =============================================================================

@contextmanager
def timer(name: str, threshold_ms: float = 0):
    """Context manager for timing code blocks.
    
    Args:
        name: Label for the timing log
        threshold_ms: Only log if duration exceeds this (0 = always log)
    """
    start = time.perf_counter()
    yield
    elapsed_ms = (time.perf_counter() - start) * 1000
    if elapsed_ms >= threshold_ms:
        print(f"⏱️  {name}: {elapsed_ms:.2f}ms")


def log_dpo_pairs(pairs: list[dict], max_action_len: int = 40):
    """Log DPO pairs in a compact format."""
    if not pairs:
        print("📋 DPO pairs: (empty)")
        return
    
    print(f"📋 DPO pairs ({len(pairs)} total):")
    for i, p in enumerate(pairs):
        chosen = p.get('chosen', '')[:max_action_len]
        rejected = p.get('rejected', '')[:max_action_len]
        c_score = p.get('chosen_score', 0)
        r_score = p.get('rejected_score', 0)
        q_gap = p.get('q_gap', abs(c_score - r_score))
        print(f"  [{i}] ✓ {chosen!r} ({c_score:.3f}) vs ✗ {rejected!r} ({r_score:.3f}) gap={q_gap:.3f}")

# =============================================================================
# Mock LLM for testing
# =============================================================================

@dataclass
class MockLLMArgs(BaseModelArgs):
    """Mock LLM args for testing."""
    model_name: str = "test/mock_llm"
    
    def make_model(self):
        return MockLLM()


class MockLLM:
    """Mock LLM that returns predictable responses for testing."""
    
    def __init__(self):
        self.call_count = 0
        self.responses = []
    
    def __call__(self, messages) -> dict:
        self.call_count += 1
        
        # Check message content to determine response type
        content = str(messages)
        
        # For action generation requests
        if "Suggest" in content and "distinct" in content:
            return dict(role="assistant", content="""
<action>
click("button[id=submit]")
</action>
<action>
fill("input[name=search]", "test")
</action>
<action>
click("a[href='/next']")
</action>
""")
        
        # For critic/evaluation requests - score extraction
        if "score" in content.lower() or "rate" in content.lower():
            return dict(role="assistant", content="Score: 0.75")
        
        # For terminal check
        if "terminal" in content.lower() or "goal achieved" in content.lower():
            return dict(role="assistant", content="No, the goal is not achieved.")
        
        # For tournament "pick best" requests
        if "best" in content.lower() and "action" in content.lower():
            return dict(role="assistant", content="1")  # Pick first action
        
        # Default response
        return dict(role="assistant", content="<action>click('button')</action>")
    
    def get_stats(self):
        return {"call_count": self.call_count}


# =============================================================================
# MCTSNode Tests
# =============================================================================

class TestMCTSNode:
    """Tests for MCTSNode dataclass."""
    
    def test_node_creation(self):
        """Test basic node creation."""
        node = MCTSNode(obs={"url": "http://test.com"}, history=["action1"])
        assert node.obs == {"url": "http://test.com"}
        assert node.history == ["action1"]
        assert node.visits == 0
        assert node.value == 0.0
        assert node.fast_reward == 0.0
        assert node.parent is None
        assert node.children == []
    
    def test_ucb1_unvisited_node(self):
        """Test UCB1 for unvisited node uses fast_reward as prior."""
        node = MCTSNode(obs={}, history=[], fast_reward=0.8)
        # Unvisited nodes should return fast_reward + exploration bonus
        ucb = node.ucb1(exploration_constant=1.41)
        # Should be fast_reward + high exploration bonus
        assert ucb > 0.8  # Has exploration bonus
        assert ucb == 0.8 + 1.41 * 10  # fast_reward + c * 10
    
    def test_ucb1_visited_node(self):
        """Test UCB1 for visited node with α=1.0 (pure MCTS)."""
        parent = MCTSNode(obs={}, history=[])
        parent.visits = 10
        
        child = MCTSNode(obs={}, history=[], parent=parent)
        child.visits = 5
        child.value = 3.0  # cumulative value
        
        # With α=1.0: Q_mixed = Q̃ = value/visits
        # UCB1 = Q_mixed + c * sqrt(ln(parent_visits) / visits)
        expected_exploitation = 3.0 / 5  # 0.6
        expected_exploration = 1.41 * math.sqrt(math.log(10) / 5)
        expected_ucb = expected_exploitation + expected_exploration
        
        ucb = child.ucb1(exploration_constant=1.41, alpha=1.0)
        assert abs(ucb - expected_ucb) < 0.01
    
    def test_node_with_children(self):
        """Test node with children."""
        parent = MCTSNode(obs={"url": "http://test.com"}, history=[])
        child1 = MCTSNode(obs={}, history=["a1"], parent=parent, action="click('1')")
        child2 = MCTSNode(obs={}, history=["a2"], parent=parent, action="click('2')")
        parent.children = [child1, child2]
        
        assert len(parent.children) == 2
        assert child1.parent is parent
        assert child1.action == "click('1')"


# =============================================================================
# Selector Tests
# =============================================================================

class TestMaxVisitSelector:
    """Tests for MaxVisitSelector."""
    
    def test_select_most_visited(self):
        """Test that selector picks most visited child (returns node)."""
        selector = MaxVisitSelector()
        
        root = MCTSNode(obs={}, history=[])
        child1 = MCTSNode(obs={}, history=["a1"], parent=root, action="action1")
        child1.visits = 3
        child1.value = 1.5
        
        child2 = MCTSNode(obs={}, history=["a2"], parent=root, action="action2")
        child2.visits = 7  # Most visited
        child2.value = 3.0
        
        child3 = MCTSNode(obs={}, history=["a3"], parent=root, action="action3")
        child3.visits = 2
        child3.value = 1.8
        
        root.children = [child1, child2, child3]
        
        selected = selector.select(root)
        # Selector returns the node, not the action string
        assert selected is child2
        assert selected.action == "action2"
    
    def test_empty_children(self):
        """Test with no children returns None."""
        selector = MaxVisitSelector()
        root = MCTSNode(obs={}, history=[])
        assert selector.select(root) is None


class TestAheadKSelector:
    """Tests for AheadKSelector."""
    
    def test_ahead_k_clear_winner(self):
        """Test ahead-k with clear winner (gap >= k)."""
        selector = AheadKSelector(k=5)
        
        root = MCTSNode(obs={}, history=[])
        child1 = MCTSNode(obs={}, history=["a1"], parent=root, action="action1")
        child1.visits = 3
        
        child2 = MCTSNode(obs={}, history=["a2"], parent=root, action="action2")
        child2.visits = 10  # 10-3=7 >= k=5, clear winner
        
        root.children = [child1, child2]
        
        selected = selector.select(root)
        assert selected is child2
        assert selected.action == "action2"
    
    def test_ahead_k_no_clear_winner(self):
        """Test ahead-k returns None when gap < k."""
        selector = AheadKSelector(k=10)
        
        root = MCTSNode(obs={}, history=[])
        child1 = MCTSNode(obs={}, history=["a1"], parent=root, action="action1")
        child1.visits = 5
        
        child2 = MCTSNode(obs={}, history=["a2"], parent=root, action="action2")
        child2.visits = 8  # 8-5=3 < k=10, no clear winner
        
        root.children = [child1, child2]
        
        selected = selector.select(root)
        # Should return None when gap < k
        assert selected is None
    
    def test_ahead_k_single_child(self):
        """Test ahead-k with single child returns that child."""
        selector = AheadKSelector(k=5)
        
        root = MCTSNode(obs={}, history=[])
        child = MCTSNode(obs={}, history=["a1"], parent=root, action="action1")
        child.visits = 3
        
        root.children = [child]
        
        selected = selector.select(root)
        assert selected is child


# =============================================================================
# Critic Tests (with mocked LLM)
# =============================================================================

@pytest.fixture
def minimal_obs_flags():
    """Minimal ObsFlags for testing critics."""
    from agentlab.agents.dynamic_prompting import ObsFlags
    return ObsFlags(
        use_html=False,
        use_ax_tree=True,
        use_screenshot=False,
    )


@pytest.fixture  
def mock_obs_for_critic():
    """Mock observation with required fields for critic tests."""
    return {
        "url": "http://test.com",
        "axtree_txt": "[1] button 'Submit'\n[2] input 'Name'",
        "pruned_html": "<button>Submit</button>",
        "screenshot": None,
        "dom_txt": "",
        "focused_element_bid": "1",
        "last_action_error": "",
        "open_pages_urls": ["http://test.com"],
        "open_pages_titles": ["Test Page"],
    }


class TestAbsoluteCritic:
    """Tests for AbsoluteCritic."""
    
    def test_evaluate_single(self, minimal_obs_flags, mock_obs_for_critic):
        """Test single action evaluation."""
        mock_llm = MockLLM()
        critic = AbsoluteCritic(mock_llm)
        
        with timer("AbsoluteCritic.evaluate"):
            score = critic.evaluate(
                goal="Click the submit button",
                action="click('1')",
                current_obs=mock_obs_for_critic,
                obs_flags=minimal_obs_flags,
                pre_obs_summary="Previous page state",
                action_error=None
            )
        
        print(f"  Score: {score:.3f}")
        
        # Returns a float score
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0


class TestTournamentCritic:
    """Tests for TournamentCritic."""
    
    def test_rank_actions_tournament(self, minimal_obs_flags, mock_obs_for_critic):
        """Test tournament ranking returns ordered actions."""
        mock_llm = MockLLM()
        critic = TournamentCritic(mock_llm)
        
        actions = ["click('1')", "fill('input', 'test')", "click('2')"]
        
        with timer("TournamentCritic.rank_actions_tournament (3 actions)"):
            ranked = critic.rank_actions_tournament(
                goal="Fill the form",
                state_obs=mock_obs_for_critic,
                actions=actions,
                obs_flags=minimal_obs_flags
            )
        
        print(f"  Ranked: {[(a[:20], f'{r:.2f}') for a, r in ranked]}")
        
        # Should return list of (action, rank) tuples
        assert len(ranked) == 3
        # Ranks should be decreasing (1.0, 0.5, 0.33...)
        ranks = [r[1] for r in ranked]
        assert ranks[0] >= ranks[1] >= ranks[2]
        # All original actions should be present
        ranked_actions = [r[0] for r in ranked]
        for action in actions:
            assert action in ranked_actions
    
    def test_rank_single_action(self, minimal_obs_flags, mock_obs_for_critic):
        """Test tournament with single action returns rank 1.0."""
        mock_llm = MockLLM()
        critic = TournamentCritic(mock_llm)
        
        actions = ["click('1')"]
        ranked = critic.rank_actions_tournament(
            goal="Click button",
            state_obs=mock_obs_for_critic,
            actions=actions,
            obs_flags=minimal_obs_flags
        )
        
        assert len(ranked) == 1
        assert ranked[0][0] == "click('1')"
        assert ranked[0][1] == 1.0
    
    def test_rank_empty_actions(self, minimal_obs_flags, mock_obs_for_critic):
        """Test tournament with empty actions returns empty list."""
        mock_llm = MockLLM()
        critic = TournamentCritic(mock_llm)
        
        ranked = critic.rank_actions_tournament(
            goal="Click button",
            state_obs=mock_obs_for_critic,
            actions=[],
            obs_flags=minimal_obs_flags
        )
        
        assert ranked == []
    
    def test_evaluate_batch_compatibility(self, minimal_obs_flags, mock_obs_for_critic):
        """Test that evaluate_batch works for backward compatibility."""
        mock_llm = MockLLM()
        critic = TournamentCritic(mock_llm)
        
        # Batch API requires specific candidate format
        candidates = [
            {"action": "click('1')", "obs": mock_obs_for_critic, "pre_obs_summary": "state", "error": None},
            {"action": "click('2')", "obs": mock_obs_for_critic, "pre_obs_summary": "state", "error": None},
        ]
        
        with timer("TournamentCritic.evaluate_batch (2 candidates)"):
            scores = critic.evaluate_batch(
                goal="Click the button",
                candidates=candidates,
                obs_flags=minimal_obs_flags
            )
        
        print(f"  Batch scores: {[f'{s:.3f}' for s in scores]}")
        
        assert len(scores) == 2
        assert all(isinstance(s, float) for s in scores)


# =============================================================================
# AgentQ Configuration Tests
# =============================================================================

class TestAgentQArgs:
    """Tests for AgentQArgs configuration."""
    
    def test_default_args(self):
        """Test default argument values."""
        args = AgentQArgs(
            chat_model_args=MockLLMArgs(),
            flags=None  # Will use defaults
        )
        
        assert args.mcts_budget == 5
        assert args.mcts_max_workers == 4
        assert args.mcts_rollout_depth == 3
        assert args.critic_type == "tournament"
        assert args.selection_strategy == "max_visit"
        assert args.use_real_rollouts is False
    
    def test_custom_args(self):
        """Test custom argument values."""
        args = AgentQArgs(
            chat_model_args=MockLLMArgs(),
            flags=None,
            mcts_budget=10,
            mcts_max_workers=2,
            mcts_rollout_depth=5,
            critic_type="absolute",
            selection_strategy="ahead_k",
            ahead_k=2,
            use_real_rollouts=True
        )
        
        assert args.mcts_budget == 10
        assert args.mcts_max_workers == 2
        assert args.critic_type == "absolute"
        assert args.selection_strategy == "ahead_k"
        assert args.ahead_k == 2
        assert args.use_real_rollouts is True


# =============================================================================
# DPO Pair Generation Tests
# =============================================================================

class TestDPOPairGeneration:
    """Tests for DPO preference pair generation."""
    
    def test_generate_dpo_pairs_basic(self):
        """Test basic DPO pair generation from tree."""
        with timer("build tree with winner/loser"):
            # Create a simple tree with clear winner/loser
            root = MCTSNode(
                obs={"url": "http://test.com", "axtree_txt": "test"},
                history=[]
            )
            root.visits = 10
            
            winner = MCTSNode(
                obs={"url": "http://test.com/page1"},
                history=["action1"],
                parent=root,
                action="click('winner')"
            )
            winner.visits = 7
            winner.value = 5.0
            
            loser = MCTSNode(
                obs={"url": "http://test.com/page2"},
                history=["action2"],
                parent=root,
                action="click('loser')"
            )
            loser.visits = 3
            loser.value = 1.0
            
            root.children = [winner, loser]
        
        with timer("generate DPO pairs (manual logic)"):
            # Mock the MCTS generate_dpo_pairs logic
            pairs = []
            if len(root.children) >= 2:
                sorted_children = sorted(root.children, key=lambda c: (c.visits, c.value), reverse=True)
                best = sorted_children[0]
                for other in sorted_children[1:]:
                    if best.visits > other.visits or best.value > other.value:
                        pairs.append({
                            'state_summary': str(root.obs.get('url', '')),
                            'goal': 'Test goal',
                            'chosen': best.action,
                            'chosen_score': best.value / max(best.visits, 1),
                            'rejected': other.action,
                            'rejected_score': other.value / max(other.visits, 1),
                        })
        
        log_dpo_pairs(pairs)
        
        assert len(pairs) == 1
        assert pairs[0]['chosen'] == "click('winner')"
        assert pairs[0]['rejected'] == "click('loser')"


# =============================================================================
# Browser Forking Tests (mocked)
# =============================================================================

class TestBrowserForking:
    """Tests for browser forking utilities."""
    
    def test_fork_result_dataclass(self):
        """Test ForkResult dataclass."""
        from agentlab.agents.browser_forking import ForkResult
        
        result = ForkResult(
            obs={"url": "http://test.com"},
            error=None,
            success=True,
            execution_time=1.5
        )
        
        assert result.success is True
        assert result.obs["url"] == "http://test.com"
        assert result.error is None
    
    def test_rollout_result_dataclass(self):
        """Test RolloutResult dataclass."""
        from agentlab.agents.browser_forking import RolloutResult, RolloutStep
        
        step = RolloutStep(
            action="click('1')",
            obs={"url": "http://test.com"},
            error=None,
            reward=0.8
        )
        
        result = RolloutResult(
            steps=[step],
            final_obs={"url": "http://final.com"},
            cumulative_reward=0.8,
            terminated=False,
            success=True
        )
        
        assert len(result.steps) == 1
        assert result.cumulative_reward == 0.8
        assert result.success is True


# =============================================================================
# Integration Tests (lightweight, no actual browser)
# =============================================================================

class TestMCTSIntegration:
    """Integration tests for MCTS with mocked components."""
    
    @pytest.fixture
    def mock_mcts(self):
        """Create MCTS with mocked LLM."""
        from bgym import HighLevelActionSetArgs

        from agentlab.agents.dynamic_prompting import ActionFlags, ObsFlags
        from agentlab.agents.generic_agent.generic_agent_prompt import (
            GenericPromptFlags,
        )
        
        mock_llm = MockLLM()
        
        flags = GenericPromptFlags(
            obs=ObsFlags(
                use_html=False,
                use_ax_tree=True,
            ),
            action=ActionFlags(
                action_set=HighLevelActionSetArgs(subsets=["bid"])
            )
        )
        
        action_set = flags.action.action_set.make_action_set()
        
        mcts = MCTS(
            chat_llm=mock_llm,
            critique_llm=mock_llm,
            action_set=action_set,
            flags=flags,
            headless=True,
            rollout_depth=2
        )
        
        return mcts
    
    @pytest.fixture
    def mock_obs(self):
        """Create a mock observation with required fields."""
        return {
            "url": "http://test.com",
            "axtree_txt": "[1] button 'Submit'\n[2] input 'Name'",
            "pruned_html": "<button>Submit</button>",
            "screenshot": None,
            "dom_txt": "",
            "focused_element_bid": "1",
            "last_action_error": "",
            "open_pages_urls": ["http://test.com"],
            "open_pages_titles": ["Test Page"],
        }
    
    def test_generate_candidate_actions(self, mock_mcts, mock_obs):
        """Test action generation."""
        node = MCTSNode(obs=mock_obs, history=[])
        
        with timer("generate_candidate_actions"):
            actions = mock_mcts.generate_candidate_actions(node, goal="Click submit", n=3)
        
        print(f"  Generated {len(actions)} actions: {[a[:30] + '...' if len(a) > 30 else a for a in actions]}")
        
        # Should return list of action strings
        assert isinstance(actions, list)
        assert len(actions) <= 3
    
    def test_select_unvisited_child(self, mock_mcts):
        """Test selection of unvisited children prioritizes fast_reward."""
        root = MCTSNode(obs={}, history=[])
        root.visits = 5
        
        child1 = MCTSNode(obs={}, history=["a1"], parent=root, action="a1", fast_reward=0.3)
        child1.visits = 0
        
        child2 = MCTSNode(obs={}, history=["a2"], parent=root, action="a2", fast_reward=0.9)
        child2.visits = 0
        
        root.children = [child1, child2]
        
        selected = mock_mcts.select(root)
        
        # Should select child2 due to higher fast_reward
        assert selected.action == "a2"
    
    def test_backpropagate(self, mock_mcts):
        """Test backpropagation updates visits and values."""
        root = MCTSNode(obs={}, history=[])
        child = MCTSNode(obs={}, history=["a1"], parent=root, action="a1")
        grandchild = MCTSNode(obs={}, history=["a1", "a2"], parent=child, action="a2")
        
        root.children = [child]
        child.children = [grandchild]
        
        with timer("backpropagate (depth=2)"):
            mock_mcts.backpropagate(grandchild, reward=0.8)
        
        print(f"  Path: grandchild→child→root, reward=0.8")
        print(f"  Values: gc={grandchild.value:.2f}, c={child.value:.2f}, r={root.value:.2f}")
        
        # All nodes in path should have updated visits and values
        assert grandchild.visits == 1
        assert grandchild.value == 0.8
        assert child.visits == 1
        assert child.value == 0.8
        assert root.visits == 1
        assert root.value == 0.8


# =============================================================================
# Thread Safety Tests
# =============================================================================

class TestThreadSafety:
    """Tests for thread safety in MCTS."""
    
    def test_tree_lock_exists(self):
        """Test that MCTS has a tree_lock."""
        from bgym import HighLevelActionSetArgs

        from agentlab.agents.dynamic_prompting import ActionFlags, ObsFlags
        from agentlab.agents.generic_agent.generic_agent_prompt import (
            GenericPromptFlags,
        )
        
        mock_llm = MockLLM()
        
        flags = GenericPromptFlags(
            obs=ObsFlags(use_html=False, use_ax_tree=True),
            action=ActionFlags(action_set=HighLevelActionSetArgs(subsets=["bid"]))
        )
        
        action_set = flags.action.action_set.make_action_set()
        
        mcts = MCTS(
            chat_llm=mock_llm,
            critique_llm=mock_llm,
            action_set=action_set,
            flags=flags,
        )
        
        # Verify tree_lock is a threading.Lock
        import threading
        assert hasattr(mcts, 'tree_lock')
        assert isinstance(mcts.tree_lock, type(threading.Lock()))
    
    def test_terminal_caching(self):
        """Test that terminal status is cached on nodes."""
        node = MCTSNode(obs={"url": "test.com"}, history=[])
        
        # Initially None
        assert node.is_terminal is None
        
        # Can be set
        node.is_terminal = True
        assert node.is_terminal is True
        
        # Can be False too
        node2 = MCTSNode(obs={}, history=[])
        node2.is_terminal = False
        assert node2.is_terminal is False


class TestVirtualLoss:
    """Tests for virtual loss handling."""
    
    def test_backprop_with_virtual_loss(self):
        """Test _backprop_with_virtual_loss accounts for pre-incremented visits."""
        from bgym import HighLevelActionSetArgs

        from agentlab.agents.dynamic_prompting import ActionFlags, ObsFlags
        from agentlab.agents.generic_agent.generic_agent_prompt import (
            GenericPromptFlags,
        )
        
        mock_llm = MockLLM()
        flags = GenericPromptFlags(
            obs=ObsFlags(use_html=False, use_ax_tree=True),
            action=ActionFlags(action_set=HighLevelActionSetArgs(subsets=["bid"]))
        )
        action_set = flags.action.action_set.make_action_set()
        
        mcts = MCTS(
            chat_llm=mock_llm,
            critique_llm=mock_llm,
            action_set=action_set,
            flags=flags,
        )
        
        root = MCTSNode(obs={}, history=[])
        child = MCTSNode(obs={}, history=["a1"], parent=root, action="a1")
        root.children = [child]
        
        # Simulate virtual loss (pre-increment)
        child.visits = 1  # Virtual loss applied
        
        # Backprop with virtual loss should only update value, not visits for the node
        mcts._backprop_with_virtual_loss(child, 0.5)
        
        # Child: value updated, visits unchanged (was pre-incremented)
        assert child.value == 0.5
        assert child.visits == 1
        
        # Parent: both visits and value incremented
        assert root.value == 0.5
        assert root.visits == 1


# =============================================================================
# New Method Tests  
# =============================================================================

class TestNewMethods:
    """Tests for new refactored methods."""
    
    @pytest.fixture
    def mock_mcts_instance(self):
        """Create MCTS instance for testing."""
        from bgym import HighLevelActionSetArgs

        from agentlab.agents.dynamic_prompting import ActionFlags, ObsFlags
        from agentlab.agents.generic_agent.generic_agent_prompt import (
            GenericPromptFlags,
        )
        
        mock_llm = MockLLM()
        flags = GenericPromptFlags(
            obs=ObsFlags(use_html=False, use_ax_tree=True),
            action=ActionFlags(action_set=HighLevelActionSetArgs(subsets=["bid"]))
        )
        action_set = flags.action.action_set.make_action_set()
        
        return MCTS(
            chat_llm=mock_llm,
            critique_llm=mock_llm,
            action_set=action_set,
            flags=flags,
        )
    
    def test_is_terminal_obs_uses_compute_is_terminal(self, mock_mcts_instance):
        """Test is_terminal_obs creates temp node and uses _compute_is_terminal."""
        obs = {
            "url": "test.com",
            "metadata": {"reward": 1.0},  # Terminal reward
            "axtree_txt": "",
            "pruned_html": "",
        }
        
        result = mock_mcts_instance.is_terminal_obs(obs, "Test goal")
        
        # Should return True because reward >= 1.0
        assert result is True
    
    def test_compute_is_terminal_none_obs(self, mock_mcts_instance):
        """Test _compute_is_terminal returns False for None obs."""
        node = MCTSNode(obs=None, history=[])
        
        result = mock_mcts_instance._compute_is_terminal(node, "Goal")
        assert result is False
    
    def test_compute_is_terminal_reward_shortcut(self, mock_mcts_instance):
        """Test _compute_is_terminal uses reward >= 1.0 shortcut."""
        node = MCTSNode(
            obs={"metadata": {"reward": 1.0}, "axtree_txt": ""},
            history=[]
        )
        
        result = mock_mcts_instance._compute_is_terminal(node, "Goal")
        assert result is True


# =============================================================================
# Mixed Q-Value Tests (Agent Q Paper Eq. 10)
# =============================================================================

class TestMixedQValue:
    """Tests for mixed Q-value: Q = α*Q̃ + (1-α)*Q̂"""
    
    def test_q_empirical(self):
        """Test Q̃ (empirical MCTS value)."""
        node = MCTSNode(obs={}, history=[])
        node.visits = 4
        node.value = 2.0  # Cumulative
        
        # Q̃ = value / visits = 2.0 / 4 = 0.5
        assert node.q_empirical == 0.5
    
    def test_q_empirical_unvisited(self):
        """Test Q̃ returns 0 for unvisited nodes."""
        node = MCTSNode(obs={}, history=[])
        assert node.q_empirical == 0.0
    
    def test_q_critic(self):
        """Test Q̂ (critic ranking value)."""
        node = MCTSNode(obs={}, history=[], fast_reward=0.8)
        assert node.q_critic == 0.8
    
    def test_q_mixed_equal_weight(self):
        """Test mixed Q with α=0.5 (equal weight)."""
        node = MCTSNode(obs={}, history=[], fast_reward=0.6)
        node.visits = 2
        node.value = 1.0  # Q̃ = 0.5
        
        # Q = 0.5 * 0.5 + 0.5 * 0.6 = 0.25 + 0.3 = 0.55
        assert abs(node.q_mixed(alpha=0.5) - 0.55) < 0.001
    
    def test_q_mixed_pure_mcts(self):
        """Test mixed Q with α=1.0 (pure MCTS)."""
        node = MCTSNode(obs={}, history=[], fast_reward=0.9)
        node.visits = 4
        node.value = 2.0  # Q̃ = 0.5
        
        # α=1.0: Q = 1.0 * 0.5 + 0.0 * 0.9 = 0.5
        assert node.q_mixed(alpha=1.0) == 0.5
    
    def test_q_mixed_pure_critic(self):
        """Test mixed Q with α=0.0 (pure critic)."""
        node = MCTSNode(obs={}, history=[], fast_reward=0.9)
        node.visits = 4
        node.value = 2.0  # Q̃ = 0.5
        
        # α=0.0: Q = 0.0 * 0.5 + 1.0 * 0.9 = 0.9
        assert node.q_mixed(alpha=0.0) == 0.9
    
    def test_q_mixed_unvisited_returns_critic(self):
        """Test unvisited nodes return pure Q̂ regardless of alpha."""
        node = MCTSNode(obs={}, history=[], fast_reward=0.7)
        node.visits = 0
        
        # For unvisited, always return Q̂
        assert node.q_mixed(alpha=0.5) == 0.7
        assert node.q_mixed(alpha=1.0) == 0.7
        assert node.q_mixed(alpha=0.0) == 0.7
    
    def test_ucb1_uses_mixed_q(self):
        """Test UCB1 uses mixed Q-value."""
        parent = MCTSNode(obs={}, history=[])
        parent.visits = 10
        
        child = MCTSNode(obs={}, history=["a1"], parent=parent, fast_reward=0.8)
        child.visits = 5
        child.value = 2.5  # Q̃ = 0.5
        
        # UCB = Q_mixed + c * sqrt(ln(N_parent) / N_child)
        # With α=0.5: Q_mixed = 0.5*0.5 + 0.5*0.8 = 0.65
        # exploration = 1.41 * sqrt(ln(10)/5) ≈ 0.958
        ucb = child.ucb1(exploration_constant=1.41, alpha=0.5)
        
        expected_q = 0.5 * 0.5 + 0.5 * 0.8
        expected_exploration = 1.41 * math.sqrt(math.log(10) / 5)
        expected_ucb = expected_q + expected_exploration
        
        assert abs(ucb - expected_ucb) < 0.01


class TestMixedQValueDPOPairs:
    """Tests for DPO pair generation with mixed Q-value and threshold."""
    
    def test_dpo_pairs_use_mixed_q(self):
        """Test DPO pairs use mixed Q-value for scoring."""
        from bgym import HighLevelActionSetArgs

        from agentlab.agents.dynamic_prompting import ActionFlags, ObsFlags
        from agentlab.agents.generic_agent.generic_agent_prompt import (
            GenericPromptFlags,
        )
        
        with timer("setup MCTS"):
            mock_llm = MockLLM()
            flags = GenericPromptFlags(
                obs=ObsFlags(use_html=False, use_ax_tree=True),
                action=ActionFlags(action_set=HighLevelActionSetArgs(subsets=["bid"]))
            )
            action_set = flags.action.action_set.make_action_set()
            
            mcts = MCTS(
                chat_llm=mock_llm,
                critique_llm=mock_llm,
                action_set=action_set,
                flags=flags,
                alpha=0.5,
                q_threshold=0.0,  # No filtering for this test
            )
        
        with timer("build tree"):
            # Create tree with two visited children
            root = MCTSNode(obs={"url": "test.com"}, history=[])
            root.visits = 10
            
            winner = MCTSNode(
                obs={}, history=["a1"], parent=root, action="click('win')",
                fast_reward=0.9  # High critic score
            )
            winner.visits = 6
            winner.value = 4.8  # Q̃ = 0.8
            
            loser = MCTSNode(
                obs={}, history=["a2"], parent=root, action="click('lose')",
                fast_reward=0.3  # Low critic score
            )
            loser.visits = 4
            loser.value = 1.2  # Q̃ = 0.3
            
            root.children = [winner, loser]
        
        with timer("generate_dpo_pairs"):
            pairs = mcts.generate_dpo_pairs(root, "Test goal")
        
        log_dpo_pairs(pairs)
        
        assert len(pairs) == 1
        pair = pairs[0]
        
        # Winner: Q_mixed = 0.5*0.8 + 0.5*0.9 = 0.85
        # Loser: Q_mixed = 0.5*0.3 + 0.5*0.3 = 0.3
        assert pair['chosen'] == "click('win')"
        assert pair['rejected'] == "click('lose')"
        assert abs(pair['chosen_score'] - 0.85) < 0.01
        assert abs(pair['rejected_score'] - 0.3) < 0.01
    
    def test_dpo_pairs_threshold_filtering(self):
        """Test DPO pairs are filtered by Q-value threshold."""
        from bgym import HighLevelActionSetArgs

        from agentlab.agents.dynamic_prompting import ActionFlags, ObsFlags
        from agentlab.agents.generic_agent.generic_agent_prompt import (
            GenericPromptFlags,
        )
        
        with timer("setup MCTS with high threshold"):
            mock_llm = MockLLM()
            flags = GenericPromptFlags(
                obs=ObsFlags(use_html=False, use_ax_tree=True),
                action=ActionFlags(action_set=HighLevelActionSetArgs(subsets=["bid"]))
            )
            action_set = flags.action.action_set.make_action_set()
            
            # High threshold: should filter out small differences
            mcts = MCTS(
                chat_llm=mock_llm,
                critique_llm=mock_llm,
                action_set=action_set,
                flags=flags,
                alpha=0.5,
                q_threshold=0.5,  # High threshold
            )
        
        with timer("build similar-score tree"):
            root = MCTSNode(obs={"url": "test.com"}, history=[])
            root.visits = 10
            
            # Two children with similar Q-values
            child1 = MCTSNode(
                obs={}, history=["a1"], parent=root, action="click('1')",
                fast_reward=0.6
            )
            child1.visits = 5
            child1.value = 3.0  # Q̃ = 0.6, Q_mixed = 0.6
            
            child2 = MCTSNode(
                obs={}, history=["a2"], parent=root, action="click('2')",
                fast_reward=0.5
            )
            child2.visits = 5
            child2.value = 2.5  # Q̃ = 0.5, Q_mixed = 0.5
            
            root.children = [child1, child2]
        
        with timer("generate_dpo_pairs (should filter)"):
            pairs = mcts.generate_dpo_pairs(root, "Test goal")
        
        print(f"  Q-gap = |0.6 - 0.5| = 0.1 < threshold=0.5 → filtered")
        log_dpo_pairs(pairs)
        
        assert len(pairs) == 0  # Filtered out


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
