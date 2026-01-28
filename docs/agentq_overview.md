# AgentQ Overview

This document is a tour of how `AgentQ` is built on top of `GenericAgent`, how its configuration flows through
`GenericAgentArgs` and `AgentQArgs`, and how the MCTS/critic/selector components are composed for action selection.
For a deeper algorithmic walkthrough of MCTS itself, see `docs/agentq_mcts.md`.

## Class Hierarchy and Responsibilities

`AgentQ` extends the generic agent stack by adding MCTS-based action selection and a critique-driven scoring loop.
The hierarchy looks like this:

- `AgentArgs` defines the interface for benchmark-specific and reproducibility settings.
- `GenericAgentArgs` adds the base LLM configuration and prompt flags, and produces a `GenericAgent`.
- `AgentQArgs` extends `GenericAgentArgs` with MCTS and critic configuration, and produces an `AgentQ`.

Code anchors:

```
class AgentArgs(AbstractAgentArgs):
    def set_benchmark(self, benchmark: Benchmark, demo_mode: bool):
        pass

    def set_reproducibility_mode(self):
        raise NotImplementedError(...)
```

```
@dataclass
class GenericAgentArgs(AgentArgs):
    chat_model_args: BaseModelArgs = None
    flags: GenericPromptFlags = None
    max_retry: int = 4

    def make_agent(self):
        return GenericAgent(
            chat_model_args=self.chat_model_args, flags=self.flags, max_retry=self.max_retry
        )
```

```
@dataclass
class AgentQArgs(GenericAgentArgs):
    critique_model_args: BaseModelArgs | None = None
    mcts_budget: int = 5
    mcts_max_workers: int = 4
    mcts_rollout_depth: int = 3
    critic_type: str = "tournament"
    selection_strategy: str = "max_visit"
    ...

    def make_agent(self):
        return AgentQ(
            chat_model_args=self.chat_model_args,
            critique_model_args=self.critique_model_args,
            flags=self.flags,
            ...
        )
```

## Configuration Flow (Args → Agent)

`AgentQArgs` extends the generic configuration with MCTS-specific knobs. It sets an `agent_name`, configures an
optional separate critique model, and instantiates `AgentQ` with parameters such as search budget, rollout depth,
and action selection strategy.

The `prepare()`/`close()` methods also ensure both the chat model and the critique model are started and stopped
when present.

## Composition Inside `AgentQ.__init__`

`AgentQ` uses composition to plug in the components it needs for MCTS:

- **Critic**: ranks or scores actions (`TournamentCritic` or `AbsoluteCritic`).
- **Selector**: chooses the best action after search (`MaxVisitSelector` or `AheadKSelector`).
- **MCTS engine**: orchestrates the search loop, using the actor LLM, critic, and browser forks.

Snippet:

```
# Instantiate modular components
critic = (
    TournamentCritic(self.chat_llm)
    if critic_type == "tournament"
    else AbsoluteCritic(self.chat_llm)
)
selector = (
    AheadKSelector(k=ahead_k) if selection_strategy == "ahead_k" else MaxVisitSelector()
)

# Initialize MCTS engine
self.mcts = MCTS(
    chat_llm=self.chat_llm,
    critique_llm=self.critique_llm,
    action_set=self.action_set,
    flags=self.flags,
    rollout_depth=self.mcts_rollout_depth,
    critic=critic,
    selector=selector,
    ...
)
```

This structure keeps `AgentQ` focused on orchestration, while letting you swap critics or selectors without changing
the search algorithm itself.

## Action Loop in `AgentQ.get_action`

`AgentQ` overrides `GenericAgent.get_action` to run MCTS and return the selected action:

1. **Preprocess observation** with the same utilities used by `GenericAgent`.
2. **Track history** and extract the goal from the observation.
3. **Run MCTS** with the configured budget/workers and any in-context DPO pairs.
4. **Fallback** to `GenericAgent` if MCTS fails to produce an action.
5. **Update buffers** (DPO pairs, action history, thoughts) and return `AgentInfo`.

Snippet:

```
best_action, root_node = self.mcts.search(
    root_obs=obs,
    root_history=history_strings,
    goal=goal,
    budget=self.mcts_budget,
    dpo_pairs=self.dpo_pairs,
    max_workers=self.mcts_max_workers,
)

if not best_action:
    return super().get_action(original_obs)
```

This preserves the same public interface as `GenericAgent` while swapping the core decision policy.

## Mixed Q-Value (Why MCTS + Critic Combine)

MCTS nodes combine outcome supervision (empirical value) with process supervision (critic ranking):

```
def q_mixed(self, alpha: float = 0.5) -> float:
    if self.visits == 0:
        return self.q_critic
    return alpha * self.q_empirical + (1 - alpha) * self.q_critic
```

This is why `AgentQArgs.alpha` matters: it tunes how much search vs. critic signal guides action selection.

## Extension Points

These are the main places to extend or tweak behavior:

- **Critics**: implement a new `BaseCritic` to score actions differently.
- **Selectors**: implement a new `ActionSelector` if you want a different policy after search.
- **Prompts**: adjust critique or actor prompts in `src/agentlab/agents/agentq/prompts.py`.
- **Rewards**: `use_env_reward` switches from critic-based scoring to environment rewards.
- **Rollouts**: `use_real_rollouts` uses real browser rollouts instead of fast scoring.

## Summary

`AgentQ` stays compatible with the `GenericAgent` API, but replaces the single-step action generation with a
search-based policy that combines MCTS exploration and LLM-based critique. Configuration flows cleanly through
`AgentQArgs`, and the core logic is composed from reusable pieces (critic, selector, and MCTS engine), keeping
the class easy to extend for new research ideas.
