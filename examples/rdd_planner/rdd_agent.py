"""
RDD Agent Integration for WorkArena Tasks

Simple integration that uses GenericAgent's built-in plan field.

This version can take either:
  - a natural-language `task` description, or
  - a BrowserGym task_name like 'workarena.servicenow.*'
and will resolve BrowserGym task_names to their goal text via a
one-step env reset before planning.

IMPORTANT: RDD planning happens LAZILY when set_task_name() is called,
not during make_agent(). This allows the experiment loop to set the task
before planning occurs.
"""

from dataclasses import dataclass
from typing import Any

import gymnasium as gym

from agentlab.agents import dynamic_prompting as dp
from agentlab.agents.generic_agent.generic_agent import GenericAgent
from agentlab.agents.generic_agent.agent_configs import GenericAgentArgs

from rdd_planner import SimplifiedRDDPlanner, RDDConfig
from obs_utils import extract_goal_text, extract_axtree_flat, extract_html, format_state_description
from plan_refiner import PlanRefiner, RefinerConfig


def _lazy_register_browsergym_task(task_name: str) -> None:
    """Register BrowserGym tasks via lazy imports (mirrors AgentLab behavior)."""
    if task_name.startswith("miniwob"):
        import browsergym.miniwob  # noqa: F401
    elif task_name.startswith("workarena"):
        import browsergym.workarena  # noqa: F401
    elif task_name.startswith("webarena"):
        import browsergym.webarena  # noqa: F401
        import browsergym.webarenalite  # noqa: F401
        try:
            import browsergym.webarena_verified  # noqa: F401
        except ImportError:
            pass
    elif task_name.startswith("visualwebarena"):
        import browsergym.visualwebarena  # noqa: F401
    elif task_name.startswith("assistantbench"):
        import browsergym.assistantbench  # noqa: F401
    elif task_name.startswith("weblinx"):
        import weblinx_browsergym  # noqa: F401


# Utility functions moved to obs_utils.py module
# Use: from obs_utils import extract_goal_text, extract_axtree_flat, format_state_description


@dataclass
class RDDAgentArgs(GenericAgentArgs):
    """
    Agent args that creates agent with RDD planner.

    The `task` field can be either:
      - a natural-language description, OR
      - a BrowserGym task name like 'workarena.servicenow.*'
    In the latter case we will look up the environment, reset it,
    and use the goal text from the observation as the planning task.
    
    IMPORTANT: Planning happens lazily when set_task_name() is called,
    not during make_agent(). This allows the experiment loop to work correctly.
    """

    task: str | None = None  # natural-language task or BrowserGym task_name
    initial_state: str = ""  # Auto-extracted from observation when available
    use_axtree: bool = True  # Whether to extract and use AXTree for planning
    use_html: bool = True    # Whether to extract and use HTML for planning
    use_plan_refiner: bool = True  # Whether to refine the plan after generation
    refiner_iterations: int = 2  # Number of refinement iterations

    def _resolve_planning_context(self) -> tuple[str, str | None, str | None, str]:
        """
        Convert self.task into planning context (goal text + optional axtree + optional html + state).
        
        Returns:
            Tuple of (goal_text, axtree_flat, html, state_description)
        """
        if not self.task:
            raise ValueError("RDDAgentArgs.task must be set.")

        # If it looks like a BrowserGym task_name, fetch its goal text.
        if any(
            self.task.startswith(prefix)
            for prefix in (
                "workarena.",
                "miniwob.",
                "webarena.",
                "visualwebarena.",
                "assistantbench.",
                "weblinx.",
            )
        ):
            task_name = self.task
            _lazy_register_browsergym_task(task_name)

            env_id = f"browsergym/{task_name}"
            env = gym.make(
                env_id,
                disable_env_checker=True,
                max_episode_steps=1,
                headless=True,
                use_raw_page_output=False,  # Changed to False to get axtree_object
            )
            try:
                obs, _info = env.reset(seed=0)
            finally:
                env.close()

            # Extract goal text
            goal_text = extract_goal_text(obs)
            if not goal_text:
                print(
                    f"Warning: could not extract goal text from env '{env_id}'. "
                    "Falling back to task_name as description."
                )
                goal_text = task_name
            else:
                print(f"Resolved BrowserGym task '{task_name}' to goal:\n  {goal_text}\n")
            
            # Extract AXTree if requested
            axtree_flat = None
            if self.use_axtree:
                axtree_flat = extract_axtree_flat(obs)
                if axtree_flat:
                    print(f"✓ Extracted AXTree: {len(axtree_flat)} characters")
                else:
                    print("⚠ No AXTree found in observation")
            
            # Extract HTML if requested
            html = None
            if self.use_html:
                html = extract_html(obs)
                if html:
                    print(f"✓ Extracted HTML: {len(html)} characters")
                    
                    # Save HTML to file for debugging
                    from pathlib import Path
                    debug_dir = Path(__file__).parent / "debug_html"
                    debug_dir.mkdir(exist_ok=True)
                    
                    # Create filename from task name
                    safe_task_name = task_name.replace(".", "_").replace("/", "_")
                    html_file = debug_dir / f"{safe_task_name}.html"
                    
                    html_file.write_text(html, encoding="utf-8")
                    print(f"  💾 Saved HTML to: {html_file}")
                else:
                    print("⚠ No HTML found in observation")
            
            print()  # Empty line for readability
            
            # Extract state description from observation
            state_desc = format_state_description(obs)
            
            return goal_text, axtree_flat, html, state_desc

        # Already a natural-language task - no AXTree, HTML, or obs available
        return self.task, None, None, self.initial_state
    
    def _generate_plan(self) -> str:
        """
        Generate and optionally refine an RDD plan.
        
        Returns:
            The final plan string (refined if use_plan_refiner is enabled)
        """
        print(f"\n🧠 Generating RDD plan for task: {self.task}")
        
        # Resolve task into planning context (goal + axtree + html + state)
        planning_task, axtree_flat, html, state_desc = self._resolve_planning_context()
        
        # Initialize planner and LLM
        planner_config = RDDConfig(max_depth=2, max_nodes=20)
        llm = self.chat_model_args.make_model()
        planner = SimplifiedRDDPlanner(llm, planner_config)
        
        # Generate initial plan
        print(f"Task: {planning_task}")
        if state_desc:
            print(f"State: {state_desc}")
        
        result = planner.plan(
            task=planning_task,
            state=state_desc,
            axtree=axtree_flat,
            html=html,
        )
        
        initial_plan = result["plan"]
        print(f"✅ Plan generated with {len(result['graph']['nodes'])} subtasks")
        
        # Refine plan if enabled
        if self.use_plan_refiner:
            print("\n" + "="*80)
            print("BEFORE REFINEMENT:")
            print("="*80)
            print(initial_plan)
            print("="*80 + "\n")
            
            refiner_config = RefinerConfig(max_iterations=self.refiner_iterations)
            refiner = PlanRefiner(llm, refiner_config)
            
            refinement_result = refiner.refine(
                plan=initial_plan,
                task=planning_task,
            )
            
            final_plan = refinement_result["refined_plan"]
            
            print("\n" + "="*80)
            print("AFTER REFINEMENT:")
            print("="*80)
            print(final_plan)
            print("="*80 + "\n")
            
            # Show refinement summary
            if refinement_result["improvement_notes"]:
                print(f"🔧 Improvements:")
                for note in refinement_result["improvement_notes"]:
                    print(f"   • {note}")
                print()
            
            return final_plan
        
        return initial_plan

    def make_agent(self):
        """
        Create agent with RDD plan.
        
        The task must be set before calling this method.
        """
        # Create regular GenericAgent
        agent = GenericAgent(
            chat_model_args=self.chat_model_args,
            flags=self.flags,
            max_retry=self.max_retry,
        )
        
        # Generate and store plan if task is set AND planning is enabled
        if self.task and self.flags.use_plan:
            agent.plan = self._generate_plan()
        elif self.task and not self.flags.use_plan:
            print("⚠ Planning disabled (use_plan=False), skipping RDD planning\n")
        else:
            print("⚠ Warning: No task set, skipping RDD planning\n")
        
        return agent


__all__ = ["RDDAgentArgs"]
