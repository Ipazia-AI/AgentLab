"""
RDD Agent Integration for WorkArena Tasks

Simple integration that uses GenericAgent's built-in plan field.

IMPORTANT: RDD planning happens LAZILY on the first get_action() call,
using the actual observation from the environment. This ensures:
  - Exact same seed/task_kwargs as the experiment
  - No redundant environment creation
  - Planning with real initial state
"""

from dataclasses import dataclass

from agentlab.agents.generic_agent.agent_configs import GenericAgentArgs
from agentlab.agents.generic_agent.generic_agent import GenericAgent

from .obs_utils import (
    extract_axtree_flat,
    extract_goal_text,
    extract_html,
    format_state_description,
)
from .plan_refiner import PlanRefiner, RefinerConfig
from .rdd_planner import RDDConfig, SimplifiedRDDPlanner


class RDDAgent(GenericAgent):
    """
    RDD Agent that performs lazy planning on first observation.
    
    This ensures planning uses the exact same seed/task/state as the experiment.
    """
    
    def __init__(
        self,
        chat_model_args,
        flags,
        max_retry=4,
        use_axtree=True,
        use_html=False,
        use_plan_refiner=True,
        refiner_iterations=2,
    ):
        super().__init__(chat_model_args, flags, max_retry)
        
        # Store RDD-specific config
        self._rdd_config = {
            'chat_model_args': chat_model_args,
            'use_axtree': use_axtree,
            'use_html': use_html,
            'use_plan_refiner': use_plan_refiner,
            'refiner_iterations': refiner_iterations,
        }
        self._plan_generated = False
    
    def get_action(self, obs):
        """Override to do lazy planning on first call."""
        # Generate plan on first observation if planning is enabled
        if not self._plan_generated and self.flags.use_plan:
            self._generate_plan_from_obs(obs)
            self._plan_generated = True
        
        # Call parent get_action
        return super().get_action(obs)
    
    def _generate_plan_from_obs(self, obs):
        """Generate RDD plan from the actual observation."""
        print("\n🧠 Generating RDD plan from initial observation...")
        
        # Extract planning context from observation
        goal_text = extract_goal_text(obs)
        if not goal_text:
            print("⚠ Warning: Could not extract goal text from observation")
            goal_text = "Complete the task"
        
        state_desc = format_state_description(obs)
        
        # Extract AXTree if requested
        axtree_flat = None
        if self._rdd_config['use_axtree']:
            axtree_flat = extract_axtree_flat(obs)
            if axtree_flat:
                print(f"✓ Extracted AXTree: {len(axtree_flat)} characters")
            else:
                print("⚠ No AXTree found in observation")
        
        # Extract HTML if requested
        html = None
        if self._rdd_config['use_html']:
            html = extract_html(obs)
            if html:
                print(f"✓ Extracted HTML: {len(html)} characters")
            else:
                print("⚠ No HTML found in observation")
        
        # Initialize planner
        planner_config = RDDConfig(max_depth=2, max_nodes=20)
        llm = self._rdd_config['chat_model_args'].make_model()
        planner = SimplifiedRDDPlanner(llm, planner_config, self.action_set)
        
        # Generate initial plan
        print(f"Task: {goal_text}")
        if state_desc:
            print(f"State: {state_desc}")
        
        result = planner.plan(
            task=goal_text,
            state=state_desc,
            axtree=axtree_flat,
            html=html,
        )
        
        initial_plan = result["plan"]
        print(f"✅ Plan generated with {len(result['graph']['nodes'])} subtasks")
        
        # Refine plan if enabled
        if self._rdd_config['use_plan_refiner']:
            print("\n" + "="*80)
            print("BEFORE REFINEMENT:")
            print("="*80)
            print(initial_plan)
            print("="*80 + "\n")
            
            refiner_config = RefinerConfig(max_iterations=self._rdd_config['refiner_iterations'])
            refiner = PlanRefiner(llm, refiner_config)
            
            refinement_result = refiner.refine(
                plan=initial_plan,
                task=goal_text,
            )
            
            final_plan = refinement_result["refined_plan"]
            
            print("\n" + "="*80)
            print("AFTER REFINEMENT:")
            print("="*80)
            print(final_plan)
            print("="*80 + "\n")
            
            # Show refinement summary
            if refinement_result["improvement_notes"]:
                print("🔧 Improvements:")
                for note in refinement_result["improvement_notes"]:
                    print(f"   • {note}")
                print()
            
            # Store planning data to JSON
            # action_set_list = [list(self.action_set.action_set)] if hasattr(self.action_set, 'action_set') else [list(self.action_set)]
            # store_planning_data_to_json(
            #     plan_before_refinement=initial_plan,
            #     plan_after_refinement=final_plan,
            #     task_name=goal_text,
            #     action_set_at_steps=action_set_list
            # )
            
            self.plan = final_plan
        else:
            self.plan = initial_plan


@dataclass
class RDDAgentArgs(GenericAgentArgs):
    """
    Agent args that creates RDDAgent with lazy planning.
    
    Planning happens on the first observation, ensuring exact same
    seed/task/state as the experiment.
    """

    use_axtree: bool = True  # Whether to extract and use AXTree for planning
    use_html: bool = False    # Whether to extract and use HTML for planning
    use_plan_refiner: bool = True  # Whether to refine the plan after generation
    refiner_iterations: int = 2  # Number of refinement iterations
    
    def make_agent(self):
        """
        Create RDDAgent with lazy planning.
        
        Planning will happen on the first get_action() call using the real observation.
        """
        agent = RDDAgent(
            chat_model_args=self.chat_model_args,
            flags=self.flags,
            max_retry=self.max_retry,
            use_axtree=self.use_axtree,
            use_html=self.use_html,
            use_plan_refiner=self.use_plan_refiner,
            refiner_iterations=self.refiner_iterations,
        )
        
        if self.flags.use_plan:
            print("✓ RDD Agent created - planning will happen on first observation\n")
        else:
            print("⚠ Planning disabled (use_plan=False)\n")
        
        return agent


def store_planning_data_to_json(
    plan_before_refinement: str,
    plan_after_refinement: str,
    task_name: str,
    action_set_at_steps: list[list[str]],
    output_dir: str = "planning_data"
) -> str:
    """
    Store planning data as JSON array element.
    
    Args:
        plan_before_refinement: Initial plan before refinement
        plan_after_refinement: Final plan after refinement
        task_name: Name of the task
        action_set_at_steps: List of action sets at each step (array of arrays)
        output_dir: Directory to store the JSON file
    
    Returns:
        Path to the saved JSON file
    """
    import json
    from datetime import datetime
    from pathlib import Path

    # Create output directory
    output_path = Path(__file__).parent / output_dir
    output_path.mkdir(exist_ok=True)
    
    # Create data entry
    data_entry = {
        "plan_before_refinement": plan_before_refinement,
        "plan_after_refinement": plan_after_refinement,
        "time": datetime.now().isoformat(),
        "task_name": task_name,
        "action_set_at_steps": action_set_at_steps
    }
    
    # Create filename from task name
    safe_task_name = task_name.replace(".", "_").replace("/", "_").replace(" ", "_")
    json_file = output_path / f"{safe_task_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    # Load existing data if file exists, otherwise create new array
    if json_file.exists():
        with open(json_file, "r", encoding="utf-8") as f:
            data_array = json.load(f)
    else:
        data_array = []
    
    # Append new entry
    data_array.append(data_entry)
    
    # Save to file
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(data_array, f, indent=2, ensure_ascii=False)
    
    print(f"  💾 Saved planning data to: {json_file}")
    return str(json_file)


__all__ = ["RDDAgent", "RDDAgentArgs", "store_planning_data_to_json"]
