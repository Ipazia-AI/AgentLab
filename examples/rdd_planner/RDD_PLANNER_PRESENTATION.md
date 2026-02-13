# RDD Planner: Recursive Task Decomposition for AI Agents

## 1. Introduction: How the Recursive Planner Works

The **RDD (Recursive Dependency Decomposition) Planner** is a hierarchical task decomposition system that breaks down complex tasks into manageable subtasks before execution.

### Core Mechanism

The planner operates in two distinct phases:

**Phase 1: Breadth-First Decomposition (BFS)**
- Starts with a root task (e.g., "Click on the numbers in ascending order")
- Systematically splits each task into smaller subtasks level-by-level
- Each subtask can specify dependencies on other subtasks using `{task_id}` notation in the description
- Continues splitting until reaching:
  - Maximum depth (default: 2 levels in current config)
  - Maximum nodes (default: 20 tasks)
  - Atomic tasks (tasks the LLM identifies as "unit problems")

**Phase 2: Depth-First Solving (DFS)**
- After all tasks are decomposed, solves them bottom-up starting from leaf nodes
- Ensures explicit dependencies (from `{task_id}` references) are solved before dependent tasks
- For atomic tasks (unit problems): LLM generates direct solutions via `unit_prompt`
- For composite tasks: LLM merges subtask solutions into coherent parent solutions via `merge_prompt`

### Key Features

1. **Dependency Tracking**: Tasks can explicitly depend on outputs from other tasks using `{task_id}` placeholders in descriptions
2. **Context-Aware Planning**: Uses AXTree (accessibility tree), HTML, current state, and available action set during planning
3. **Plan Refinement**: Optional iterative refinement (default: 2 iterations) removes redundancy while preserving structure
4. **Template-Based Prompting**: Uses three prompt templates: `split_prompt` (decompose), `unit_prompt` (solve atomic), `merge_prompt` (combine solutions)

### Example Flow

```
Task: "Click on the numbers in ascending order"
    ↓
BFS Decomposition (splits tasks level-by-level):
├─ 1. Identify coordinates of numbers (1,2,3,4,5)
│   ├─ 1.1. Extract coordinates of '1', '2', '3', '4', '5'
│   └─ 1.2. Combine coordinates into structured format
├─ 2. Create mouse_click sequence based on {1}
├─ 3. Implement delay (noop) between clicks
└─ 4. Combine actions from {2} and {3}
    ↓
DFS Solving (solves dependencies bottom-up):
Solve: 1.1 → 1.2 → 1 → 2 → 3 → 4 → Root
(Each task's solution substitutes {dependency} references)
```

---

## 2. Motivation: Why Recursive Planning Can Help

### The Core Hypothesis

Standard AI agents execute tasks reactively - they observe the current state and decide on the next action without a comprehensive plan. The RDD Planner tests whether **upfront hierarchical planning** can improve agent performance by providing strategic guidance before execution begins.

### Potential Advantages of Planning

**1. Divide and Conquer**
- Complex tasks are broken into manageable subtasks
- Each subtask has clear scope and objectives
- Reduces the complexity of individual decisions

**2. Strategic Foresight**
- Plan is generated upfront with full context (task goal, current state, available capabilities)
- Agent follows a roadmap rather than exploring randomly
- Potentially reduces wasted actions

**3. Explicit Dependency Management**
- Tasks can reference outputs from prerequisite tasks using `{task_id}` syntax
- During solving phase, `{task_id}` placeholders are replaced with actual solutions from dependency tasks
- Ensures proper execution order (dependencies solved before dependent tasks)
- Prevents "cart before horse" errors (e.g., trying to click before identifying coordinates)

**4. Better Context Utilization**
- Planning phase sees the full initial context (page structure, available actions)
- Execution phase follows the plan with focused observations
- Separates "what to do" (planning) from "how to do it" (execution)

**5. Hierarchical Abstraction**
- High-level tasks provide strategic overview of what needs to be done
- Low-level subtasks provide specific implementation details
- Multi-level decomposition allows reasoning at appropriate granularity

### What We Observed

From the initial experiments on "Click on the numbers in ascending order" task:
- **Plan Generation Works**: Successfully generated hierarchical plan with 18 subtasks (4 main tasks, 14 sub-subtasks)
- **Refinement Reduces Redundancy**: Plan reduced from 18 → 13 lines (iteration 1) → 8 lines (iteration 2)
- **Dependencies Are Tracked**: Tasks reference outputs (e.g., "based on sub-problem [1]", "from sub-problems [2] and [3]")
- **Context Extraction Works**: Successfully extracted AXTree (147 chars) and HTML (4022 chars) from initial page state

### What Needs Validation

To prove the value of RDD planning, we need to measure:
- **Success Rate**: Does planning improve task completion vs. reactive baseline?
- **Efficiency**: Does planning reduce the number of actions/tokens needed?
- **Failure Modes**: Where do planned agents fail vs. reactive agents?
- **Cost Trade-offs**: Does upfront planning cost justify execution improvements?

---

## 3. Integration into AgentLab

I integrated the RDD Planner into AgentLab by building on top of the existing `GenericAgent` architecture, ensuring minimal disruption to the codebase while adding powerful planning capabilities. The core planning mechanism is domain-agnostic and can be applied to various task types.

### Architecture Overview

```
RDDAgentArgs (extends GenericAgentArgs)
    ↓
Creates GenericAgent with flags.use_plan = True
    ↓
Generates RDD Plan before execution
    ↓
Plan stored in agent.plan field
    ↓
GenericAgent uses plan during execution
```

### Key Components

**1. RDDAgentArgs Class** (`rdd_agent.py`)
- Extends `GenericAgentArgs` for seamless integration
- Handles lazy task resolution (BrowserGym task names → goal text)
- Orchestrates planning before agent execution
- Manages context extraction (AXTree, HTML, state)

**2. SimplifiedRDDPlanner** (`rdd_planner.py`)
- Core planning engine implementing BFS decomposition + DFS solving
- Configurable via `RDDConfig`: `max_depth`, `max_nodes`, `unit_statement`
- Uses three prompt templates: `split_prompt.txt`, `unit_prompt.txt`, `merge_prompt.txt`
- Parses subtasks from LLM responses (bullet/numbered lists)
- Resolves `{task_id}` dependencies and substitutes solutions during solving phase

**3. PlanRefiner** (`plan_refiner.py`)
- Post-processing step using LLM to clean up generated plans
- Configured via `RefinerConfig`: `max_iterations` (default: 2)
- Uses `refine_prompt.txt` template to guide refinement
- Removes exact duplicates, circular logic, and redundant verification steps
- Tracks improvement notes (e.g., "Reduced from 18 to 13 lines")

**4. Context Extraction Utilities** (`obs_utils.py`)
- `extract_goal_text(obs)`: Extracts task goal from BrowserGym observations
- `extract_axtree_flat(obs)`: Flattens accessibility tree for LLM consumption
- `extract_html(obs)`: Extracts HTML content from observations
- `format_state_description(obs)`: Creates concise state descriptions (e.g., "On page X at URL Y")

### Integration Flow

1. **Agent Creation**
   - `RDDAgentArgs.make_agent()` creates standard `GenericAgent`
   - Agent's action set is determined by benchmark and flags

2. **Lazy Planning**
   - When task is set, planning is triggered
   - For environment-based tasks: environment is temporarily created to extract goal + initial state context
   - For natural language tasks: uses task string directly

3. **Context Extraction**
   - Goal text from task description or environment
   - Structured state representation from initial environment
   - Detailed context for planning (e.g., available resources, current state)
   - Available actions from agent's action set

4. **Plan Generation**
   - `SimplifiedRDDPlanner` initialized with: `llm`, `RDDConfig(max_depth=2, max_nodes=20)`, `action_set`
   - Calls `planner.plan(task, state, axtree, html)` which returns dict with `plan` text and `graph` structure
   - BFS phase: splits tasks level-by-level using `split_prompt`, stops at max_depth/max_nodes/unit problems
   - DFS phase: solves tasks bottom-up, substituting `{task_id}` with solutions, using `unit_prompt` and `merge_prompt`

5. **Plan Refinement** (Optional, enabled by default)
   - `PlanRefiner` initialized with `llm` and `RefinerConfig(max_iterations=2)`
   - Calls `refiner.refine(plan, task)` which iteratively improves the plan
   - Returns dict with `refined_plan` and `improvement_notes`

6. **Execution**
   - Plan stored in `agent.plan` field
   - `GenericAgent` includes plan in prompts during execution
   - Agent references plan while taking actions

### Configuration

Plans are controlled via flags and arguments:

```python
RDDAgentArgs(
    task="task_identifier_or_description",  # Task to plan for
    use_axtree=True,           # Include structured state representation
    use_html=True,             # Include detailed context
    use_plan_refiner=True,     # Enable refinement
    refiner_iterations=2,      # Refinement passes
    flags=GenericPromptFlags(
        use_plan=True,         # Must be True for planning
        action=ActionFlags(...) # Defines available actions
    )
)
```

### Design Principles

1. **Minimal Invasiveness**: Builds on existing `GenericAgent` via composition, not modification
2. **Lazy Evaluation**: Planning happens when task is set, not at agent construction
3. **Flexibility**: Works with environment-based tasks or natural language descriptions
4. **Modularity**: Planning, refinement, and execution are separate concerns
5. **Compatibility**: Leverages AgentLab's existing LLM infrastructure and action sets

---

## 4. Implications for Ipazia

### What We've Learned

**1. Hierarchical Planning is Feasible**
- LLMs can effectively decompose complex tasks into subtasks
- Two-phase BFS/DFS approach produces coherent plans
- Dependency tracking enables multi-step reasoning

**2. Context Matters**
- Providing structured state representation improves plan quality
- Action set awareness makes plans more realistic
- Initial state context helps ground planning in reality

**3. Refinement Adds Value**
- LLMs sometimes generate redundant subtasks
- Post-processing can improve plan efficiency
- Iterative refinement converges to cleaner plans

**4. Integration Patterns**
- Lazy planning enables proper experiment setup
- Separation of planning and execution is clean
- GenericAgent's plan field provides natural extension point

### Business Value for Ipazia

**What We've Proven:**
- **Technical Feasibility**: RDD planning works - we can generate hierarchical plans with dependencies
- **Integration is Clean**: Minimal changes to existing AgentLab codebase
- **Plans are Readable**: Human-interpretable hierarchical structure with explicit dependencies

**What Needs Validation:**

**1. Does Planning Improve Success Rates?**
- Hypothesis: Strategic planning reduces trial-and-error, improving task completion
- Need: Benchmark evaluation comparing RDD agents vs. baseline agents
- Metric: Success rate on 50+ diverse tasks

**2. Does Planning Reduce Costs?**
- Hypothesis: Upfront planning cost is offset by more efficient execution
- Need: Token usage analysis (planning + execution vs. execution-only)
- Metric: Total tokens per task, cost per successful completion

**3. Is Debugging Actually Easier?**
- Hypothesis: Plan structure helps identify failure points
- Need: Failure analysis on tasks where agent fails
- Metric: Time to diagnose issue, clarity of failure mode

**4. Does It Scale to Complex Tasks?**
- Hypothesis: RDD handles multi-step tasks better than reactive agents
- Need: Evaluation on tasks with 10+ steps, multiple dependencies
- Metric: Success rate vs. task complexity

**If Validated, Business Impact:**
- Higher success rates → more deployable agents → more revenue
- Lower costs → better margins
- Better interpretability → easier maintenance → lower support costs
- Technical differentiation → competitive advantage

### Vision for General Settings

The RDD approach is domain-agnostic and could be applied beyond the current implementation:

**Different Task Domains:**
- Data analysis tasks (collect data → process → analyze → report)
- Software development tasks (understand requirements → design → implement → test)
- Customer support workflows (gather info → diagnose → resolve → follow-up)
- Research tasks (formulate question → search literature → synthesize → conclude)

**Key Adaptation Requirements:**
- Domain-specific prompt templates (split/unit/merge prompts tailored to domain)
- Appropriate context extraction (what "state" means varies by domain)
- Domain-relevant action sets (different capabilities for different domains)

**General Pattern:**
1. Task description + initial context → Planning phase (BFS decomposition)
2. Hierarchical plan with dependencies → Execution phase (follow plan)
3. Execution results → Success/failure analysis

The core BFS/DFS algorithm remains the same; only the prompts and context change per domain.

### Risks & Challenges

**Technical Challenges**
- Planning quality depends on LLM capabilities
- Stale plans if environment state changes significantly during execution
- Overhead of planning step (latency + computational cost)
- Dependency resolution can fail with complex graphs

**Business Challenges**
- Need to demonstrate ROI on real customer tasks
- Customers may prefer fast first-action over planning delay
- Integration complexity with existing agent infrastructure
- Requires evaluation framework to measure improvements

---

## Summary

The RDD Planner demonstrates that **hierarchical task decomposition is a viable approach for improving AI agent performance across diverse domains**. By separating strategic planning from tactical execution, we can build more reliable, efficient, and interpretable agents.

For Ipazia, this represents an opportunity to explore whether hierarchical planning can:
- Improve agent performance on complex multi-step tasks
- Provide better interpretability through explicit plan structures
- Enable more systematic debugging and improvement of agent behavior

The integration into AgentLab proves the concept is technically feasible. The next step is rigorous evaluation to quantify the benefits and identify the task types where planning provides the most value.
