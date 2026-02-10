"""
Standalone Recursive Task Decomposition (RDD) Planner Example

This module implements recursive task decomposition to break down
complex web navigation tasks into manageable subtasks. It's designed to work
with any web agent framework, including AgentLab.

Usage:
    python rdd_planner_example.py --task "Order a shirt on amazon.com" --state "On homepage"
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass
class RDDConfig:
    """Configuration for Recursive Task Decomposition"""
    max_depth: int = 3
    max_nodes: int = 20
    unit_statement: str = "this is a unit problem"
    prompt_dir: Path = Path(__file__).parent / "prompts"


class SimplifiedRDDPlanner:
    """
    Simplified Recursive Dependency Decomposition planner for web tasks.
    
    This is a streamlined version of the RecursivePrompting class,
    focused on web agent task planning.
    """
    
    def __init__(self, llm: Any, config: RDDConfig):
        """
        Initialize the RDD planner with an LLM and configuration.

        The `llm` should be a callable compatible with AgentLab chat models:
        it must accept a list of messages (or a `Discussion`) and return an
        object with a `"content"` field containing the assistant reply.
        """
        self.llm = llm  # Language model for generating decompositions and solutions
        self.config = config  # Configuration (max_depth, max_nodes, etc.)
        
        # Cache of all problems/tasks in the decomposition tree
        # Key: problem_id (e.g., "task_1"), Value: problem dict with description, subtasks, solution, etc.
        self.problems_cache = {}
        
        # Counter for generating unique task IDs
        self.id_counter = 0
        
        # Load prompt templates from files
        # These templates guide the LLM on how to split, solve, and merge tasks
        self.split_prompt = self._load_prompt("split_prompt.txt")  # For decomposing tasks
        self.unit_prompt = self._load_prompt("unit_prompt.txt")    # For solving atomic tasks
        self.merge_prompt = self._load_prompt("merge_prompt.txt")  # For combining solutions
    
    def _load_prompt(self, filename: str) -> str:
        """Load a prompt template from file"""
        path = self.config.prompt_dir / filename
        with open(path, 'r') as f:
            return f.read()
    
    def _next_id(self) -> str:
        """Generate next unique ID"""
        self.id_counter += 1
        return f"task_{self.id_counter}"
    
    def _format_axtree_context(self, max_chars: int = 3000) -> str:
        """
        Format the accessibility tree for inclusion in prompts.
        
        Args:
            max_chars: Maximum characters to include (truncate if longer)
        
        Returns:
            Formatted AXTree context string, or empty string if no AXTree
        """
        if not hasattr(self, 'axtree') or self.axtree is None:
            return ""
        
        axtree = self.axtree
        
        # Truncate if too long
        if len(axtree) > max_chars:
            truncated = axtree[:max_chars]
            # Try to truncate at a line boundary
            last_newline = truncated.rfind('\n')
            if last_newline > max_chars * 0.8:  # If we can save at least 20%
                truncated = truncated[:last_newline]
            
            return f"""Page Structure (Accessibility Tree):
```
{truncated}
... (truncated, showing first {len(truncated)} of {len(axtree)} characters)
```
"""
        else:
            return f"""Page Structure (Accessibility Tree):
```
{axtree}
```
"""
    
    def _format_html_context(self, max_chars: int = 5000) -> str:
        """
        Format the HTML content for inclusion in prompts.
        
        Args:
            max_chars: Maximum characters to include (truncate if longer)
        
        Returns:
            Formatted HTML context string, or empty string if no HTML
        """
        if not hasattr(self, 'html') or self.html is None:
            return ""
        
        html = self.html
        
        # Truncate if too long
        if len(html) > max_chars:
            truncated = html[:max_chars]
            # Try to truncate at a tag boundary
            last_tag_close = truncated.rfind('>')
            if last_tag_close > max_chars * 0.8:  # If we can save at least 20%
                truncated = truncated[:last_tag_close + 1]
            
            return f"""Page HTML:
```html
{truncated}
... (truncated, showing first {len(truncated)} of {len(html)} characters)
```
"""
        else:
            return f"""Page HTML:
```html
{html}
```
"""
    
    def plan(self, task: str, state: str, axtree: str = None, html: str = None) -> dict:
        """
        Generate a hierarchical plan for the given task using BFS with dependencies.
        
        Args:
            task: The task description (e.g., "Order a pizza on amazon.com")
            state: Current state description (e.g., "On homepage, not logged in")
            axtree: Optional accessibility tree (flattened) for page context
            html: Optional HTML content for page context
        
        Returns:
            dict: Hierarchical plan structure with tasks and subtasks
        """
        print(f"\nPlanning for task: {task}")
        print(f"Current state: {state}")
        if axtree:
            print(f"AXTree provided: {len(axtree)} characters")
        else:
            print("No AXTree provided")
        if html:
            print(f"HTML provided: {len(html)} characters\n")
        else:
            print("No HTML provided\n")
        
        # Reset state
        self.problems_cache = {}
        self.id_counter = 0
        
        # Store axtree and html for use in prompts
        self.axtree = axtree
        self.html = html
        
        # Create root problem
        root_id = self._next_id()
        root_problem = {
            "id": root_id,
            "description": task,
            "state": state,
            "depth": 0,
            "subtasks": [],
            "solution": None,
            "is_unit": False
        }
        self.problems_cache[root_id] = root_problem
        
        # Decompose using BFS with dependencies
        self._decompose(root_id, state)
        
        # Generate the plan
        plan = self._generate_plan_text(root_id)
        
        print("Plan:")
        print(plan)

        return {
            "task": task,
            "state": state,
            "axtree_provided": axtree is not None,
            "html_provided": html is not None,
            "plan": plan,
            "graph": self._export_graph(root_id)
        }
    
    def _decompose(self, problem_id: str, state: str):
        """
        PHASE 1: Decompose using BFS (Breadth-First Search)
        
        This method splits tasks level-by-level:
        - Level 0: Root task
        - Level 1: Direct subtasks of root
        - Level 2: Subtasks of level 1 tasks
        - etc.
        
        After all tasks are split, PHASE 2 solves them bottom-up using DFS.
        
        Args:
            problem_id: ID of the root problem to decompose
            state: Current state description (e.g., "On homepage, logged in")
        """
        from queue import Queue
        
        # Phase 1: BFS decomposition
        
        # ============================================================================
        # PHASE 1: BFS DECOMPOSITION (Split tasks level-by-level)
        # ============================================================================
        
        # Queue for BFS: processes tasks in order they were added (FIFO)
        # This ensures we split all tasks at depth N before moving to depth N+1
        to_split = Queue()
        to_split.put(problem_id)  # Start with root task
        
        # BFS Loop: Process each task in the queue
        while not to_split.empty():
            # Get next task from queue (FIFO order)
            current_id = to_split.get()
            current = self.problems_cache[current_id]
            depth = current["depth"]  # How deep in the tree (0 = root)
            
            # ========================================================================
            # STOPPING CONDITIONS: When to stop decomposing
            # ========================================================================
            
            # Stop 1: Reached maximum depth (prevent infinite recursion)
            if depth >= self.config.max_depth:
                # Max depth reached
                current["is_unit"] = True  # Mark as unit problem (solve directly)
                continue  # Skip to next task in queue
            
            # Stop 2: Too many nodes created (prevent memory issues)
            if len(self.problems_cache) >= self.config.max_nodes:
                # Max nodes reached
                current["is_unit"] = True  # Mark as unit problem
                continue
            
            # ========================================================================
            # ASK LLM TO SPLIT THE TASK
            # ========================================================================
            
            # Decomposing task into subtasks
            
            # Build prompt from template, filling in variables
            split_prompt_text = self.split_prompt.format(
                problem=current["description"],      # The task to split
                state=state,                         # Current state context
                unit_statement=self.config.unit_statement,  # What to output for unit problems
                axtree_context=self._format_axtree_context(),  # Page structure
                html_context=self._format_html_context(),      # Page HTML
                ancestors="",                        # Parent task context (empty for now)
                examples=""                          # Few-shot examples (empty for now)
            )
            
            # Call LLM to get decomposition
            # OpenRouter/AgentLab LLMs use __call__() with message format
            messages = [{"role": "user", "content": split_prompt_text}]
            response_obj = self.llm(messages)
            response = response_obj["content"]  # Extract text from AIMessage
            
            # ========================================================================
            # CHECK IF TASK IS SIMPLE ENOUGH (UNIT PROBLEM)
            # ========================================================================
            
            # If LLM says "this is a unit problem", don't split further
            if self.config.unit_statement.lower() in response.lower():
                # Task identified as unit problem
                current["is_unit"] = True  # Mark as atomic task
                continue  # Skip to next task
            
            # ========================================================================
            # PARSE SUBTASKS FROM LLM RESPONSE
            # ========================================================================
            
            # Extract subtasks from LLM's bullet-point or numbered list
            subtasks = self._parse_subtasks(response, current_id, depth + 1)
            
            # If no subtasks found, treat as unit problem
            if not subtasks:
                # No subtasks found, treating as unit problem
                current["is_unit"] = True
                continue
            
            # Store subtask IDs in current task
            current["subtasks"] = [st["id"] for st in subtasks]
            # Task decomposed into subtasks
            
            # ========================================================================
            # ADD SUBTASKS TO BFS QUEUE (for next level of decomposition)
            # ========================================================================
            
            # Each subtask will be processed in turn (breadth-first)
            for subtask in subtasks:
                to_split.put(subtask["id"])  # Add to end of queue
        
        # ============================================================================
        # PHASE 2: DFS SOLVING (Solve dependencies bottom-up)
        # ============================================================================
        
        # Now that all tasks are split, solve them from leaves to root
        # Uses DFS to ensure child tasks (dependencies) are solved before parents
        # Phase 2: DFS solving
        self._solve_with_dependencies(problem_id, set())
    
    def _solve_with_dependencies(self, problem_id: str, visited: set):
        """
        PHASE 2: Solve tasks using DFS (Depth-First Search) with explicit dependencies.
        
        This ensures dependencies are solved before their parents:
        1. Recursively solve all explicit dependencies first (from dependency_ids)
        2. Recursively solve all child tasks (from subtasks)
        3. Then solve the current task (using dependency and child solutions)
        
        Example:
            Task A has subtasks [B, C]
            Task B has subtasks [D, E]
            Task C depends on B (explicit dependency)
            
            Solving order: D → E → B → C → A
            (Dependencies first, then leaves to root)
        
        Args:
            problem_id: ID of the problem to solve
            visited: Set of already-solved problem IDs (prevents cycles)
        """
        # Prevent solving the same task twice (cycle detection)
        if problem_id in visited:
            return
        
        visited.add(problem_id)  # Mark as being processed
        problem = self.problems_cache[problem_id]
        
        # ========================================================================
        # STEP 1: Recursively solve all explicit dependencies first
        # ========================================================================
        
        # Solve tasks that this task explicitly depends on (from {dep_id} references)
        if problem.get("dependency_ids"):
            for dep_id in problem["dependency_ids"]:
                if dep_id not in visited:
                    self._solve_with_dependencies(dep_id, visited)  # Recursive call
        
        # ========================================================================
        # STEP 2: Recursively solve all subtasks (child dependencies)
        # ========================================================================
        
        # DFS: Go deep into subtasks before solving current task
        for subtask_id in problem["subtasks"]:
            if subtask_id not in visited:
                self._solve_with_dependencies(subtask_id, visited)  # Recursive call
        
        # ========================================================================
        # STEP 3: Now solve THIS task (all dependencies are solved)
        # ========================================================================
        
        if problem["is_unit"] or not problem["subtasks"]:
            # UNIT PROBLEM: No subtasks, solve directly with LLM
            self._solve_unit(problem_id, problem.get("state", ""))
        else:
            # COMPOSITE PROBLEM: Has subtasks, merge their solutions
            self._merge_solutions(problem_id)
    
    def _parse_subtasks(self, response: str, parent_id: str, depth: int) -> list[dict]:
        """
        Parse subtasks from LLM's text response.
        
        Expected format with identifiers and dependencies:
        - [id1] Do something
        - [id2] Do something else that depends on {id1}
        
        Or numbered:
        1. [task_a] First step
        2. [task_b] Second step using result from {task_a}
        
        Args:
            response: LLM's text response
            parent_id: ID of the parent task
            depth: Depth of these subtasks in the tree
        
        Returns:
            List of subtask dictionaries with dependencies
        """
        import re
        
        subtasks = []
        id_to_internal_id = {}  # Map from user-defined [id] to internal task_N
        
        # Parse each line looking for bullet points or numbered items
        for line in response.split('\n'):
            line = line.strip()
            if not line:
                continue  # Skip empty lines
            
            # Extract description from different formats
            desc = None
            
            # Format 1: Bullet points (- or •)
            if line.startswith('-') or line.startswith('•'):
                desc = line[1:].strip()  # Remove bullet and whitespace
            
            # Format 2: Numbered list (1. 2. 3.)
            elif line and line[0].isdigit() and '.' in line:
                desc = line.split('.', 1)[1].strip()  # Remove number and period
            
            else:
                continue  # Skip lines that don't match expected format
            
            # Clean up description: remove "Subtask N:" prefix if present
            if desc and desc.lower().startswith('subtask'):
                parts = desc.split(':', 1)
                if len(parts) > 1:
                    desc = parts[1].strip()
            
            if not desc:
                continue
            
            # Extract identifier [id] if present
            user_id = None
            id_match = re.match(r'^\[([^\]]+)\]\s*(.*)$', desc)
            if id_match:
                user_id = id_match.group(1)  # Extract identifier
                desc = id_match.group(2)     # Remove [id] from description
            
            # FALLBACK: If no [id] provided, auto-generate a semantic ID
            # This handles cases where LLM doesn't follow the format
            if not user_id:
                user_id = f"subtask_{len(subtasks) + 1}"
            
            # Extract dependencies {dep_id} from description
            # Only match simple identifiers (alphanumeric + underscore/hyphen)
            # This avoids matching Python dicts like {'key': 'value'}
            dependencies = re.findall(r'\{([a-zA-Z0-9_-]+)\}', desc)
            
            # Create subtask object and add to cache
            subtask_id = self._next_id()  # Generate unique internal ID
            subtask = {
                "id": subtask_id,
                "user_id": user_id,          # User-defined identifier (e.g., "task_a")
                "description": desc,
                "parent": parent_id,
                "depth": depth,
                "subtasks": [],              # Will be filled if this task is split
                "solution": None,            # Will be filled when solved
                "is_unit": False,            # Will be set to True if not split further
                "dependencies": dependencies # List of user_ids this task depends on
            }
            self.problems_cache[subtask_id] = subtask  # Add to global cache
            subtasks.append(subtask)
            
            # Map user-defined ID to internal ID for dependency resolution
            if user_id:
                id_to_internal_id[user_id] = subtask_id
        
        # Resolve dependencies: convert user_ids to internal task_ids
        for i, subtask in enumerate(subtasks):
            resolved_deps = []
            for dep_user_id in subtask["dependencies"]:
                if dep_user_id in id_to_internal_id:
                    # Direct match found
                    resolved_deps.append(id_to_internal_id[dep_user_id])
                else:
                    # FALLBACK 1: Try positional mapping for common patterns
                    # E.g., {Worker_1} might refer to the first subtask
                    positional_match = re.match(r'(?:Worker|Task|Step)_(\d+)', dep_user_id, re.IGNORECASE)
                    if positional_match:
                        pos = int(positional_match.group(1)) - 1  # Convert to 0-indexed
                        if 0 <= pos < len(subtasks):
                            resolved_deps.append(subtasks[pos]["id"])
                            print(f"Fallback (positional): Mapped {{{dep_user_id}}} to subtask at position {pos + 1}")
                            continue
                    
                    # FALLBACK 2: Try semantic matching
                    # E.g., {navigate} might refer to a task with "navigate" in its description
                    dep_lower = dep_user_id.lower()
                    for prev_idx in range(i):  # Only look at previous tasks (dependencies must come first)
                        prev_task = subtasks[prev_idx]
                        prev_desc = prev_task["description"].lower()
                        prev_user_id = (prev_task.get("user_id") or "").lower()
                        
                        # Check if dependency name appears in task description or user_id
                        if dep_lower in prev_desc or dep_lower in prev_user_id:
                            resolved_deps.append(prev_task["id"])
                            print(f"Fallback (semantic): Mapped {{{dep_user_id}}} to task '{prev_task['user_id']}' (contains '{dep_user_id}')")
                            break
                    else:
                        # Still not found - warn user
                        print(f"ERROR: Undefined dependency {{{dep_user_id}}} - no matching task found!")
                    # If still not found, skip it (LLM hallucinated)
            subtask["dependency_ids"] = resolved_deps  # Internal task IDs
        
        # Return all parsed subtasks (no explicit width limit)
        return subtasks
    
    def _solve_unit(self, problem_id: str, state: str):
        """
        Solve a unit (atomic) problem using the LLM.
        
        A unit problem is one that cannot be decomposed further.
        The LLM is asked to provide a direct solution.
        
        Before solving, substitute any dependency references {dep_id} with their solutions.
        
        Args:
            problem_id: ID of the unit problem to solve
            state: Current state description
        """
        problem = self.problems_cache[problem_id]
        
        # Substitute dependency placeholders {dep_id} with actual solutions
        description_with_deps = problem["description"]
        if problem.get("dependency_ids"):
            import re
            for dep_id in problem["dependency_ids"]:
                if dep_id in self.problems_cache:
                    dep_task = self.problems_cache[dep_id]
                    dep_user_id = dep_task.get("user_id", dep_id)
                    dep_solution = dep_task.get("solution", "[solution not available]")
                    # Replace {dep_id} with the actual solution using string replace
                    # (avoids regex escape errors from backslashes in solutions)
                    description_with_deps = description_with_deps.replace(
                        f'{{{dep_user_id}}}',
                        dep_solution
                    )
        
        # Build prompt for solving this unit problem
        unit_prompt_text = self.unit_prompt.format(
            problem=description_with_deps,  # Task description with dependencies resolved
            state=state,                     # Current state
            axtree_context=self._format_axtree_context(),  # Page structure
            html_context=self._format_html_context(),      # Page HTML
            ancestors="",                    # Parent context (empty for now)
            examples=""                      # Few-shot examples (empty for now)
        )
        
        # Get solution from LLM
        messages = [{"role": "user", "content": unit_prompt_text}]
        response_obj = self.llm(messages)
        problem["solution"] = response_obj["content"]  # Store solution
    
    def _merge_solutions(self, problem_id: str):
        """
        Merge solutions from subtasks to solve the parent task.
        
        This is called when a task has subtasks that have all been solved.
        The LLM combines the subtask solutions into a coherent solution for the parent.
        
        Args:
            problem_id: ID of the parent problem
        """
        problem = self.problems_cache[problem_id]
        
        # Build a formatted list of subtask descriptions and their solutions
        subsolutions = []
        for i, subtask_id in enumerate(problem["subtasks"], 1):
            subtask = self.problems_cache[subtask_id]
            subsolutions.append(
                f"{i}. {subtask['description']}\n   Solution: {subtask['solution']}"
            )
        
        subsolutions_text = "\n".join(subsolutions)
        
        # Build prompt asking LLM to merge the subtask solutions
        merge_prompt_text = self.merge_prompt.format(
            problem=problem["description"],    # Original task
            subsolutions=subsolutions_text,     # All subtask solutions
            ancestors="",                       # Parent context (empty for now)
            examples=""                         # Few-shot examples (empty for now)
        )
        
            # Get merged solution from LLM
        messages = [{"role": "user", "content": merge_prompt_text}]
        response_obj = self.llm(messages)
        problem["solution"] = response_obj["content"]  # Store merged solution
    
    def _generate_plan_text(self, problem_id: str, number_prefix: str = "", indent: int = 0) -> str:
        """
        Generate a readable plan text with hierarchical numbering and explicit dependencies.
        
        Args:
            problem_id: ID of the problem to generate plan for
            number_prefix: Current numbering prefix (e.g., "1", "1.1", "1.2.1")
            indent: Indentation level
        
        Returns:
            Formatted plan text with hierarchical numbering and dependency annotations
        """
        problem = self.problems_cache[problem_id]
        prefix = "  " * indent
        
        # Format: "1. Task description" or "1.1. Subtask description"
        task_line = f"{prefix}{number_prefix}. {problem['description']}" if number_prefix else f"{prefix}{problem['description']}"
        
        # Add dependency annotation if present
        if problem.get("dependency_ids"):
            # Find user_ids of dependencies for display
            dep_labels = []
            for dep_id in problem["dependency_ids"]:
                if dep_id in self.problems_cache:
                    dep_task = self.problems_cache[dep_id]
                    if dep_task.get("user_id"):
                        dep_labels.append(dep_task["user_id"])
                    else:
                        dep_labels.append(dep_id)  # Fallback to internal ID
            
            if dep_labels:
                task_line += f" [depends on: {', '.join(dep_labels)}]"
        
        plan_lines = [task_line]
        
        # Add subtasks with hierarchical numbering
        if problem["subtasks"]:
            for i, subtask_id in enumerate(problem["subtasks"], 1):
                # Build hierarchical number (e.g., "1.1", "1.2", "2.1.3")
                if number_prefix:
                    sub_number = f"{number_prefix}.{i}"
                else:
                    sub_number = str(i)
                
                plan_lines.append(
                    self._generate_plan_text(subtask_id, sub_number, indent + 1)
                )
        
        return "\n".join(plan_lines)
    
    def _export_graph(self, root_id: str) -> dict:
        """Export the decomposition graph as a dictionary"""
        return {
            "root_id": root_id,
            "nodes": {
                pid: {
                    "description": p["description"],
                    "subtasks": p["subtasks"],
                    "solution": p["solution"],
                    "is_unit": p["is_unit"],
                    "depth": p["depth"]
                }
                for pid, p in self.problems_cache.items()
            }
        }


def main():
    """CLI entrypoint kept for backwards compatibility."""
    raise RuntimeError(
        "Use `SimplifiedRDDPlanner` with a real chat model, "
        "for example via `RDDGenericAgent` in `rdd_agentlab_integration.py`."
    )


if __name__ == "__main__":
    main()
