from bgym import AbstractActionSet

from agentlab.agents.structured_agent.hpa_prompt import HPAPromptFlags

from .andor_tree import Node, NodeState, NodeStatus, NodeType
from .plan_telemetry import PlanTelemetry
from .stack import Stack
from .tree_update_engine import TreeUpdateEngine


class HPA:
    def __init__(
        self,
        chat_llm,
        action_set: AbstractActionSet,
        flags: HPAPromptFlags,
        budget: int = 1000,
        max_depth: int = 3,
    ):
        self.chat_llm = chat_llm
        self.action_set = action_set
        self.flags = flags
        self.budget = budget
        self.max_depth = max_depth
        self.pending_node: Node | None = None
        self._counter: int = 0
        self.retries: int = 0
        self.max_retries: int = 3
        self.previous_attempt_summaries: list[str] = []

        self.tree_update_engine = TreeUpdateEngine(
            chat_llm=self.chat_llm, action_set=self.action_set, flags=self.flags
        )

    def get_plan(self) -> tuple[list[str], list[str]]:
        return self.stack.plan

    def set_goal(self, obs_first: dict):
        self.goal = obs_first["goal"]
        self.stack = Stack(self.goal)
        self.telemetry = PlanTelemetry(self.stack)
        self.retries = 0

    def get_action_node(self, expansion_function, recovery_function) -> Node | None:
        self.telemetry.reset_for_action()
        while True:
            while self.stack.items:
                node, state = self.stack.items.pop()
                
                # Skip recovering root node if it has been executed more than 1 time, go for the retry
                if node.depth == 0 and node.execution_count >= 1:
                    break

                if node.status == NodeStatus.NOT_RECOVERABLE:
                    self.stack.propagate_failure(node)
                    continue

                if node.status == NodeStatus.DELETED:
                    continue

                if state == NodeState.ENTERING:
                    if node.type == NodeType.UNKNOWN:
                        tree_context_before = node.to_tree_context_entry()

                        def on_unknown_expanded(expanded_node: Node):
                            self.telemetry.record_expansion(
                                node=expanded_node,
                                retry=self.retries,
                                tree_context_before=tree_context_before,
                            )

                    else:

                        def on_unknown_expanded(_):
                            pass

                    self.pending_node = self.stack.process_node_entering(
                        node=node,
                        expansion_function=expansion_function,
                        recovery_function=recovery_function,
                        get_tree_context=self.get_tree_context,
                        on_unknown_expanded=on_unknown_expanded,
                    )
                    if self.pending_node.type == NodeType.ACTION:
                        return self.pending_node

                elif state == NodeState.EXITING:
                    self.stack.process_node_exiting(node)

                elif state == NodeState.FAILED:
                    self.stack.process_node_failed(node)

                self._counter += 1
                if self._counter >= self.budget:
                    break

            if self.stack.items or self.stack.completed_nodes:
                self.previous_attempt_summaries.append(self.stack.summary)
            self.retries += 1
            # Create a new root node with id based on the retries
            self.stack.clean(retries=self.retries)

    def complete_action_node(
        self,
        model_name: str,
        constraints: str,
        progress: str,
        suggestion: str,
        obs_history: list[dict],
        verification_function=None,
    ) -> dict:
        action_error = obs_history[-1].get("last_action_error", "")

        if action_error:
            self.pending_node.status = NodeStatus.NOT_RECOVERABLE
            self.pending_node.action_error = action_error
        elif verification_function is not None:
            is_success, explanation = verification_function(self.pending_node)
            if is_success:
                self.pending_node.status = NodeStatus.SUCCESS
            else:
                self.pending_node.status = NodeStatus.NOT_RECOVERABLE
                self.pending_node.action_error = explanation
        else:
            self.pending_node.status = NodeStatus.SUCCESS
        self.telemetry.set_action_outcome(self.pending_node.status)

        print(f"{60*'='}\nAction Verification\n{60*'='}\n")
        print(f"Action: {self.pending_node.description}")
        print(f"Previous Action: {self.pending_node.action}")
        print(f"Error: {self.pending_node.action_error}")
        print(f"Is Success: {True if self.pending_node.status == NodeStatus.SUCCESS else False}")
        print(f"{60*'='}\n")
        
        result = self.tree_update_engine.apply(
            model_name=model_name,
            task_description=self.goal,
            constraints=constraints,
            progress=progress,
            suggestion=suggestion,
            obs_history=obs_history,
            pending_node=self.pending_node,
            stack=self.stack.items,
        )
        self.telemetry.set_tree_update_result(result)
        return result

    def get_telemetry(self, step_index: int | None = None) -> dict:
        return self.telemetry.build_plan_info(
            step_index=step_index,
            pending_node=self.pending_node,
            max_depth=self.max_depth,
            budget=self.budget,
            max_retries=self.max_retries,
            search_counter=self._counter,
        )

    def get_tree_context(self, expanding_node: Node, show_deleted: bool = False) -> str:
        lines = [e.format() for e in expanding_node.to_tree_context_entry(show_deleted)]
        if self.previous_attempt_summaries:
            lines.append("\n## Previous Failed Attempts:")
            for i, summary in enumerate(self.previous_attempt_summaries):
                lines.insert(0, f"\n### Attempt {i + 1}:\n{summary}")

        return "\n".join(lines)
