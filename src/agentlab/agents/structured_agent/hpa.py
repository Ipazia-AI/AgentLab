import logging
from typing import List, Tuple
from .andor_tree import NodeType, NodeStatus, NodeState, Node

class HPA:
    def __init__(
        self,
        chat_llm,
        action_set,
        budget: int = 1000,
        max_revision_count: int = 3,
    ):
        self.chat_llm = chat_llm
        self.action_set = action_set
        self.budget = budget # remove
        self.max_revision_count = max_revision_count
        self.stack: List[Tuple[Node, NodeState]] = []
        self.goal: str | None = None
        self._counter = 0

    def reset(self, root_node: Node, goal: str | None = None):
        self.stack = [(root_node, NodeState.ENTERING)]
        self.goal = goal
        self._counter = 0

    def run_until_action(self) -> Node | None:
        while self.stack:
            node, state = self.stack.pop()

            if node.status == NodeStatus.PRUNED:
                self._propagate_failure(node)
                continue

            if node.status == NodeStatus.DELETED:
                continue

            if state == NodeState.ENTERING:
                action_node = self._process_node_entering(node, self.goal)
                if action_node is not None:
                    return action_node

            elif state == NodeState.EXITING:
                self._process_node_exiting(node)

            elif state == NodeState.FAILED:
                self._process_node_failed(node)

            self._counter += 1
            if self._counter >= self.budget:
                break

        return None

    def finalize_action(self, node: Node, obs: dict, success: bool):
        if success:
            self._global_tree_update()
            self._update_observations(node, obs)
            node.status = NodeStatus.SUCCESS
        else:
            node.status = NodeStatus.FAILED

    # Algo 2 from the HPA paper
    def _process_node_entering(self, node: Node, goal: str | None) -> Node | None:
        if node.parent and node.parent.type == NodeType.OR:
            self._rollback_context(node.parent)

        self._set_context(node)
        node.execution_count += 1

        if node.status == NodeStatus.UNVISITED:
            node.type = self._expand_node(node, goal)
            #self.stack.append((node, NodeState.EXITING))

        if node.type == NodeType.ACTION:
            self.stack.append((node, NodeState.EXITING))
            return node

        if node.type == NodeType.AND:
            if self._is_successful(node):
                return None
            if self._has_valid_child(node):
                for child in reversed(node.children):
                    if child.status not in self._closed_statuses():
                        self.stack.append((child, NodeState.ENTERING))
            else:
                node.status = NodeStatus.FAILED

        if node.type == NodeType.OR:
            if self._is_successful(node):
                return None
            if self._is_valid_or(node):
                child = self._select_promising_child(node)
                self.stack.append((child, NodeState.ENTERING))
            else:
                node.status = NodeStatus.FAILED
        return None

    # Algo 3 from the HPA paper
    def _process_node_exiting(self, node: Node):
        if node.type == NodeType.ACTION:
            if node.status in {NodeStatus.FAILED, NodeStatus.PRUNED}:
                self.stack.append((node, NodeState.FAILED))
            return
        else:
            node.status = NodeStatus.SUCCESS

        if node.type == NodeType.AND:
            if self._is_successful_and(node):
                if self._check_and_complete(node):
                    node.status = NodeStatus.SUCCESS
                    return
            
            node.status = NodeStatus.FAILED
            self.stack.append((node, NodeState.FAILED))

        elif node.type == NodeType.OR:
            if self._is_successful_or(node):
                node.status = NodeStatus.SUCCESS
            else:
                node.status = NodeStatus.FAILED
                self.stack.append((node, NodeState.FAILED))

    # Algo 4 from the HPA paper
    def _process_node_failed(self, node: Node):
        if node.type == NodeType.ACTION:
            node.status = NodeStatus.PRUNED
            self._propagate_failure(node)
            return

        elif node.type == NodeType.AND:
            if self._has_success(node) and self._check_and_complete(node):
                node.status = NodeStatus.SUCCESS
                return

            if not self._is_valid_and(node):
                if node.revision_count < self.max_revision_count:
                    revised = self._revise_and(node)
                    self._synchronize_stack()
                    if revised:
                        node.status = NodeStatus.VISITED
                        self.stack.append((node, NodeState.ENTERING))
                else:
                    self._prune(node)
                    self._synchronize_stack()
                    self._propagate_failure(node)

        elif node.type == NodeType.OR:
            if self._is_valid_or(node):
                node.status = NodeStatus.VISITED
                self.stack.append((node, NodeState.ENTERING))
                return

            if node.revision_count < self.max_revision_count:
                revised = self._revise_or(node)
                self._synchronize_stack()
                if revised:
                    node.status = NodeStatus.VISITED
                    self.stack.append((node, NodeState.ENTERING))
            else:
                self._prune(node)
                self._synchronize_stack()
                self._propagate_failure(node)

    def _expand_node(self, node: Node, goal: str | None) -> NodeType:
        return NodeType.ACTION

    def _select_promising_child(self, node: Node) -> Node:
        # Need to implement child selection logic
        return node.children[0]

    def _rollback_context(self, node: Node): 
        pass

    def _set_context(self, node: Node): 
        pass

    def _global_tree_update(self): 
        pass

    def _update_observations(self, node: Node, obs: dict): 
        pass

    def _propagate_failure(self, node: Node): 
        pass

    def _synchronize_stack(self): 
        pass

    def _revise_and(self, node: Node) -> bool:
        node.revision_count += 1
        return False

    def _revise_or(self, node: Node) -> bool:
        node.revision_count += 1
        return False

    def _prune(self, node: Node):
        node.status = NodeStatus.PRUNED

    def _closed_statuses(self):
        return {NodeStatus.SUCCESS, NodeStatus.FAILED, NodeStatus.PRUNED}

    def _is_successful(self, node: Node) -> bool: 
        return node.status == NodeStatus.SUCCESS
    
    def _is_successful_and(self, node: Node) -> bool: 
        valid_children = self._valid_children(node)
        if not valid_children:
            return False
        return all(c.status == NodeStatus.SUCCESS for c in valid_children)
    
    def _is_successful_or(self, node: Node) -> bool: 
        return any(c.status == NodeStatus.SUCCESS for c in node.children)

    def _has_valid_child(self, node: Node) -> bool: 
        return any(c.status not in {NodeStatus.PRUNED, NodeStatus.DELETED} for c in node.children)
    
    def _has_success(self, node: Node) -> bool: 
        if node.type == NodeType.AND:
            return all(c.status == NodeStatus.SUCCESS for c in node.children)
        if node.type == NodeType.OR:
            return any(c.status == NodeStatus.SUCCESS for c in node.children)
        return False
    
    def _is_valid_and(self, node: Node) -> bool: 
        return all(c.status not in {NodeStatus.PRUNED, NodeStatus.DELETED} for c in node.children)
    
    def _is_valid_or(self, node: Node) -> bool: 
        return any(c.status not in {NodeStatus.PRUNED, NodeStatus.DELETED} for c in node.children)
    
    def _check_and_complete(self, node: Node) -> bool: 
        valid_children = self._valid_children(node)
        if not valid_children:
            return False
        return all(c.status == NodeStatus.SUCCESS for c in valid_children)

    def _valid_children(self, node: Node) -> list[Node]:
        return [c for c in node.children if c.status not in {NodeStatus.PRUNED, NodeStatus.DELETED}]

    def _is_root(self, node: Node) -> bool:
        return node.parent is None


def main():
    root = Node(type=NodeType.UNKNOWN, text="Solve task")
    agent = HPA(chat_llm=None, action_set=None)
    agent.reset(root_node=root, goal="Complete the task")
    next_action = agent.run_until_action()
    if next_action is not None:
        logging.info("Next action: %s", next_action.text)


if __name__ == "__main__":
    main()
