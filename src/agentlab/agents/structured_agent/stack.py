from .andor_tree import Node, NodeState, NodeStatus, NodeType


class Stack:
    items: list[tuple[Node, NodeState]]
    completed_nodes: list[Node]

    def __init__(self, goal: str) -> None:
        self.goal = goal
        self.root = Node(type=NodeType.UNKNOWN, description=goal)
        self.items: list[tuple[Node, NodeState]] = [
            (self.root, NodeState.ENTERING)
        ]
        self.completed_nodes: list[Node] = []

    def clean(self) -> None:
        self.root = Node(type=NodeType.UNKNOWN, description=self.goal)
        self.items = [(self.root, NodeState.ENTERING)]

    def process_node_entering(
        self,
        node: Node,
        expansion_function,
        recovery_function,
        get_tree_context,
        on_unknown_expanded,
    ) -> Node:
        node.execution_count += 1

        if node.status == NodeStatus.RECOVERABLE:
            node.revision_count += 1
            if node.revision_count > node.max_revision_count:
                node.status = NodeStatus.NOT_RECOVERABLE
                self.propagate_failure(node)
                return node
            tree_context = get_tree_context(node)
            print(f"\n{'='*60}\nRecovering node {node.id}\n{'='*60}\n{tree_context}\n{'='*60}\n")
            recovery_function(node, tree_context)
            node.discard_useless_children()
            node.status = NodeStatus.VISITED

        elif node.type == NodeType.UNKNOWN:
            tree_context = get_tree_context(node)
            print(f"\n{'='*60}\nExpanding node {node.id}\n{'='*60}\n{tree_context}\n{'='*60}\n")
            expansion_function(node, tree_context)
            node.status = NodeStatus.VISITED
            on_unknown_expanded(node)

        if node.type == NodeType.ACTION:
            self.items.append((node, NodeState.EXITING))
            return node

        elif node.type in {NodeType.AND, NodeType.OR}:
            if node.children and node.all_deleted_children:
                node.status = NodeStatus.DELETED
                return node

            if node.type == NodeType.AND:
                if node.successful_children_and:
                    return node
                self.items.append((node, NodeState.EXITING))
                if node.valid_children_and:
                    for child in reversed(node.children):
                        if not child.closed:
                            self.items.append((child, NodeState.ENTERING))
                else:
                    node.status = NodeStatus.RECOVERABLE

            else:  # OR
                if node.successful_children_or:
                    return node
                self.items.append((node, NodeState.EXITING))
                if node.valid_children_or:
                    child = self.select_promising_child(node)
                    self.items.append((child, NodeState.ENTERING))
                else:
                    node.status = NodeStatus.RECOVERABLE

        return node

    def process_node_exiting(self, node: Node) -> None:
        if node.type == NodeType.ACTION:
            self.completed_nodes.append(node)
            if node.status in {NodeStatus.RECOVERABLE, NodeStatus.NOT_RECOVERABLE}:
                self.items.append((node, NodeState.FAILED))

        elif node.type in {NodeType.AND, NodeType.OR}:
            is_success = False
            if node.type == NodeType.AND:
                is_success = node.successful_children_and
            elif node.type == NodeType.OR:
                is_success = node.successful_children_or
                
            if is_success:
                node.status = NodeStatus.SUCCESS
                return
            
            if node.status != NodeStatus.RECOVERABLE:
                node.status = NodeStatus.RECOVERABLE
            self.items.append((node, NodeState.FAILED))

    def process_node_failed(self, node: Node) -> None:
        if node.type == NodeType.ACTION:
            node.status = NodeStatus.NOT_RECOVERABLE
            self.propagate_failure(node)
            return

        if node.status != NodeStatus.RECOVERABLE:
            node.status = NodeStatus.RECOVERABLE
        self.items.append((node, NodeState.ENTERING))

    def select_promising_child(self, node: Node) -> Node:
        valid_children = node.valid_children
        return valid_children[0]

    def propagate_failure(self, node: Node) -> None:
        parent = node.parent
        if parent is None:
            return

        def mark_deleted_subtree(n: Node, deleted_ids: set):
            if n.is_preserved_status:
                return
            if n.status != NodeStatus.DELETED:
                n.status = NodeStatus.DELETED
            deleted_ids.add(n.id)
            for c in n.children:
                mark_deleted_subtree(c, deleted_ids)

        deleted_ids: set = set()

        if parent.type == NodeType.AND:
            for sibling in parent.children:
                mark_deleted_subtree(sibling, deleted_ids)

        elif parent.type == NodeType.OR:
            mark_deleted_subtree(node, deleted_ids)

        else:
            raise ValueError(f"Unexpected parent type: {parent.type}")

        if deleted_ids:
            self.items = [(n, st) for (n, st) in self.items if n.id not in deleted_ids]

        if parent.status not in {NodeStatus.NOT_RECOVERABLE, NodeStatus.DELETED}:
            parent.status = NodeStatus.RECOVERABLE

    def check_and_complete(self, node: Node) -> bool:
        valid_children = node.valid_children
        if not valid_children:
            return False
        return all(c.status == NodeStatus.SUCCESS for c in valid_children)

    @property
    def summary(self) -> str:
        #root = self.items[0][0] if self.items else None
        if self.root is None:
            return

        root_context = self.root.to_tree_context_entry()
        return "\n".join(entry.format(disable_marker=True) for entry in root_context)

    @property
    def plan(self) -> tuple[list[str], list[str]]:
        completed_plan = [node.prompt_description for node in self.completed_nodes]
        pending_plan = [
            node.prompt_description
            for node, state in reversed(self.items)
            if node.id != "0"
            and state != NodeState.EXITING
            and (node.status == NodeStatus.UNVISITED or node.type == NodeType.UNKNOWN)
        ]
        return completed_plan, pending_plan
