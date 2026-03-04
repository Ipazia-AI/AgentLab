from dataclasses import dataclass

from agentlab.agents.structured_agent.stack import Stack

from .andor_tree import Node, NodeState, NodeStatus, NodeType


@dataclass
class TreeContextEntry:
    depth: int
    node_id: str
    description: str
    status: str
    type_label: str
    marker_expand: bool = False
    action_error: str | None = None

    def format(self) -> str:
        indent = "  " * self.depth
        status_prefix = ""
        if self.status == "FAIL":
            err = f": {self.action_error}" if self.action_error else ""
            status_prefix = f"[FAIL{err}] "
        elif self.status:
            status_prefix = f"[{self.status}] "
        type_label = f" ({self.type_label})" if self.type_label else ""
        marker = "  ← EXPAND THIS NODE" if self.marker_expand else ""
        return f"{indent}{status_prefix}{self.node_id}{type_label}: {self.description}{marker}"


@dataclass
class TreeExpansionTrace:
    retry: int
    expanded_node: Node
    children_after: list[Node]
    tree_context_before: list[TreeContextEntry]
    tree_context_after: list[TreeContextEntry]


class PlanTelemetry:
    def __init__(self, stack: Stack) -> None:
        self.stack = stack
        self.last_tree_evolution: list[TreeExpansionTrace] = []
        self.last_tree_update_result: dict | None = None
        self.last_pending_outcome: str | None = None

    def reset_for_action(self) -> None:
        self.last_tree_evolution = []

    def set_action_outcome(self, status: NodeStatus) -> None:
        self.last_pending_outcome = status.name

    def set_tree_update_result(self, result: dict | None) -> None:
        self.last_tree_update_result = result

    def record_expansion(
        self, node: Node, retry: int, tree_context_before: list[TreeContextEntry]
    ) -> None:
        tree_context_after = node.to_tree_context_entry()
        self.last_tree_evolution.append(
            TreeExpansionTrace(
                retry=retry,
                expanded_node=node,
                children_after=list(node.children),
                tree_context_before=tree_context_before,
                tree_context_after=tree_context_after,
            )
        )

    def get_last_tree_evolution(self, pending_node: Node | None) -> dict:
        expansions = []
        for order, expansion in enumerate(self.last_tree_evolution, start=1):
            expanded_depth = expansion.expanded_node.depth
            valid_children = expansion.expanded_node.valid_children
            selected_child_id = None
            if expansion.expanded_node.type == NodeType.OR and valid_children:
                selected_child_id = valid_children[0].id
            expansions.append(
                {
                    "order": order,
                    "retry": expansion.retry,
                    "expanded_node_id": expansion.expanded_node.id,
                    "expanded_node_description": expansion.expanded_node.description,
                    "expanded_node_type_after": expansion.expanded_node.type.name,
                    "expanded_node_status_after": expansion.expanded_node.status.name,
                    "expanded_node_depth": expanded_depth,
                    "children_after": [child.to_dict() for child in expansion.children_after],
                    "n_children": len(expansion.children_after),
                    "n_valid_children": len(valid_children),
                    "n_pruned_children": len(
                        [
                            c
                            for c in expansion.expanded_node.children
                            if c.status == NodeStatus.PRUNED
                        ]
                    ),
                    "selected_child_id": selected_child_id,
                    "tree_context_before": "\n".join(
                        entry.format() for entry in expansion.tree_context_before
                    ),
                    "tree_context_after": "\n".join(
                        entry.format() for entry in expansion.tree_context_after
                    ),
                }
            )
        selected_action_node = None
        if pending_node is not None and pending_node.type == NodeType.ACTION:
            selected_action_node = pending_node.to_dict()
        return {"expansions": expansions, "selected_action_node": selected_action_node}

    def build_plan_info(
        self,
        *,
        step_index: int | None,
        pending_node: Node | None,
        max_depth: int,
        budget: int,
        max_retries: int,
        search_counter: int,
    ) -> dict:
        completed_plan_steps, future_plan_steps = self.stack.plan
        current_plan_step = pending_node.description if pending_node is not None else None
        tree_evolution = self.get_last_tree_evolution(pending_node)
        expansions = tree_evolution.get("expansions", [])
        tree_update_result = self.last_tree_update_result or {}
        tree_counts = self._build_tree_counts(pending_node, self.stack.items)
        tree_snapshot = self._build_tree_snapshot(pending_node, self.stack.items)
        max_depth_reached = tree_counts.get("max_depth_reached", 0)
        hit_limit_this_step = any(
            expansion.get("expanded_node_depth", -1) >= max_depth for expansion in expansions
        )
        selected_node = tree_evolution.get("selected_action_node")
        selected_node_depth = None if selected_node is None else selected_node.get("depth")

        return {
            "config": {
                "max_depth": max_depth,
                "budget": budget,
                "max_retries": max_retries,
            },
            "step_context": {
                "step_index": step_index,
                "pending_outcome_previous_step": self.last_pending_outcome,
                "search_counter": search_counter,
            },
            "plan_steps": {
                "current": current_plan_step,
                "completed": completed_plan_steps,
                "future": future_plan_steps,
            },
            "tree_update": {
                "prune": tree_update_result.get("prune", []),
                "update": tree_update_result.get("update", {}),
                "n_pruned": len(tree_update_result.get("prune", [])),
                "n_updated": len(tree_update_result.get("update", {})),
            },
            "selection": {
                "selected_action_node": selected_node,
                "selected_action_node_depth": selected_node_depth,
            },
            "expansions": expansions,
            "tree_counts": tree_counts,
            "tree_snapshot": tree_snapshot,
            "depth": {
                "limit": max_depth,
                "max_depth_reached": max_depth_reached,
                "hit_limit_this_step": hit_limit_this_step,
            },
        }

    def _get_root_node(
        self, pending_node: Node | None, stack: list[tuple[Node, NodeState]]
    ) -> Node | None:
        if pending_node is not None:
            root = pending_node
            while root.parent is not None:
                root = root.parent
            return root
        if stack:
            root = stack[0][0]
            while root.parent is not None:
                root = root.parent
            return root
        return None

    def _walk_tree(self, root: Node, include_deleted: bool = True) -> list[Node]:
        nodes: list[Node] = []

        def _walk(node: Node):
            if not include_deleted and node.status == NodeStatus.DELETED:
                return
            nodes.append(node)
            for child in node.children:
                _walk(child)

        _walk(root)
        return nodes

    def _build_tree_counts(
        self, pending_node: Node | None, stack_items: list[tuple[Node, NodeState]]
    ) -> dict:
        root = self._get_root_node(pending_node, stack_items)
        if root is None:
            return {
                "n_nodes_total": 0,
                "n_nodes_alive": 0,
                "by_type": {},
                "by_status": {},
                "by_depth": {},
                "max_depth_reached": 0,
            }

        all_nodes = self._walk_tree(root, include_deleted=True)
        alive_nodes = [n for n in all_nodes if n.status != NodeStatus.DELETED]

        by_type = {}
        by_status = {}
        by_depth = {}
        max_depth_reached = 0

        for node in all_nodes:
            by_type[node.type.name] = by_type.get(node.type.name, 0) + 1
            by_status[node.status.name] = by_status.get(node.status.name, 0) + 1
            depth_str = str(node.depth)
            by_depth[depth_str] = by_depth.get(depth_str, 0) + 1
            max_depth_reached = max(max_depth_reached, node.depth)

        return {
            "n_nodes_total": len(all_nodes),
            "n_nodes_alive": len(alive_nodes),
            "by_type": by_type,
            "by_status": by_status,
            "by_depth": by_depth,
            "max_depth_reached": max_depth_reached,
        }

    def _build_tree_snapshot(
        self, pending_node: Node | None, stack_items: list[tuple[Node, NodeState]]
    ) -> list[dict]:
        root = self._get_root_node(pending_node, stack_items)
        if root is None:
            return []
        all_nodes = self._walk_tree(root, include_deleted=True)
        return [node.to_dict() for node in all_nodes]
