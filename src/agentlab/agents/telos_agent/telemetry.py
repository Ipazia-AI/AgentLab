"""Record Telos planner events and dump a structured plan trace for AgentInfo."""

from collections.abc import Sequence
from typing import Any

from telos import ActionNode, Result
from telos.core.interfaces import (
    Expansion,
    Insights,
    Planner,
    PlanningContext,
    Verification,
)
from telos.core.model import Node, NodeId
from telos.planner.llm_planner import LLMPlanner

from agentlab.agents.telos_agent.planner_llm import AgentLabPlannerLLM


class RecordingPlannerLLM:
    """Wrap a PlannerLLM and keep the last completion string."""

    def __init__(self, inner: AgentLabPlannerLLM):
        self._inner = inner
        self.last_raw = ""

    def complete(self, messages: Sequence) -> str:
        self.last_raw = self._inner.complete(messages) or ""
        return self.last_raw

    def get_stats(self) -> dict:
        return self._inner.get_stats()


class RecordingPlanner:
    """Planner wrapper that records expand / recover / verify / prune calls."""

    def __init__(self, inner: Planner, llm: RecordingPlannerLLM | None = None):
        self._inner = inner
        self._llm = llm
        self.reset()

    def reset(self) -> None:
        self.expansions: list[dict] = []
        self.recoveries: list[dict] = []
        self.verifications: list[dict] = []
        self.prunes: list[list[str]] = []

    def _raw(self) -> str:
        return self._llm.last_raw if self._llm is not None else ""

    def form_insights(self, ctx: PlanningContext) -> Insights:
        return self._inner.form_insights(ctx)

    def expand(self, node: Node, ctx: PlanningContext) -> Expansion:
        expansion = self._inner.expand(node, ctx)
        self.expansions.append(
            {
                "node_id": node.id,
                "type": expansion.type.name,
                "description": expansion.description,
                "children": list(expansion.children),
                "tree_view": ctx.tree_view,
            }
        )
        return expansion

    def recover(self, node: Node, children: Sequence[Node], ctx: PlanningContext) -> list[str]:
        new_children = self._inner.recover(node, children, ctx)
        self.recoveries.append(
            {
                "node_id": node.id,
                "node_description": node.description,
                "new_children": list(new_children),
                "raw": self._raw(),
                "tree_view": ctx.tree_view,
            }
        )
        return new_children

    def verify(self, node: Node, ctx: PlanningContext) -> Verification:
        verification = self._inner.verify(node, ctx)
        self.verifications.append(
            {
                "node_id": node.id,
                "node_description": node.description,
                "action_success": verification.action_success,
                "goal_reached": verification.goal_reached,
                "raw": self._raw(),
            }
        )
        return verification

    def prune(self, node: Node, ctx: PlanningContext) -> list[NodeId]:
        ids = self._inner.prune(node, ctx)
        self.prunes.append([str(nid) for nid in ids])
        return ids


def make_recording_planner(planner_llm: AgentLabPlannerLLM) -> tuple[RecordingPlannerLLM, RecordingPlanner]:
    recording_llm = RecordingPlannerLLM(planner_llm)
    recorder = RecordingPlanner(LLMPlanner(recording_llm), recording_llm)
    return recording_llm, recorder


def build_telos_plan(
    state: Any,
    recorder: RecordingPlanner | None,
    outcome: ActionNode | Result,
    grounded: Any,
    last_action_error: str,
    completed_steps: list[str],
) -> dict:
    """Build a JSON-friendly plan dump from a Telos snapshot and recorder."""
    nodes = _nodes_from_state(state)
    nodes_by_id = {n["id"]: n for n in nodes if "id" in n}
    is_result = isinstance(outcome, Result)
    selected_id = None if is_result else outcome.id
    selected = None if is_result else nodes_by_id.get(selected_id, {})
    current = None if is_result else outcome.description
    grounded_action = grounded.action if grounded is not None else None
    insights = _insights_dict(getattr(state, "insights", None) if state is not None else None)
    tree_view = _tree_view(state)
    horizon = getattr(state, "horizon_T", None) if state is not None else None
    pending = getattr(state, "pending", None) if state is not None else None
    terminal = _terminal_name(state, outcome)

    return {
        "config": {"horizon_T": horizon},
        "step_context": {
            "t": getattr(state, "t", None) if state is not None else None,
            "pending": pending,
            "terminal": terminal,
        },
        "insights": insights,
        "stack": _stack_frames(state),
        "tree_snapshot": nodes,
        "tree_view": tree_view,
        "tree_counts": _tree_counts(nodes),
        "selection": None
        if is_result
        else {
            "id": selected_id,
            "description": current,
            "parent": selected.get("parent", getattr(outcome, "parent", None)),
            "type": selected.get("type"),
            "status": selected.get("status"),
            "grounded_action": grounded_action,
        },
        "plan_steps": {
            "current": current,
            "completed": list(completed_steps),
            "future": _future_steps(nodes, selected_id),
        },
        "expansions": list(recorder.expansions) if recorder is not None else [],
        "recoveries": list(recorder.recoveries) if recorder is not None else [],
        "action_verification": _action_verification(recorder, last_action_error),
        "prune": _prune_ids(recorder),
    }


def format_telos_plan_markdown(plan: dict | None) -> str:
    if not isinstance(plan, dict) or not plan:
        return "No Telos plan data available for this step."

    lines = ["## Telos Plan"]
    step = plan.get("step_context") or {}
    lines.append(f"### Step\n- t: `{step.get('t')}`\n- pending: `{step.get('pending')}`")
    if step.get("terminal"):
        lines.append(f"- terminal: **{step['terminal']}**")

    insights = plan.get("insights")
    lines.append("### Insights")
    if isinstance(insights, dict):
        lines.append(f"- Constraints: {insights.get('constraints') or 'N/A'}")
        lines.append(f"- Progress: {insights.get('progress') or 'N/A'}")
        lines.append(f"- Suggestion: {insights.get('suggestion') or 'N/A'}")
    else:
        lines.append("- None")

    plan_steps = plan.get("plan_steps") or {}
    lines.append(f"### Current Step\n- {plan_steps.get('current') or 'N/A'}")
    lines.append("### Completed Steps")
    completed = plan_steps.get("completed") or []
    lines.extend([f"- {step_desc}" for step_desc in completed] or ["- None"])
    lines.append("### Future Steps")
    future = plan_steps.get("future") or []
    lines.extend([f"- {step_desc}" for step_desc in future] or ["- None"])

    selection = plan.get("selection")
    lines.append("### Selection")
    if isinstance(selection, dict):
        lines.append(
            f"- Node: `{selection.get('id')}` ({selection.get('type')}, {selection.get('status')})\n"
            f"- Description: {selection.get('description')}\n"
            f"- Grounded action: `{selection.get('grounded_action')}`"
        )
    else:
        lines.append("- None")

    verification = plan.get("action_verification")
    lines.append("### Action Verification (Previous Step)")
    if isinstance(verification, dict):
        lines.append(
            f"- Node: `{verification.get('node_id', 'N/A')}`\n"
            f"- Success: **{verification.get('action_success')}**\n"
            f"- Goal reached: **{verification.get('goal_reached')}**"
        )
        if verification.get("error"):
            lines.append(f"- Error: {verification['error']}")
        if verification.get("raw"):
            lines.append("##### Raw verification")
            lines.append(_code_block(verification["raw"]))
    else:
        lines.append("- None")

    recoveries = plan.get("recoveries") or []
    lines.append("### Recoveries")
    if recoveries:
        for i, recovery in enumerate(recoveries, start=1):
            lines.append(
                f"#### Recovery {i}\n- Node: `{recovery.get('node_id')}`\n"
                f"- New children: {recovery.get('new_children')}"
            )
            if recovery.get("raw"):
                lines.append("##### Raw recovery")
                lines.append(_code_block(recovery["raw"]))
    else:
        lines.append("- None")

    expansions = plan.get("expansions") or []
    lines.append("### Expansions")
    if expansions:
        for i, expansion in enumerate(expansions, start=1):
            lines.append(
                f"#### Expansion {i}\n- Node: `{expansion.get('node_id')}`\n"
                f"- Type: `{expansion.get('type')}`\n"
                f"- Description: {expansion.get('description')}"
            )
            children = expansion.get("children") or []
            if children:
                lines.append("##### Children")
                lines.extend([f"- {child}" for child in children])
    else:
        lines.append("- None")

    prune = plan.get("prune") or []
    lines.append("### Prune")
    if prune:
        lines.extend([f"- `{nid}`" for nid in prune])
    else:
        lines.append("- None")

    stack = plan.get("stack") or []
    lines.append("### Stack")
    if stack:
        lines.extend([f"- `{frame.get('node_id')}` {frame.get('state')}" for frame in stack])
    else:
        lines.append("- None")

    tree_view = plan.get("tree_view")
    lines.append("### Tree")
    if tree_view:
        lines.append(_code_block(tree_view))
    else:
        lines.append("- None")

    return "\n".join(lines)


def _code_block(text: str) -> str:
    return f"```\n{text}\n```"


def _nodes_from_state(state: Any) -> list[dict]:
    tree = getattr(state, "tree", None) if state is not None else None
    to_dict = getattr(tree, "to_dict", None)
    if callable(to_dict):
        data = to_dict() or {}
        return list(data.get("nodes") or [])
    return []


def _tree_view(state: Any) -> str | None:
    tree = getattr(state, "tree", None) if state is not None else None
    render = getattr(tree, "render_view", None)
    if callable(render):
        return render()
    return None


def _insights_dict(insights: Any) -> dict | None:
    if insights is None:
        return None
    constraints = getattr(insights, "constraints", None)
    progress = getattr(insights, "progress", None)
    suggestion = getattr(insights, "suggestion", None)
    if constraints is None and progress is None and suggestion is None:
        return None
    return {
        "constraints": constraints,
        "progress": progress,
        "suggestion": suggestion,
    }


def _stack_frames(state: Any) -> list[dict]:
    stack = getattr(state, "stack", None) if state is not None else None
    if not stack:
        return []
    frames = []
    for item in stack:
        if isinstance(item, dict):
            frames.append(item)
            continue
        node_id, stack_state = item
        name = getattr(stack_state, "name", stack_state)
        frames.append({"node_id": node_id, "state": name})
    return frames


def _tree_counts(nodes: list[dict]) -> dict:
    by_type: dict[str, int] = {}
    by_status: dict[str, int] = {}
    by_depth: dict[str, int] = {}
    max_depth = 0
    for node in nodes:
        node_type = node.get("type") or "UNKNOWN"
        status = node.get("status") or "UNVISITED"
        depth = str(node.get("id") or "0").count(".")
        by_type[node_type] = by_type.get(node_type, 0) + 1
        by_status[status] = by_status.get(status, 0) + 1
        by_depth[str(depth)] = by_depth.get(str(depth), 0) + 1
        max_depth = max(max_depth, depth)
    alive = sum(1 for node in nodes if node.get("status") != "DELETED")
    return {
        "n_nodes_total": len(nodes),
        "n_nodes_alive": alive,
        "by_type": by_type,
        "by_status": by_status,
        "by_depth": by_depth,
        "max_depth_reached": max_depth,
    }


def _future_steps(nodes: list[dict], selected_id: str | None) -> list[str]:
    future = []
    for node in nodes:
        if node.get("id") == selected_id:
            continue
        if node.get("type") == "UNKNOWN" or node.get("status") == "UNVISITED":
            desc = node.get("description")
            if desc:
                future.append(desc)
    return future


def _terminal_name(state: Any, outcome: ActionNode | Result) -> str | None:
    if isinstance(outcome, Result):
        return outcome.status.name
    terminal = getattr(state, "terminal", None) if state is not None else None
    status = getattr(terminal, "status", None)
    return getattr(status, "name", None)


def _action_verification(recorder: RecordingPlanner | None, last_action_error: str) -> dict | None:
    error = last_action_error or ""
    if recorder is not None and recorder.verifications:
        verification = dict(recorder.verifications[-1])
        verification["error"] = error
        return verification
    if error:
        return {
            "node_id": None,
            "node_description": None,
            "action_success": None,
            "goal_reached": None,
            "raw": None,
            "error": error,
        }
    return None


def _prune_ids(recorder: RecordingPlanner | None) -> list[str]:
    if recorder is None:
        return []
    ids: list[str] = []
    for proposed in recorder.prunes:
        ids.extend(proposed)
    return ids
