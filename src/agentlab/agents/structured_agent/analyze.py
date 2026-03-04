def _code_block(text: str) -> str:
    return f"```\n{text}\n```"


def format_hpa_plan_markdown(plan: dict | None) -> str:
    if not isinstance(plan, dict) or not plan:
        return "No HPA plan data available for this step."

    plan_steps = plan.get("plan_steps", {})
    current = plan_steps.get("current")
    completed = plan_steps.get("completed") or []
    future = plan_steps.get("future") or []
    tree_update = plan.get("tree_update", {})
    expansions = plan.get("expansions", [])
    selection = plan.get("selection", {})
    selected_action_node = selection.get("selected_action_node")

    lines = ["## HPA Plan"]
    lines.append(f"### Current Step\n- {current if current else 'N/A'}")

    lines.append("### Completed Steps")
    if completed:
        lines.extend([f"- {step}" for step in completed])
    else:
        lines.append("- None")

    lines.append("### Future Steps")
    if future:
        lines.extend([f"- {step}" for step in future])
    else:
        lines.append("- None")

    lines.append("### Tree Update Result")
    if not isinstance(tree_update, dict):
        lines.append("- None")
    else:
        pruned_nodes = tree_update.get("prune", [])
        updated_nodes = tree_update.get("update", {})

        lines.append("#### Prune")
        if isinstance(pruned_nodes, list) and pruned_nodes:
            lines.extend([f"- `{node_id}`" for node_id in pruned_nodes])
        elif isinstance(pruned_nodes, list):
            lines.append("- None")
        else:
            lines.append(_code_block(str(pruned_nodes)))

        lines.append("#### Update")
        if isinstance(updated_nodes, dict) and updated_nodes:
            for node_id, new_desc in updated_nodes.items():
                lines.append(f"- `{node_id}` -> {new_desc}")
        elif isinstance(updated_nodes, dict):
            lines.append("- None")
        else:
            lines.append(_code_block(str(updated_nodes)))

    lines.append("### Tree Evolution To Current Step")
    if expansions and isinstance(expansions[0], dict) and "order" in expansions[0]:
        expansions = sorted(expansions, key=lambda e: e.get("order", 0))
    if expansions:
        for expansion in expansions:
            order = expansion.get("order", "N/A")
            node_id = expansion.get("expanded_node_id", "N/A")
            node_type = expansion.get("expanded_node_type_after", "N/A")
            node_desc = expansion.get("expanded_node_description", "N/A")
            retry = expansion.get("retry", "N/A")
            lines.append(
                f"#### Expansion {order} (retry {retry})\n- Node: `{node_id}`\n- Type after expansion: `{node_type}`\n- Description: {node_desc}"
            )

            children_after = expansion.get("children_after", [])
            lines.append("##### Children After Expansion")
            if children_after:
                for child in children_after:
                    child_id = child.get("id", "N/A")
                    child_type = child.get("type", "N/A")
                    child_status = child.get("status", "N/A")
                    child_desc = child.get("description", "N/A")
                    lines.append(f"- `{child_id}` ({child_type}, {child_status}): {child_desc}")
            else:
                lines.append("- None")

            tree_after = expansion.get("tree_context_after")
            if tree_after:
                lines.append("##### Tree Context After Expansion")
                lines.append(_code_block(tree_after))
    else:
        lines.append("- No expansion trace available for this step.")

    lines.append("#### Selected Action Node")
    if isinstance(selected_action_node, dict):
        selected_id = selected_action_node.get("id", "N/A")
        selected_desc = selected_action_node.get("description", "N/A")
        selected_parent_id = selected_action_node.get("parent_id", "N/A")
        selected_parent_type = selected_action_node.get("parent_type", "N/A")
        lines.append(
            f"- Node: `{selected_id}`\n- Description: {selected_desc}\n- Parent: `{selected_parent_id}` ({selected_parent_type})"
        )
    else:
        lines.append("- None")

    return "\n".join(lines)
