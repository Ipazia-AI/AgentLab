import html
import os

import graphviz
import imageio
import networkx as nx
import numpy as np
import pydot
from PIL import Image


class TreeVisualizer:
    def __init__(self, data):
        self.data = data
        self.steps = sorted(data.keys(), key=lambda x: int(x.split("_")[1]))
        self.type_colors = {
            "OR": "#EBF5FB", "AND": "#D6EAF8", 
            "ACTION": "#85C1E9", "UNKNOWN": "#F4F6F7"
        }
        self.status_colors = {
            "SUCCESS": "#1E8449", "VISITED": "#2E86C1", 
            "UNVISITED": "#ABB2B9", "PRUNED": "#CB4335"
        }

    def get_global_layout(self):
        """Calculates fixed positions and bounding box for stability."""
        full_G = nx.DiGraph()
        for step_name in self.steps:
            snapshot = self.data[step_name].get("tree_snapshot", [])
            for node in snapshot:
                full_G.add_node(node["id"])
                if node.get("parent_id"):
                    full_G.add_edge(node["parent_id"], node["id"])
        
        pydot_graph = nx.drawing.nx_pydot.to_pydot(full_G)
        pydot_graph.set_rankdir("LR")
        layout_dot = pydot_graph.create_dot(prog="dot").decode("utf-8")
        positioned_graphs = pydot.graph_from_dot_data(layout_dot)
        
        fixed_pos = {}
        coords = []
        if positioned_graphs:
            for node in positioned_graphs[0].get_nodes():
                name = node.get_name().strip('"')
                if name in ("graph", "node", "edge"): 
                    continue
                pos = node.get_pos()
                if pos:
                    clean_pos = pos.replace('"', '')
                    fixed_pos[name] = f"{clean_pos}!"
                    coords.append([float(x) for x in clean_pos.split(',')])

        coords = np.array(coords)
        anchors = {}
        if len(coords) > 0:
            min_x, min_y = coords.min(axis=0)
            max_x, max_y = coords.max(axis=0)
            anchors = {
                "top_left": f"{min_x-50},{max_y+50}!",
                "bottom_right": f"{max_x+50},{min_y-50}!"
            }
        return fixed_pos, list(full_G.nodes()), anchors

    def create_frame(self, step_name, fixed_layout, all_node_ids, anchors):
        step_data = self.data[step_name]
        snapshot = step_data.get("tree_snapshot", [])
        updates = step_data.get("tree_update", {})
        pruned_ids = updates.get("prune", [])
        
        retry_val = 0
        if step_data.get("expansions"):
            retry_val = step_data["expansions"][0].get("retry", 0)

        # Using neato and splines=line for maximum arrow stability
        dot = graphviz.Digraph(engine="neato", format="png")
        dot.attr(overlap="false", splines="line", bgcolor="white", pad="0.5")
        
        # Lock viewport scale
        for aid, apos in anchors.items():
            dot.node(aid, label="", pos=apos, style="invis")

        # TITLE LOGIC: Sanitize and format
        step_info = f"Step: {step_name.split('_')[1]} | Retry: {retry_val}"
        
        # ESCAPE the action text to prevent "<node_expansion>" from being treated as an HTML tag
        raw_action = step_data['plan_steps']['current']
        safe_action = html.escape(raw_action[:80]) 
        
        title_html = (
            f'<<FONT FACE="Helvetica-Bold" POINT-SIZE="18">{step_info}</FONT>'
            f'<BR/>'
            f'<FONT FACE="Helvetica" POINT-SIZE="11">{safe_action}...</FONT>>'
        )
        
        dot.attr(label=title_html, labelloc="t", fontcolor="#1B4F72")

        current_node_map = {str(n["id"]): n for n in snapshot}
        
        for nid in all_node_ids:
            pos = fixed_layout.get(nid)
            if not pos: 
                continue

            if nid in current_node_map and current_node_map[nid]["status"] != "DELETED":
                node_data = current_node_map[nid]
                is_pruning = nid in pruned_ids
                
                bg_color = self.type_colors.get(node_data["type"], "#FFFFFF")
                border_color = self.status_colors["PRUNED"] if is_pruning else self.status_colors.get(node_data["status"], "#BDC3C7")
                pen_width = "5" if is_pruning or node_data["status"] == "SUCCESS" else "2"
                style = "filled, rounded, dashed" if is_pruning else "filled, rounded"
                
                label = f"""<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0">
                               <TR><TD><FONT POINT-SIZE="8" COLOR="#7F8C8D">{nid}</FONT></TD></TR>
                               <TR><TD><B><FONT POINT-SIZE="11">{node_data['type']}</FONT></B></TD></TR>
                               {f'<TR><TD><I><FONT POINT-SIZE="9" COLOR="red">PRUNED</FONT></I></TD></TR>' if is_pruning else ''}
                            </TABLE>>"""

                dot.node(nid, label=label, pos=pos, style=style, fillcolor=bg_color, 
                         color=border_color, penwidth=pen_width, shape="rect", fontname="Helvetica")
                
                pid = str(node_data.get("parent_id"))
                if pid and pid in current_node_map and current_node_map[pid]["status"] != "DELETED":
                    dot.edge(pid, nid, color="#ABB2B9", penwidth="1.5", arrowhead="vee")
            else:
                dot.node(nid, label="", pos=pos, style="invis")

        return dot

def generate_visual_evolution(data, output_gif="tree_evolution.gif", fps=1):
    viz = TreeVisualizer(data)
    fixed_layout, all_node_ids, anchors = viz.get_global_layout()
    
    temp_folder = "frames"
    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder)
    
    frames = []
    print(f"Processing steps at {fps} FPS...")
    for i, step_name in enumerate(viz.steps):
        gv = viz.create_frame(step_name, fixed_layout, all_node_ids, anchors)
        path = os.path.join(temp_folder, f"frame_{i:03d}")
        gv.render(path, cleanup=True)
        
        img = Image.open(f"{path}.png").convert("RGB")
        frames.append(np.array(img))

    max_h = max(f.shape[0] for f in frames)
    max_w = max(f.shape[1] for f in frames)
    
    final_frames = []
    for f in frames:
        new_im = Image.new("RGB", (max_w, max_h), (255, 255, 255))
        new_im.paste(Image.fromarray(f), (0, 0))
        final_frames.append(np.array(new_im))

    imageio.mimsave(output_gif, final_frames, fps=fps, loop=0)
    print(f"Animation saved to {output_gif}")
    
    # Cleanup
    for f in os.listdir(temp_folder):
        os.remove(os.path.join(temp_folder, f))
    os.rmdir(temp_folder)

# def get_global_layout(data):
#     """
#     Calculates a single global layout for all nodes present across all steps.
#     """
#     full_G = nx.DiGraph()
#     for step in data.values():
#         for node in step["tree_snapshot"]:
#             full_G.add_node(node["id"])
#             if node["parent_id"]:
#                 full_G.add_edge(node["parent_id"], node["id"])

#     pydot_graph = nx.drawing.nx_pydot.to_pydot(full_G)
#     pydot_graph.set_rankdir("LR")

#     layout_dot = pydot_graph.create_dot(prog="dot").decode("utf-8")
#     positioned_graphs = pydot.graph_from_dot_data(layout_dot)

#     if not positioned_graphs:
#         return {}, []

#     positioned_graph = positioned_graphs[0]
#     fixed_pos = {}
#     all_node_ids = []

#     for node in positioned_graph.get_nodes():
#         name = node.get_name().strip('"')
#         if name in ("graph", "node", "edge"):
#             continue
#         pos = node.get_pos()
#         if pos:
#             clean_pos = pos.replace('"', "")
#             fixed_pos[name] = f"{clean_pos}!"
#             all_node_ids.append(name)

#     return fixed_pos, all_node_ids


# def create_fixed_graphviz_step(snapshot, fixed_layout, all_node_ids, title="Tree Step"):
#     """
#     Renders a snapshot with a modern blue theme and rounded rectangles.
#     """
#     # CSS-inspired Blue Palette
#     type_colors = {
#         "OR": "#EBF5FB",  # Extra Light Blue
#         "AND": "#AED6F1",  # Light Blue
#         "ACTION": "#5DADE2",  # Professional Blue
#         "UNKNOWN": "#F4F6F7",  # Ghost White
#     }

#     status_border_colors = {
#         "SUCCESS": "#1B4F72",  # Dark Navy
#         "VISITED": "#2E86C1",  # Ocean Blue
#         "UNVISITED": "#D6DBDF",  # Silver Blue
#     }

#     dot = graphviz.Digraph(engine="neato", format="png")
#     dot.attr(overlap="false", splines="true", bgcolor="white", pad="0.5")
#     dot.attr(
#         label=f"\n\n{title}\n",
#         labelloc="t",
#         fontsize="24",
#         fontname="Helvetica-Bold",
#         fontcolor="#1B4F72",
#     )

#     current_nodes = {str(n["id"]): n for n in snapshot}
#     current_edges = {(str(n["parent_id"]), str(n["id"])) for n in snapshot if n["parent_id"]}

#     for nid in all_node_ids:
#         pos = fixed_layout.get(nid)
#         if not pos:
#             continue

#         if nid in current_nodes:
#             node_data = current_nodes[nid]
#             ntype = node_data["type"]
#             status = node_data["status"]

#             # Label with ID in small font and Type in Bold
#             label = f"""<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="2">
#                            <TR><TD><FONT POINT-SIZE="9" COLOR="#2E86C1">{nid}</FONT></TD></TR>
#                            <TR><TD><B><FONT POINT-SIZE="12" COLOR="#1B4F72">{ntype}</FONT></B></TD></TR>
#                         </TABLE>>"""

#             # Forcing "rect" and "rounded" for every node
#             dot.node(
#                 nid,
#                 label=label,
#                 pos=pos,
#                 style="filled, rounded",
#                 fillcolor=type_colors.get(ntype, "#FFFFFF"),
#                 color=status_border_colors.get(status, "#BDC3C7"),
#                 penwidth="4" if status == "SUCCESS" else "1.5",
#                 shape="rect",
#                 fontname="Helvetica",
#             )
#         else:
#             # Invisible placeholder
#             dot.node(nid, label="", pos=pos, style="invis", width="1.0", height="0.6")

#     # Draw Edges with a subtle blue-grey color
#     for parent, child in current_edges:
#         dot.edge(parent, child, color="#AED6F1", penwidth="2", arrowhead="vee")

#     return dot


# def generate_tree_animation(data, filename="blue_tree_evolution.gif", fps=1):
#     temp_folder = "temp_frames"
#     if not os.path.exists(temp_folder):
#         os.makedirs(temp_folder)

#     fixed_layout, all_node_ids = get_global_layout(data)
#     images = []
#     steps = sorted(data.keys(), key=lambda x: int(x.split("_")[1]))

#     print(f"Generating frames for {len(steps)} steps...")

#     for i, step_name in enumerate(steps):
#         snapshot = data[step_name]["tree_snapshot"]
#         gv_graph = create_fixed_graphviz_step(
#             snapshot, fixed_layout, all_node_ids, title=f"Exploration: {step_name}"
#         )

#         frame_base = os.path.join(temp_folder, f"frame_{i:03d}")
#         gv_graph.render(frame_base, cleanup=True)

#         img = Image.open(f"{frame_base}.png").convert("RGB")
#         images.append(np.array(img))

#     # Standardize sizes to prevent imageio errors
#     max_h = max(im.shape[0] for im in images)
#     max_w = max(im.shape[1] for im in images)

#     final_frames = []
#     for im in images:
#         if im.shape[0] != max_h or im.shape[1] != max_w:
#             new_im = Image.new("RGB", (max_w, max_h), (255, 255, 255))
#             new_im.paste(Image.fromarray(im), (0, 0))
#             final_frames.append(np.array(new_im))
#         else:
#             final_frames.append(im)

#     imageio.mimsave(filename, final_frames, fps=fps, loop=0)
#     print(f"Done! Animation saved as: {filename}")

#     # Cleanup
#     for f in os.listdir(temp_folder):
#         os.remove(os.path.join(temp_folder, f))
#     os.rmdir(temp_folder)