"""Merge-tree icicle plot, shared across MSSeg instances.

Instance-agnostic: it draws any merge tree in the flat ``{nodes, roots}`` JSON
produced by the core (each node = ``{node_id, value, voxel_count, children:[i...]}``,
``roots`` = a list of node indices), coloring each box by its subtree's deepest
minimum. Fully iterative (explicit stacks) so a very deep tree can't overflow.
"""
from matplotlib.patches import Rectangle

from .palette import min_color


def draw_icicle(ax, tree, color_of_min=min_color):
    """Draw the voxel-count merge-tree icicle from the flat {nodes, roots} format,
    each box colored by its subtree's deepest minimum via color_of_min(node_id).
    Fully iterative (explicit stacks) so a million-deep tree can't overflow."""
    nodes = tree["nodes"]
    ax.clear()
    if not nodes:
        return 0.0, 1.0
    n = len(nodes)

    # Pass 1 (post-order): lo = lowest leaf value in subtree, rep = its node_id.
    lo = [0.0] * n
    rep = [0] * n
    stack = [(r, False) for r in tree["roots"]]
    while stack:
        idx, done = stack.pop()
        kids = nodes[idx]["children"]
        if done:
            if not kids:
                lo[idx] = nodes[idx]["value"]
                rep[idx] = nodes[idx]["node_id"]
            else:
                best_lo, best_rep = None, None
                for c in kids:
                    if best_lo is None or lo[c] < best_lo:
                        best_lo, best_rep = lo[c], rep[c]
                lo[idx], rep[idx] = best_lo, best_rep
        else:
            stack.append((idx, True))
            for c in kids:
                stack.append((c, False))

    vmin = min(nd["value"] for nd in nodes)
    vmax = max(nd["value"] for nd in nodes)
    vspan = (vmax - vmin) or 1.0
    root_top = vmax + 0.05 * vspan

    # Pass 2 (pre-order): x0 per node; children sorted by lo, laid left-to-right.
    boxes = []
    cursor = 0.0
    root_items = []
    for r in sorted(tree["roots"], key=lambda k: lo[k]):
        root_items.append((r, cursor, root_top))
        cursor += float(nodes[r]["voxel_count"])
    total = cursor or 1.0
    stack = list(reversed(root_items))
    while stack:
        idx, x0, pv = stack.pop()
        nd = nodes[idx]
        boxes.append((x0, nd["value"], float(nd["voxel_count"]), pv - nd["value"],
                      color_of_min(rep[idx])))
        c_cursor = x0
        child_items = []
        for c in sorted(nd["children"], key=lambda k: lo[k]):
            child_items.append((c, c_cursor, nd["value"]))
            c_cursor += float(nodes[c]["voxel_count"])
        stack.extend(reversed(child_items))

    for x0, y0, w, h, color in boxes:
        ax.add_patch(Rectangle((x0, y0), w, h, facecolor=color, edgecolor="none"))
    ax.set_xlim(0, total)
    ax.set_ylim(vmin - 0.02 * vspan, root_top)
    ax.set_xlabel("voxel count (feature size)")
    ax.set_ylabel("function value")
    return vmin - 0.02 * vspan, root_top
