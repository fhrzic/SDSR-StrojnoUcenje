# =========================
# Plotter: Draw layered network + random weights/biases in [-1,1]
# =========================
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from IPython.display import display, Math, HTML

# Unicode superscript helper (used only in console, if needed elsewhere)
def _superscript_int(n: int) -> str:
    _map = str.maketrans("0123456789()-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁽⁾⁻")
    return str(n).translate(_map)

def draw_network(
    layer_sizes,
    figsize=(10, 5),
    node_radius=0.15,
    layer_gap=2.0,
    vpad=0.8,
    show_weights=True,
    seed=None,
    edge_cmap_name="tab20",
    edge_alpha=0.9,
    show_column_labels=True,
    activations=None,
    activation_fontsize=9,
    activation_color="#222222",
    print_weight_matrices=False,
    indices_start=1,
    # Bias drawing controls
    draw_bias_nodes=True,
    bias_radius=0.12,
    bias_offset_factor=0.9,
    bias_edge_alpha=0.85,
    bias_color="#333333",
    # Display options
    latex_weight_print=False,  # True → LaTeX (MathJax) blocks in Jupyter
    latex_precision=1,
):
    """
    Draw a fully-connected neural network diagram with Θ and bias vectors.
    - Legend/print uses θ_{ij}^ℓ format (subscript ij, superscript layer).
    - Bias entries in print/display are in TOP→DOWN order (top neuron = 1).
    - On-figure bias labels are numeric values.
    Returns: weights_mats, biases_mats, total_params
    """

    rng = np.random.default_rng(seed)
    n_layers = len(layer_sizes)

    # Normalize activations
    if activations is None:
        activations = ['-'] + ['r'] * max(0, n_layers - 2) + ['s']
    else:
        if len(activations) == n_layers - 1:
            activations = ['-'] + list(activations)
        elif len(activations) != n_layers:
            raise ValueError("Length of activations must match number of layers.")

    def act_to_letter(a):
        a = (a or '-').lower()
        return {'r':'r', 'relu':'r', 's':'σ', 'sigmoid':'σ',
                '-':'', 'none':'', 'linear':''}.get(a, a)

    # Figure
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect('equal')
    ax.axis('off')

    xs = np.arange(n_layers) * layer_gap
    max_nodes = max(layer_sizes)
    y_span = max_nodes - 1 + vpad * 2

    # Vertical positions per layer (top→bottom)
    layer_ys = []
    for n in layer_sizes:
        ys = np.array([0.0]) if n == 1 else np.linspace(-(n-1)/2, (n-1)/2, n)
        layer_ys.append(ys)
    scale = y_span / max(max_nodes-1, 1)
    layer_ys = [ys * scale for ys in layer_ys]

    # Helpers
    base_fractions = [1/3, 1/2, 2/3, 5/6, 1/4, 3/4]
    def fraction_for_source(i): return base_fractions[i % len(base_fractions)]
    perp_offset = 0.05 * layer_gap
    cmap = mpl.colormaps.get_cmap(edge_cmap_name)

    # Random weights + biases in [-1,1], rounded to 1 decimal
    weights_mats, biases_mats = [], []
    total_params = 0
    for L in range(n_layers - 1):
        n_src, n_tgt = layer_sizes[L], layer_sizes[L+1]
        W = np.round(rng.uniform(-1, 1, size=(n_src, n_tgt)), 1)
        b = np.round(rng.uniform(-1, 1, size=(n_tgt,)), 1)
        weights_mats.append(W)
        biases_mats.append(b)
        total_params += n_src * n_tgt + n_tgt

    # Draw edges + numeric weight labels
    for L in range(n_layers - 1):
        x0, x1 = xs[L], xs[L+1]
        n_src, n_tgt = layer_sizes[L], layer_sizes[L+1]
        W = weights_mats[L]
        denom = max(n_src*n_tgt - 1, 1)

        for si, y0 in enumerate(layer_ys[L]):
            t = fraction_for_source(si)
            for ti, y1 in enumerate(layer_ys[L+1]):
                k = si*n_tgt + ti
                c = cmap(k/denom)
                ax.plot([x0, x1], [y0, y1], color=c, alpha=edge_alpha, linewidth=1.2, zorder=2)

                if show_weights:
                    w = W[si, ti]
                    dx, dy = (x1-x0), (y1-y0)
                    px, py = x0 + t*dx, y0 + t*dy
                    seg_len = np.hypot(dx, dy)
                    nx, ny = (-dy/seg_len, dx/seg_len) if seg_len>0 else (0,1)
                    sign = 1 if (ti % 2 == 0) else -1
                    jitter = ((si % 3)-1)*0.25
                    ox, oy = (sign+jitter)*perp_offset*nx, (sign+jitter)*perp_offset*ny

                    ax.text(px+ox, py+oy, f"{w:.1f}",
                            fontsize=8, ha='center', va='center',
                            color=c,
                            bbox=dict(boxstyle='round,pad=0.1', fc='white', ec='none', alpha=0.7),
                            zorder=3)

    # Bias nodes + edges; on-figure bias labels are NUMERIC
    if draw_bias_nodes:
        for L in range(n_layers - 1):
            x_bias = xs[L]
            y_bias = layer_ys[L].max() + bias_offset_factor

            # circle "1"
            ax.add_patch(plt.Circle((x_bias, y_bias), bias_radius,
                                    facecolor="white", edgecolor=bias_color, linewidth=2, zorder=5))
            ax.text(x_bias, y_bias, "1", ha="center", va="center",
                    fontsize=10, color=bias_color, zorder=6)

            b_vec = biases_mats[L]
            for j, y_tgt in enumerate(layer_ys[L+1]):
                ax.plot([x_bias, xs[L+1]], [y_bias, y_tgt],
                        color=bias_color, alpha=bias_edge_alpha, linewidth=1.1, linestyle='-', zorder=1)
                if show_weights:
                    t = 1/3
                    dx, dy = (xs[L+1]-x_bias), (y_tgt-y_bias)
                    px, py = x_bias + t*dx, y_bias + t*dy
                    seg_len = np.hypot(dx, dy)
                    nx, ny = (-dy/seg_len, dx/seg_len) if seg_len>0 else (0,1)
                    ox, oy = 0.04 * layer_gap * nx, 0.04 * layer_gap * ny
                    # numeric bias value on the graph
                    ax.text(px+ox, py+oy, f"{b_vec[j]:.{latex_precision}f}",
                            fontsize=9, ha='center', va='center',
                            color=bias_color,
                            bbox=dict(boxstyle='round,pad=0.1', fc='white', ec='none', alpha=0.7),
                            zorder=4)

    # Nodes + activation letters
    for L, (x, ys) in enumerate(zip(xs, layer_ys)):
        letter = act_to_letter(activations[L])
        for y in ys:
            ax.add_patch(plt.Circle((x, y), node_radius, fill=False, linewidth=2, zorder=4))
            if letter:
                ax.text(x, y, letter, ha='center', va='center',
                        fontsize=activation_fontsize, color=activation_color, zorder=5)

    # Column labels (MathText)
    if show_column_labels:
        L_last = n_layers - 1
        hidden_labels = [rf"$f^{{{i}}}\,|\,h^{{{i}}}$" for i in range(1, L_last)]
        final_label = rf"$f^{{{L_last}}}\,|\,h^{{{L_last}}}\!\to y$"
        labels = ["$x$"] + hidden_labels + [final_label]

        y_min = min(min(ys) for ys in layer_ys) - node_radius*2
        y_label = y_min - node_radius*1.2
        for x, lab in zip(xs, labels):
            ax.text(x, y_label, lab, ha='center', va='top', fontsize=11)
        y_bottom = y_label - 0.2
    else:
        y_bottom = min(min(ys) for ys in layer_ys) - node_radius*2

    # y-limits include bias circles
    bias_tops = [layer_ys[L].max() + bias_offset_factor for L in range(n_layers - 1)]
    y_top_nodes = max(max(ys) for ys in layer_ys) + node_radius*2
    y_top_bias = (max(bias_tops) + bias_radius*2) if bias_tops else y_top_nodes
    ax.set_xlim(xs[0] - layer_gap*0.6, xs[-1] + layer_gap*0.6)
    ax.set_ylim(y_bottom, max(y_top_nodes, y_top_bias))
    plt.show()

    # ---------- Print / Display matrices ----------
    def _console_theta_block(L, W, b):
        """Console text with θ_{ij}^L and TOP→DOWN bias entries."""
        n_src, n_tgt = W.shape
        print(f"Θ{L} (layer {L} → {L+1}, shape {W.shape}):")
        rows = []
        # symbolic names (top=1 orientation for rows/cols)
        for i_print in range(n_src):
            row = []
            for j_print in range(n_tgt):
                ii = i_print + indices_start
                jj = j_print + indices_start
                row.append(f"θ_{{{ii}{jj}}}^{L}")
            rows.append("[" + ", ".join(row) + "]")
        print("[" + ", ".join(rows) + "]")
        print("Legend (top=1, bottom=N):")
        # numeric values (reverse to print top→down)
        for i_print, i in enumerate(range(n_src - 1, -1, -1)):
            for j_print, j in enumerate(range(n_tgt - 1, -1, -1)):
                ii = i_print + indices_start
                jj = j_print + indices_start
                print(f"  θ_{{{ii}{jj}}}^{L} = {W[i, j]:.{latex_precision}f}")

        # Bias legend in TOP→DOWN order:
        b_topdown = b[::-1]  # reverse so index 1 is top neuron
        print("  Bias (top→down):")
        for j_disp, val in enumerate(b_topdown, start=indices_start):
            print(f"    b^{L}_{j_disp} = {val:.{latex_precision}f}")
        print()

    def _latex_theta_block(L, W, b):
        """Pretty MathJax: θ_{ij}^L and b^L_j rendered in TOP→DOWN order."""
        n_src, n_tgt = W.shape
        # symbolic θ matrix (top=1 orientation)
        sym_rows = []
        for i_print in range(n_src):
            row_syms = []
            for j_print in range(n_tgt):
                ii = i_print + indices_start
                jj = j_print + indices_start
                row_syms.append(fr"\theta_{{{ii}{jj}}}^{{{L}}}")
            sym_rows.append(" & ".join(row_syms))
        sym_matrix = r"\\ ".join(sym_rows)
        display(HTML("<hr style='border:1px solid #999; margin:8px 0;'>"))
        display(Math(fr"\boldsymbol{{\Theta}}^{{{L}}} \in \mathbb{{R}}^{{{n_src}\times{n_tgt}}}"))
        display(Math(r"\begin{bmatrix}" + sym_matrix + r"\end{bmatrix}"))

        # numeric θ legend (top=1 orientation)
        leg_rows = []
        for i_print, i in enumerate(range(n_src - 1, -1, -1)):
            for j_print, j in enumerate(range(n_tgt - 1, -1, -1)):
                ii = i_print + indices_start
                jj = j_print + indices_start
                leg_rows.append(fr"\theta_{{{ii}{jj}}}^{{{L}}} &= {W[i, j]:.{latex_precision}f}")
        display(Math(r"\begin{aligned}" + r"\\ ".join(leg_rows) + r"\end{aligned}"))

        # Bias entries in TOP→DOWN order
        b_topdown = b[::-1]
        b_rows = [
            fr"b^{{{L}}}_{{{j_disp}}} &= {val:.{latex_precision}f}"
            for j_disp, val in enumerate(b_topdown, start=indices_start)
        ]
        display(Math(r"\begin{aligned}" + r"\\ ".join(b_rows) + r"\end{aligned}"))

    if print_weight_matrices:
        for L, (W, b) in enumerate(zip(weights_mats, biases_mats)):
            if latex_weight_print:
                _latex_theta_block(L, W, b)
            else:
                _console_theta_block(L, W, b)

    print(f"Total number of parameters (weights + biases): {total_params}")
    return weights_mats, biases_mats, total_params
