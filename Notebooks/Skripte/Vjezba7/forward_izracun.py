# =========================
# Forward pass (TOP→BOTTOM) with f/h trace — bias flip fixed
# =========================
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from IPython.display import display, Math, HTML

# =========================
# Activations (+ derivatives) & utilities
# =========================
def _normalize_activations(layer_sizes, activations):
    L = len(layer_sizes)
    if activations is None:
        return ['-'] + ['r'] * max(0, L - 2) + ['s']
    if len(activations) == L - 1:
        return ['-'] + list(activations)
    if len(activations) != L:
        raise ValueError(f"`activations` must have length {L} or {L-1}. Got {len(activations)}.")
    return activations

def _act_fn(code):
    c = (code or '-').strip().lower()
    if c in ('-', 'none', 'linear', 'identity'):
        return lambda x: x
    if c in ('r', 'relu'):
        return lambda x: np.maximum(x, 0.0)
    if c in ('s', 'sigmoid'):
        return lambda x: 1.0 / (1.0 + np.exp(-x))
    if c in ('t', 'tanh'):
        return lambda x: np.tanh(x)
    return lambda x: x

def _act_prime_from_fh(code, f, h):
    c = (code or '-').strip().lower()
    if c in ('-', 'none', 'linear', 'identity'):
        return np.ones_like(f)
    if c in ('r', 'relu'):
        return (f > 0).astype(f.dtype)
    if c in ('s', 'sigmoid'):
        return h * (1.0 - h)
    if c in ('t', 'tanh'):
        return 1.0 - h**2
    return np.ones_like(f)

def _ensure_2d(x):
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[None, :]
    if x.ndim != 2:
        raise ValueError("Input must be 1D or 2D (B, D).")
    return x

def _to2d(a):
    a = np.asarray(a)
    return a if a.ndim == 2 else a[None, :]

# =========================
# Forward pass (TOP→BOTTOM) with LaTeX trace
# =========================
def forward_pass_trace_topdown(
    inputs,
    weights_mats,      # Θ printed orientation (top row = top neuron)
    biases=None,       # bias vectors in printed orientation (top→down)
    activations=None,
    print_trace=True,
    indices_start=1,
    *,
    latex_trace=True,  # pretty LaTeX output in Jupyter
    hr=True            # horizontal rule between layers
):
    """
    Forward pass that computes f and h values using TOP→BOTTOM neuron ordering,
    exactly matching printed Θ matrices (top row = top neuron).

    Fix: biases are flipped to computational order together with Θ.
    """

    # 1) Convert printed Θ (top-first) to internal computational order (bottom-first)
    Wmats = [W[::-1, ::-1] for W in weights_mats]

    # 2) Prepare biases and flip them the SAME way (top-first -> bottom-first)
    if biases is None:
        biases = [np.zeros(W.shape[1]) for W in Wmats]
    # flip each bias vector so index aligns with flipped columns
    Bmats = [b[::-1] for b in biases]

    # 3) Shapes/activations
    n_layers = len(Wmats) + 1
    layer_sizes = [Wmats[0].shape[0]] + [W.shape[1] for W in Wmats]
    acts = _normalize_activations(layer_sizes, activations)

    # 4) Inputs in top-first order (matches your printed convention)
    H = _ensure_2d(inputs)  # (B, n_in)

    # 5) Run forward pass with flipped W and flipped b
    h_values = [H.copy()]   # layer 0 post-activation (inputs)
    f_values = [None]       # no pre-activation for input layer

    for L, (W, b) in enumerate(zip(Wmats, Bmats), start=1):
        Z = H @ W + b              # pre-activation
        H = _act_fn(acts[L])(Z)    # post-activation using target layer's activation
        f_values.append(Z)
        h_values.append(H)

    # 6) Squeeze for single-sample inputs (keep old behavior)
    single = (np.asarray(inputs).ndim == 1)
    if single:
        f_values = [None if f is None else f[0] for f in f_values]
        h_values = [h[0] for h in h_values]

    # 7) Pretty LaTeX trace (optional)
    if print_trace:
        if latex_trace:
            display(HTML("<div style='font-weight:700; font-size:14px;'>Forward trace (top→bottom)</div>"))

            # Layer 0 (inputs)
            vals = [fr"h^{{(0)}}_{{{j}}} = {h_values[0][j-1]:.4f}"
                    for j in range(indices_start, indices_start + h_values[0].shape[-1])]
            display(Math(r"\text{Input }(h^{(0)}):\quad " + r",\; ".join(vals)))

            if hr:
                display(HTML("<hr style='border:1px solid #ccc; margin:6px 0;'>"))

            # Layers 1..L
            for L in range(1, n_layers):
                d_out = h_values[L].shape[-1]

                # Equations for layer L
                display(Math(
                    fr"\textbf{{Layer }}{L}:"
                    r"\quad f^{(" + f"{L}" + r")} = h^{(" + f"{L-1}" + r")}\,\Theta^{(" + f"{L-1}" + r")} + b^{(" + f"{L-1}" + r")},"
                    r"\qquad h^{(" + f"{L}" + r")} = a_{" + f"{L}" + r"}\!\left(f^{(" + f"{L}" + r")}\right)"
                ))

                # Component-wise values
                f_list = [fr"f^{{({L})}}_{{{j}}} = {f_values[L][j-1]:.4f}"
                          for j in range(indices_start, indices_start + d_out)]
                h_list = [fr"h^{{({L})}}_{{{j}}} = {h_values[L][j-1]:.4f}"
                          for j in range(indices_start, indices_start + d_out)]
                display(Math(r"\text{Pre-activation }f:\quad " + r",\; ".join(f_list)))
                display(Math(r"\text{Post-activation }h:\quad " + r",\; ".join(h_list)
                            + fr"\qquad (\text{{activation}}= {acts[L]})"))

                if hr and L < n_layers - 1:
                    display(HTML("<hr style='border:1px solid #eee; margin:6px 0;'>"))
        else:
            # Plain-text fallback
            for L in range(n_layers):
                print(f"\n=== Layer {L} ===")
                if L == 0:
                    print("Input (h^(0)):")
                    for j, val in enumerate(h_values[L], start=indices_start):
                        print(f"h^{0}_{j} = {val:.4f}")
                else:
                    print("Pre-activation (f):")
                    for j, val in enumerate(f_values[L], start=indices_start):
                        print(f"f^{L}_{j} = {val:.4f}")
                    print("Post-activation (h):")
                        # show activation code for the layer
                    for j, val in enumerate(h_values[L], start=indices_start):
                        print(f"h^{L}_{j} = {val:.4f}   (activation = {acts[L]})")

    # Return forward caches + flipped weights/biases actually used in compute
    return f_values, h_values, Wmats, Bmats, acts
