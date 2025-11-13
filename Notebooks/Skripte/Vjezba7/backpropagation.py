
# =========================
# Forward pass
# =========================
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from IPython.display import display, Math, HTML
import os
import sys

# Add project files
current_file_path = os.path.abspath(__file__)
main_dir =  os.path.dirname(current_file_path)
sys.path.append(main_dir)

from loss import *
from forward_izracun import * 
from forward_izracun import _ensure_2d, _act_prime_from_fh, _act_fn

# =========================
# Backward pass (TOP→BOTTOM), returns dΘ in printed orientation
# =========================
def backward_pass_topdown(
    inputs,
    weights_mats,   # Θ in printed orientation from draw_network
    y_true,
    biases=None,
    activations=None,
    loss='mse',
    return_internal=False
):
    # Forward (computes with top→bottom / printed orientation semantics)
    f_vals, h_vals, Wmats, biases_use, acts = forward_pass_trace_topdown(
        inputs, weights_mats, biases=biases, activations=activations, print_trace=False
    )

    # Shapes for loss – force 2D, then reshape y to match
    H_out2D = _ensure_2d(h_vals[-1])
    Y2D     = _ensure_2d(y_true)
    if Y2D.size != H_out2D.size:
        raise ValueError(f"y_true size {Y2D.size} must match output size {H_out2D.size}.")
    if Y2D.shape != H_out2D.shape:
        Y2D = Y2D.reshape(H_out2D.shape)

    loss_val, dL_dh_2D = loss_and_dh_out(H_out2D, Y2D, loss=loss)

    n_layers = len(Wmats) + 1
    dH = [None] * n_layers
    dF = [None] * n_layers
    dB = [None] * (n_layers - 1)
    dW_internal = [None] * (n_layers - 1)

    # final layer post-activation gradient
    dH[-1] = dL_dh_2D if np.asarray(h_vals[-1]).ndim == 2 else dL_dh_2D[0]

    # Backprop
    for L in range(n_layers - 1, 0, -1):
        fL = f_vals[L]
        h_prev = h_vals[L-1]
        act_code = acts[L]

        fL_2D     = _ensure_2d(fL)
        h_prev_2D = _ensure_2d(h_prev)
        dH_L_2D   = _ensure_2d(dH[L])

        a_prime_2D = _act_prime_from_fh(act_code, fL_2D, _act_fn(act_code)(fL_2D))
        dF_L_2D    = dH_L_2D * a_prime_2D
        dF[L]      = dF_L_2D if np.asarray(fL).ndim == 2 else dF_L_2D[0]

        # internal (computational) weight/bias grads (optional debug)
        dW_internal[L-1] = h_prev_2D.T @ dF_L_2D
        dB[L-1] = dF_L_2D.sum(axis=0)

        # propagate to previous layer
        dH_prev_2D = dF_L_2D @ Wmats[L-1].T
        dH[L-1] = dH_prev_2D if np.asarray(h_prev).ndim == 2 else dH_prev_2D[0]

    # --- Compute dTheta DIRECTLY in PRINTED orientation and CACHE ingredients ---
    dTheta_print = []
    cache_Hprev_print = []   # list of (B, n_in_prev) used for dTheta at each layer
    cache_dF_print     = []  # list of (B, n_out)     used for dTheta at each layer
    for L in range(1, n_layers):
        H_prev_print_2D = _ensure_2d(h_vals[L-1])  # printed top→bottom
        dF_print_2D     = _ensure_2d(dF[L])        # printed top→bottom
        cache_Hprev_print.append(H_prev_print_2D.copy())
        cache_dF_print.append(dF_print_2D.copy())
        dTheta_print.append(H_prev_print_2D.T @ dF_print_2D)

    results = {
        "loss": loss_val,
        "y_pred": h_vals[-1],
        "dTheta": dTheta_print,             # printed orientation
        "dbiases": dB,
        "dF": dF,                           # printed order
        "dH": dH,                           # printed order
        "activations": acts,
        # NEW: exact matrices used to build dTheta (for perfect expansion)
        "cache_Hprev_print": cache_Hprev_print,
        "cache_dF_print": cache_dF_print,
    }
    if return_internal:
        results["dW_internal"] = dW_internal
        results["W_internal"] = Wmats
    return results



# =========================
# Backward derivative TRACE (mirrors forward trace)
# =========================
import numpy as np
from IPython.display import display, Math

def backward_pass_trace_topdown(
    inputs,
    weights_mats,          # Θ from draw_network (printed orientation)
    y_true,
    biases=None,
    activations=None,
    loss='mse',
    indices_start=1,
    print_trace=True,
    latex_precision=6,     # digits to show in LaTeX numbers
):
    """
    Backward derivative trace in TOP→BOTTOM order with LaTeX-rendered equations.
    Requires backward_pass_topdown(...) to return:
      - dTheta (printed orientation),
      - dbiases, dF, dH in printed order,
      - cache_Hprev_print[L-1], cache_dF_print[L-1] used to build dTheta[L-1]
    """

    # Compute grads + caches (printed orientation)
    res = backward_pass_topdown(
        inputs=inputs,
        weights_mats=weights_mats,
        y_true=y_true,
        biases=biases,
        activations=activations,
        loss=loss,
        return_internal=True
    )

    if not print_trace:
        return res

    # Context values (not used for the expansion math)
    f_vals, h_vals, W_internal, biases_use, acts = forward_pass_trace_topdown(
        inputs=inputs, weights_mats=weights_mats, biases=biases, activations=activations, print_trace=False
    )

    def _to2d(a):
        a = np.asarray(a)
        return a if a.ndim == 2 else a[None, :]

    B = _to2d(inputs).shape[0]
    n_layers = len(W_internal) + 1

    print("\n======================")
    print(" BACKWARD DERIVATIVE TRACE (top→bottom, LaTeX)")
    print("======================")

    for L in range(n_layers - 1, 0, -1):
        print(f"\n=== Layer {L} ===")

        # Pretty arrays (already in printed order)
        dH_flat = np.asarray(res["dH"][L]).ravel()
        dF_flat = np.asarray(res["dF"][L]).ravel()

        # 1) dL/dh^L (post-activation)
        print("dL/dh (post-activation):")
        dh_exprs = [
            fr"\frac{{\partial L}}{{\partial h_{{{L}{j}}}}} = {val:.{latex_precision}f}"
            for j, val in enumerate(dH_flat, start=indices_start)
        ]
        display(Math(r"\\ ".join(dh_exprs)))

        # 2) dL/df^L (pre-activation)
        print("dL/df (pre-activation):")
        df_exprs = [
            fr"\frac{{\partial L}}{{\partial f_{{{L}{j}}}}} = {val:.{latex_precision}f}"
            for j, val in enumerate(dF_flat, start=indices_start)
        ]
        display(Math(r"\\ ".join(df_exprs)))
        display(Math(r"\text{since } \frac{\partial L}{\partial f} = \frac{\partial L}{\partial h}\cdot a'(f)"))

        # 3) dL/dΘ^{(L-1)} in printed orientation
        print("dL/dΘ (weights) in printed orientation (top row=1, left col=1):")
        dTheta_print = res["dTheta"][L-1]
        n_in_prev, n_out = dTheta_print.shape

        # Print as rows
        for i_disp in range(indices_start, indices_start + n_in_prev):
            row_terms = []
            for j_disp in range(indices_start, indices_start + n_out):
                gij = dTheta_print[i_disp - indices_start, j_disp - indices_start]
                row_terms.append(fr"\frac{{\partial L}}{{\partial \Theta^{{{L-1}}}_{{{i_disp}{j_disp}}}}} = {gij:.{latex_precision}f}")
            display(Math(r"\quad, \; ".join(row_terms)))

        # 3b) Single-sample expansion — use the EXACT cached tensors that built dΘ
        if B == 1:
            print("  (single-sample expansion: ∂L/∂Θ = h^{L-1} · ∂L/∂f^L, using cached tensors)")
            Hprev_c = res["cache_Hprev_print"][L-1][0]  # shape (n_in_prev,)
            dF_c    = res["cache_dF_print"][L-1][0]     # shape (n_out,)

            for i_disp in range(indices_start, indices_start + n_in_prev):
                for j_disp in range(indices_start, indices_start + n_out):
                    hterm = Hprev_c[i_disp - indices_start]
                    dfterm = dF_c[j_disp - indices_start]
                    term = hterm * dfterm
                    gij  = dTheta_print[i_disp - indices_start, j_disp - indices_start]
                    # Show the numeric outer-product equality
                    display(Math(
                        fr"\Theta^{{{L-1}}}_{{{i_disp}{j_disp}}}: "
                        fr"h^{{{L-1}}}_{{{i_disp}}}\cdot \frac{{\partial L}}{{\partial f_{{{L}{j_disp}}}}}"
                        fr"= {hterm:.{latex_precision}f}\times {dfterm:.{latex_precision}f}"
                        fr"= {term:.{latex_precision}f} \;\; \overset{{\small ?}}={{}}\; {gij:.{latex_precision}f}"
                    ))
                    # Ensure perfect match
                    assert np.allclose(term, gij, atol=1e-10), \
                        f"Mismatch at Θ^{L-1}_{{{i_disp}{j_disp}}}: term={term}, printed={gij}"
        else:
            # Optional: for batches, show sample #0 expansion approximately
            print("  (batch) showing expansion for sample #0:")
            Hprev_c = res["cache_Hprev_print"][L-1][0]
            dF_c    = res["cache_dF_print"][L-1][0]
            for i_disp in range(indices_start, indices_start + n_in_prev):
                for j_disp in range(indices_start, indices_start + n_out):
                    term = Hprev_c[i_disp - indices_start] * dF_c[j_disp - indices_start]
                    display(Math(
                        fr"\Theta^{{{L-1}}}_{{{i_disp}{j_disp}}}: "
                        fr"h^{{{L-1}}}_{{{i_disp}}}\cdot \frac{{\partial L}}{{\partial f_{{{L}{j_disp}}}}}"
                        fr"\approx {term:.{latex_precision}f}"
                    ))

        # 4) dL/db^{(L-1)} (sum over batch)
        print("dL/db (biases) (sum over batch):")
        # If your forward pass has NO biases, these are logically zero/unused; still showing for completeness.
        db = np.asarray(res["dbiases"][L-1]).ravel()
        db_exprs = [
            fr"\frac{{\partial L}}{{\partial b^{{{L-1}}}_{{{j}}}}} = {val:.{latex_precision}f}"
            for j, val in enumerate(db, start=indices_start)
        ]
        display(Math(r"\\ ".join(db_exprs)))

        # 5) dL/dh^{(L-1)} relation
        print("dL/dh (previous layer) relation:")
        dH_prev_flat = np.asarray(res["dH"][L-1]).ravel()
        rel_exprs = [
            fr"\frac{{\partial L}}{{\partial h_{{{L-1}{i}}}}} = "
            fr"\sum_j \frac{{\partial L}}{{\partial f_{{{L}j}}}}\;\Theta^{{{L-1}}}_{{{i}j}}"
            fr" \;=\; {val:.{latex_precision}f}"+r"\\\\"
            for i, val in enumerate(dH_prev_flat, start=indices_start)
        ]
        display(Math(r"\\ ".join(rel_exprs)))

    print("\n——— Summary ———")
    display(Math(fr"\text{{Loss}} \;=\; {res['loss']:.8f}"))
    # Show y_pred rounded (display-only)
    ypred = np.asarray(res["y_pred"]).ravel()
    display(Math(fr"\hat{{y}} \;=\; {np.round(ypred, 2)}"))
    return res



