# =========================
# Loss
# =========================
import numpy as np

def loss_and_dh_out(h_out, y, loss='mse', eps=1e-12):
    """
    h_out: (B, D) or (D,)
    y:     same total elements; reshaped to match h_out if needed.
    Returns: loss_scalar, dL/dh (same shape as h_out)
    """
    h = np.asarray(h_out, dtype=float)
    y = np.asarray(y, dtype=float)

    if y.size != h.size:
        raise ValueError(f"y size {y.size} must match h_out size {h.size}.")
    if y.shape != h.shape:
        y = y.reshape(h.shape)

    if h.ndim == 1:
        h = h[None, :]
        y = y[None, :]

    B, D = h.shape
    N = B * D

    if loss.lower() == 'mse':
        diff = h - y
        L = 0.5 * np.sum(diff * diff) / N
        grad = diff / N
        return L, (grad if h_out.ndim == 2 else grad[0])

    elif loss.lower() == 'bce':
        h_clipped = np.clip(h, eps, 1 - eps)
        L = np.mean(- y * np.log(h_clipped) - (1 - y) * np.log(1 - h_clipped))
        grad = (h - y) / (h_clipped * (1 - h_clipped)) / N
        return L, (grad if h_out.ndim == 2 else grad[0])

    else:
        raise ValueError("loss must be 'mse' or 'bce'")
