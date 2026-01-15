"""
3D prikaz embeddinga i izračun alignment score (dot product).
Riječi su na hrvatskom: Kralj, Kraljica, Pas.

Pokretanje:
    python plot_embeddings_3d.py
"""

import numpy as np
import matplotlib.pyplot as plt
import torch

def dot_product(a: np.ndarray, b: np.ndarray) -> float:
    """Skalarni (dot) produkt dvaju vektora."""
    return float(np.dot(a, b))


def plot_vectors_3d(vectors: dict, title: str = "3D prikaz embeddinga"):
    """
    3D plot vektora iz ishodišta (0,0,0) prema točkama u prostoru.
    Ne postavlja boje ručno (matplotlib default).
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(title)

    # Osi i mreža
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.set_zlabel("Dim 3")
    ax.grid(True)

    origin = np.zeros(3, dtype=float)

    # Plot i anotacije
    for name, vec in vectors.items():
        vec = np.asarray(vec, dtype=float)

        # Vektor kao linija iz ishodišta
        ax.plot([origin[0], vec[0]],
                [origin[1], vec[1]],
                [origin[2], vec[2]])

        # Točka na kraju vektora
        ax.scatter(vec[0], vec[1], vec[2])

        # Oznaka riječi
        ax.text(vec[0], vec[1], vec[2], f" {name}", fontsize=10)

    # Automatsko skaliranje osi da sve bude vidljivo
    all_vecs = np.array(list(vectors.values()), dtype=float)
    mins = np.minimum(all_vecs.min(axis=0), origin)
    maxs = np.maximum(all_vecs.max(axis=0), origin)

    # Malo "paddinga" radi čitljivosti
    pad = 0.05
    ax.set_xlim(mins[0] - pad, maxs[0] + pad)
    ax.set_ylim(mins[1] - pad, maxs[1] + pad)
    ax.set_zlim(mins[2] - pad, maxs[2] + pad)

    plt.show()

def softmax_all_alignment_scores_torch(embeddings: dict):
    """
    Compute all pairwise alignment scores, concatenate them,
    and apply PyTorch SoftMax globally.

    Parameters
    ----------
    embeddings : dict[str, np.ndarray | torch.Tensor]

    Returns
    -------
    scores : dict[(str, str), float]
        Raw dot product alignment scores
    weights : dict[(str, str), float]
        SoftMax-normalized attention weights
    """

    # Convert embeddings to torch tensors
    emb = {
        k: torch.tensor(v, dtype=torch.float32)
        for k, v in embeddings.items()
    }

    scores = {}
    score_list = []

    keys = list(emb.keys())

    # Compute all pairwise dot products
    for i in range(len(keys)):
        for j in range(len(keys)):
            pair = (keys[i], keys[j])
            score = torch.dot(emb[keys[i]], emb[keys[j]])
            scores[pair] = score.item()
            score_list.append(score)

    # Concatenate scores and apply SoftMax
    score_tensor = torch.stack(score_list)
    weights_tensor = torch.softmax(score_tensor, dim=0)

    # Map SoftMax values back to word pairs
    weights = {}
    idx = 0
    for pair in scores:
        weights[pair] = weights_tensor[idx].item()
        idx += 1

    return scores, weights