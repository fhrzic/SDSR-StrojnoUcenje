import time
import numpy as np
from tqdm.notebook import tqdm
from matplotlib import pyplot as plt
import torch

# Kreiranje dviju funkcija koje izračunavaju sumu kvadrata svih elemenata matrice A
def noVectorized(A):
    S = 0
    for i in range(A.shape[0]):
        for j in range(A.shape[0]):
            S += A[i,j] * A [i, j]
    return S

def vectorized(A):
    S = np.sum(np.multiply(A, A))
    return S

def vectorized_torch(A):
    S = A*A.sum()
    return S


def main(nLow: int = 1,
         nHigh: int = 224,
         reps: int = 100,
         rng_seed: int | None = 42,
         run_torch: bool = False):

    # Select device for torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Random seed
    rng = np.random.default_rng(rng_seed)

    # Prostor za iteratore
    sizes = np.arange(nLow, nHigh)
    # Spremnici za statistike
    vec_mean, vec_ci = [], []
    nvec_mean, nvec_ci = [], []
    tvec_mean, tvec_ci = [], []
    for n in tqdm(sizes):
        t_vec = np.empty(reps, dtype=float)
        t_nvec = np.empty(reps, dtype=float)
        t_tvec = np.empty(reps, dtype=float)

        # Ponavljanje experimenta
        for r in range(reps):
            A = rng.random((n, n))

            if run_torch:
                A_torch = torch.tensor(A, dtype= torch.float32, device=device)

                start = time.perf_counter()
                _ = vectorized_torch(A_torch)
                t_tvec[r] = time.perf_counter() - start

            start = time.perf_counter()
            _ = vectorized(A)
            t_vec[r] = time.perf_counter() - start
            start = time.perf_counter()
            _ = noVectorized(A)
            t_nvec[r] = time.perf_counter() - start



        # Statistika: sredina i 95% CI (normalna aproksimacija)
        vm = t_vec.mean()
        vs = t_vec.std(ddof=1)
        vh = 1.96 * vs / np.sqrt(reps)   # half-width CI

        nm = t_nvec.mean()
        ns = t_nvec.std(ddof=1)
        nh = 1.96 * ns / np.sqrt(reps)

        if run_torch:
            tvm = t_tvec.mean()
            tvs = t_tvec.std(ddof=1)
            tvh = 1.96 * tvs / np.sqrt(reps)

            tvec_mean.append(tvm); tvec_ci.append(tvh)

        vec_mean.append(vm); vec_ci.append(vh)
        nvec_mean.append(nm); nvec_ci.append(nh)

        # crtanje
    plt.figure(figsize=(10, 5))
    # vektorizirano
    plt.plot(sizes, vec_mean, label="Vektorizirano (sredina)", linewidth=1.5)
    plt.fill_between(sizes, np.array(vec_mean)-np.array(vec_ci), np.array(vec_mean)+np.array(vec_ci),
                     alpha=0.5, label="Vektorizirano 95% CI")
    # nevektorizirano
    plt.plot(sizes, nvec_mean, label="Nevektorizirano (sredina)", linewidth=1.5)
    plt.fill_between(sizes, np.array(nvec_mean)-np.array(nvec_ci), np.array(nvec_mean)+np.array(nvec_ci),
                     alpha=0.5, label="Nevektorizirano 95% CI")
    
    # torch
    if run_torch:
        plt.plot(sizes, tvec_mean, label=f"PyTorch ({device.type}) — vektorizirano (sredina)", linewidth=1.5)
        plt.fill_between(sizes,
                         np.array(tvec_mean) - np.array(tvec_ci),
                         np.array(tvec_mean) + np.array(tvec_ci),
                         alpha=0.5, label=f"PyTorch ({device.type}) — vektorizirano 95% CI")

    plt.ylabel("Sekunde")
    plt.xlabel("Veličina matrice (n za n×n)")
    plt.title("Vektorizirano vs. nevektorizirano — 100 ponavljanja po veličini (95% CI)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main(nLow=1, nHigh=224, reps=100, rng_seed=42)