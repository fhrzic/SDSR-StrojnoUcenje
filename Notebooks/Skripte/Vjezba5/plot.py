"""
Niz funkcija za crtanje podataka
"""

# Knjižnica
import sys
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from ipywidgets import FloatSlider, HBox, VBox, Output, Label, Layout
from IPython.display import display

# Add project files
current_file_path = os.path.abspath(__file__)
main_dir =  os.path.dirname(current_file_path)
print(main_dir)
sys.path.append(main_dir)

from dataloader import *
from models import *
from loss_funkcije import *
from loss_funkcije_binarne import *

def create_scatter_x_band_widget(path_to_csv: str,
                                             var1: str = "temp",   # x_label
                                             var2: str = "cnt"):   # y_label
    """
    Interaktivni widget:
      - Slider za numerički x (kontinuirano)
      - Slider za 'varijancu' σ (širina pojasa oko x: [x-σ, x+σ])
      - Lijevo: cijeli scatter + vertikalni pojas i highlight točaka u pojasu
      - Desno: samo točke iz pojasa + Normalna aproksimacija Pr(y|x={x_value}) s OSJENČANIM područjem ispod krivulje

    Podatke dohvaća iz:
        bikeRentalDataset(path_to_csv=..., input_label=var1, target_label=var2)
        ordinary_dataloader(dataset=..., batch_size=1)

    Args:
        path_to_csv: putanja do CSV-a
        var1: naziv ulaznog stupca (x)
        var2: naziv ciljnog stupca (y)
    """
    # --- Fiksna tolerancija za float usporedbe (ne izlažemo korisniku) ---
    _rtol = 1e-12
    _atol_min = 1e-12  # apsolutni minimum
    # ------------------------------------

    assert path_to_csv is not None, "path_to_csv must not be None."
    assert os.path.exists(path_to_csv), f"Path does not exist: {path_to_csv}"

    # -----------------------
    # Dohvat podataka iz DL
    # -----------------------
    _dataset = bikeRentalDataset(path_to_csv=path_to_csv,
                                 input_label=var1,
                                 target_label=var2,
                                 normalizacija=True)

    _dataloader = ordinary_dataloader(dataset=_dataset, batch_size=1)

    _xs, _ys = [], []
    for _x_batch, _y_batch in _dataloader:
        # očekujemo skalarne tenzore po sample-u → squeeze -> item
        _xs.append(float(_x_batch.squeeze().item()))
        _ys.append(float(_y_batch.squeeze().item()))

    _x = np.asarray(_xs, dtype=float)
    _y = np.asarray(_ys, dtype=float)

    if _x.size == 0 or _y.size == 0:
        raise ValueError("Dataloader vratio je prazan skup podataka.")

    # Preimenujemo zbog ostatka koda
    _x_vals_all = _x
    _y_vals_all = _y

    _x_label = var1
    _y_label = var2

    _x_min, _x_max = float(np.min(_x_vals_all)), float(np.max(_x_vals_all))
    _y_min, _y_max = float(np.min(_y_vals_all)), float(np.max(_y_vals_all))
    _x_range = (_x_max - _x_min) if _x_max > _x_min else 1.0
    _y_range = (_y_max - _y_min) if _y_max > _y_min else 1.0

    _x_pad = 0.02 * _x_range
    _y_pad = 0.05 * _y_range

    # σ (varijanca/širina) inicijalno 5% raspona x; max 30% raspona
    _sigma_init = 0.05 * _x_range
    _sigma_max = 0.30 * _x_range
    _sigma_min = max(_atol_min, 0.001 * _x_range)  # da nikad ne bude nula

    # Widgeti
    _x_slider = FloatSlider(
        value=float(np.median(_x_vals_all)),
        min=_x_min, max=_x_max, step=_x_range/1000.0,
        description=f"{_x_label}:",
        readout_format=".4f",
        layout=Layout(width="95%")
    )
    _sigma_slider = FloatSlider(
        value=_sigma_init,
        min=_sigma_min, max=_sigma_max, step=_x_range/1000.0,
        description="σ (pojas):",
        readout_format=".4f",
        layout=Layout(width="95%"),
        continuous_update=True
    )
    _out = Output()

    def _normal_pdf(_y_grid: np.ndarray, _mu: float, _sigma: float) -> np.ndarray:
        # Stabilizacija: ako je sigma ~ 0, nametni minimalnu vrijednost
        _sigma_safe = max(_sigma, 1e-12)
        _coef = 1.0 / (np.sqrt(2.0 * np.pi) * _sigma_safe)
        _z = (_y_grid - _mu) / _sigma_safe
        return _coef * np.exp(-0.5 * _z * _z)

    def _redraw(_x_center: float, _sigma: float):
        with _out:
            _out.clear_output(wait=True)
            _fig, (_ax_all, _ax_sel) = plt.subplots(1, 2, figsize=(11, 4.8))

            # Lijevi graf: cijeli scatter
            _ax_all.scatter(_x_vals_all, _y_vals_all, alpha=0.6)

            # Pojas [x-σ, x+σ]
            _x_left, _x_right = _x_center - _sigma, _x_center + _sigma
            _ax_all.axvline(_x_left, linestyle="--", alpha=0.6)
            _ax_all.axvline(_x_right, linestyle="--", alpha=0.6)
            _ax_all.fill_betweenx([_y_min - _y_pad, _y_max + _y_pad],
                                  _x_left, _x_right, alpha=0.12)

            # Mask točaka u pojasu
            _mask_band = (_x_vals_all >= _x_left) & (_x_vals_all <= _x_right)
            _sel_x = _x_vals_all[_mask_band]
            _sel_y = _y_vals_all[_mask_band]

            if _sel_y.size > 0:
                # highlight odabranih točaka
                _ax_all.scatter(_sel_x, _sel_y, s=50, edgecolor="k", linewidths=0.8)

            _ax_all.set_title(f"{_x_label} vs {_y_label}")
            _ax_all.set_xlabel(_x_label)
            _ax_all.set_ylabel(_y_label)
            _ax_all.set_xlim(_x_min - _x_pad, _x_max + _x_pad)
            _ax_all.set_ylim(_y_min - _y_pad, _y_max + _y_pad)
            _ax_all.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

            # Desni graf: samo točke u pojasu + Normalna aproksimacija Pr(y|x) (krivulja + OSJENA)
            if _sel_y.size > 0:
                # Točke (uz minimalni jitter da se vide preklapanja)
                _jitter = (np.random.rand(_sel_y.size) - 0.5) * 0.01 * _x_range
                _ax_sel.scatter(_sel_x + _jitter, _sel_y, alpha=0.9, label="Podaci")

                # Procjena μ i σ iz selektiranih y-vrijednosti
                _mu = float(np.mean(_sel_y))
                _sigma_y = float(np.std(_sel_y, ddof=1)) if _sel_y.size > 1 else 0.0

                # y-grid za krivulju
                _y_lo = min(_y_min - _y_pad, np.min(_sel_y) - 0.1 * _y_range)
                _y_hi = max(_y_max + _y_pad, np.max(_sel_y) + 0.1 * _y_range)
                _y_grid = np.linspace(_y_lo, _y_hi, 400)

                # Normal PDF po y
                _pdf = _normal_pdf(_y_grid, _mu, _sigma_y)

                # Skala za crtanje PDF-a kao horizontalni odmak od _x_center
                # (normaliziramo na [0, 1] pa skaliramo na širinu 3*_x_pad)
                _pdf_max = float(np.max(_pdf)) if np.max(_pdf) > 0 else 1.0
                _scale = 3.0 * _x_pad
                _x_curve = _x_center + (_pdf / _pdf_max) * _scale

                # OSJENČAJ PODRUČJE ISPOD KRIVULJE
                _ax_sel.fill_betweenx(_y_grid, _x_center, _x_curve, alpha=0.25)

                # Krivulja Pr(y|x=...)
                _label_curve = f"Pr(y|x={_x_center:.4f})"
                _ax_sel.plot(_x_curve, _y_grid, linewidth=2.0, label=_label_curve)
                _ax_sel.axvline(_x_center, linestyle="--", alpha=0.6)

                _ax_sel.set_title(f"Točke u pojasu: [{_x_left:.4f}, {_x_right:.4f}]")
                _ax_sel.legend(loc="best")
            else:
                _ax_sel.text(0.5, 0.5, "Nema točaka u pojasu", ha="center", va="center",
                             transform=_ax_sel.transAxes)

            _ax_sel.set_xlabel(_x_label)
            _ax_sel.set_ylabel(_y_label)
            _ax_sel.set_ylim(_y_min - _y_pad, _y_max + _y_pad)
            # Fokus x-osi oko pojasa i normalne krivulje
            _ax_sel.set_xlim(_x_left - 2*_x_pad, _x_right + 4*_x_pad)
            _ax_sel.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

            plt.tight_layout()
            plt.show()

    def _on_change(_change):
        if _change["name"] == "value":
            _redraw(_x_slider.value, _sigma_slider.value)

    _x_slider.observe(_on_change, names="value")
    _sigma_slider.observe(_on_change, names="value")

    # inicijalni prikaz
    _redraw(_x_slider.value, _sigma_slider.value)

    _ui = VBox([
        Label("Odaberi numerički x i širinu pojasa (σ). Desno: Pr(y|x) kao Normalna krivulja (osjenčano)."),
        _x_slider,
        _sigma_slider,
        _out
    ])
    display(_ui)

@torch.no_grad()
def model_over_dataloader_widget(path_to_csv: str):
    var1 = "temp"
    var2 = "cnt"
    n_points = 400

    # Load dataset + loader
    dataset = bikeRentalDataset(path_to_csv=path_to_csv,
                                input_label=var1,
                                target_label=var2,
                                normalizacija=True)
    dataloader = ordinary_dataloader(dataset=dataset, batch_size=1)

    # Extract all data
    xs, ys = [], []
    for xb, yb in dataloader:
        xs.append(float(xb.squeeze().item()))
        ys.append(float(yb.squeeze().item()))
    xs = np.asarray(xs, float)
    ys = np.asarray(ys, float)

    # Precompute input grid
    x_line = np.linspace(xs.min(), xs.max(), n_points).astype(np.float32)
    x_t = torch.from_numpy(x_line)

    # Sliders
    psi0_slider = FloatSlider(value=0.0, min=-1.5, max=1.0, step=0.000005,
                              description="ψ₀:", continuous_update=True)

    sigma_slider = FloatSlider(value=0.05, min=0.0, max=0.5, step=0.0005,
                               description="σ:", continuous_update=True)

    out = Output()

    def redraw(change=None):
        with out:
            out.clear_output(wait=True)

            # Instantiate model with current psi0
            model = model_1_3_1(psi0=float(psi0_slider.value))

            # Forward
            y_pred = model(x_t).cpu().numpy().squeeze()

            # Compute band
            sigma = float(sigma_slider.value)
            y_low  = y_pred - sigma
            y_high = y_pred + sigma

            # Plot
            plt.figure(figsize=(8,5))
            plt.scatter(xs, ys, alpha=0.6, label="Data")

            # --- shaded band around model prediction ---
            plt.fill_between(x_line, y_low, y_high,
                             color="red", alpha=0.20, label=f"±σ (σ={sigma:.3f})")

            # model curve
            plt.plot(x_line, y_pred, linewidth=2.5, color="red", label=f"Model (ψ₀={psi0_slider.value:.3f})")

            plt.xlabel(var1)
            plt.ylabel(var2)
            plt.title("Model 1-3-1 vs. Data")
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.legend()
            plt.tight_layout()
            plt.show()

    psi0_slider.observe(redraw, names="value")
    sigma_slider.observe(redraw, names="value")
    redraw()

    display(VBox([
        Label("Adjust ψ₀ and σ below:"),
        psi0_slider,
        sigma_slider,
        out
    ]))


@torch.no_grad()
def plot_losses_over_psi0(path_to_csv: str,
                          sigma: float = 0.35,
                          normalizacija: bool = True):

    _var1, _var2 = "temp", "cnt"
    _dataset = bikeRentalDataset(path_to_csv=path_to_csv,
                                 input_label=_var1,
                                 target_label=_var2,
                                 normalizacija=normalizacija)
    _dataloader = ordinary_dataloader(dataset=_dataset, batch_size=1)

    _xs, _ys = [], []
    for _x_batch, _y_batch in _dataloader:
        _xs.append(float(_x_batch.squeeze().item()))
        _ys.append(float(_y_batch.squeeze().item()))

    _x = np.asarray(_xs, dtype=float)
    _y_train = np.asarray(_ys, dtype=float)

    # Dense sweep for ψ0
    _psi0_values = np.linspace(-1.0, 1.0, 401)

    _sse_values, _lik_values, _nll_values = [], [], []
    _x_t = torch.from_numpy(_x.astype(np.float32))

    for _psi0 in _psi0_values:
        _model = model_1_3_1(psi0=float(_psi0))
        _mu_pred = _model(_x_t).cpu().numpy().squeeze()

        _sse_values.append(compute_sum_of_squares(_y_train, _mu_pred))
        _lik_values.append(compute_likelihood(_y_train, _mu_pred, sigma=sigma))
        _nll_values.append(compute_negative_log_likelihood(_y_train, _mu_pred, sigma=sigma))

    _sse_values = np.array(_sse_values)
    _lik_values = np.array(_lik_values)
    _nll_values = np.array(_nll_values)

    # --- Best ψ0 and values ---
    _best_psi_sse = _psi0_values[np.argmin(_sse_values)]
    _best_sse = np.min(_sse_values)

    _best_psi_lik = _psi0_values[np.argmax(_lik_values)]
    _best_lik = np.max(_lik_values)

    _best_psi_nll = _psi0_values[np.argmin(_nll_values)]
    _best_nll = np.min(_nll_values)

    print("Best values:")
    print(f"  SSE:  best ψ₀ = { _best_psi_sse:.4f},  SSE = { _best_sse:.6f}")
    print(f"  L  :  best ψ₀ = { _best_psi_lik:.4f},  L   = { _best_lik:.6e}")
    print(f"  NLL:  best ψ₀ = { _best_psi_nll:.4f},  NLL = { _best_nll:.6f}")

    # --- Plot ---
    _fig, _ax = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)

    # SSE
    _ax[0].plot(_psi0_values, _sse_values)
    _ax[0].axhline(_best_sse, color="red", linestyle="--")
    _ax[0].axvline(_best_psi_sse, color="red", linestyle="--")
    _ax[0].set_title(f"SSE (best ψ₀={_best_psi_sse:.3f})")

    # Likelihood
    _ax[1].plot(_psi0_values, _lik_values)
    _ax[1].axhline(_best_lik, color="red", linestyle="--")
    _ax[1].axvline(_best_psi_lik, color="red", linestyle="--")
    _ax[1].set_title(f"Likelihood (best ψ₀={_best_psi_lik:.3f})")

    # NLL
    _ax[2].plot(_psi0_values, _nll_values)
    _ax[2].axhline(_best_nll, color="red", linestyle="--")
    _ax[2].axvline(_best_psi_nll, color="red", linestyle="--")
    _ax[2].set_title(f"NLL (best ψ₀={_best_psi_nll:.3f})")

    for _a in _ax:
        _a.set_xlabel("ψ₀")
        _a.set_ylabel("Loss")
        _a.grid(True, linestyle="--", alpha=0.5)

    plt.show()


def create_plot_b_widget(path_to_csv: str,
                       input_label: str = "glucose_postprandial",
                       target_label: str = "diagnosed_diabetes",
                       normalizacija: bool = True,
                       device: str = "cpu"):
    """
    Interaktivni widget:
      - Učitava podatke
      - Evaluira model na uniformnoj mreži x∈[0,1]
      - Prikazuje dvije figure: bez sigmoid i sa sigmoid
      - Linije modela su ZELENE
      - Slider kontrolira psi0 parametar
      - Desni graf ima vodoravnu liniju y=0.5
    """
    assert os.path.exists(path_to_csv), f"CSV path ne postoji: {path_to_csv}"

    # -------------------
    # Učitaj podatke za scatter
    # -------------------
    _dataset = bikeRentalDataset(path_to_csv=path_to_csv,
                                 input_label=input_label,
                                 target_label=target_label,
                                 normalizacija=normalizacija)
    _dataloader = ordinary_dataloader(dataset=_dataset, batch_size=1)

    _xs, _ys = [], []
    for _x_batch, _y_batch in _dataloader:
        _xs.append(float(_x_batch.squeeze().item()))
        _ys.append(float(_y_batch.squeeze().item()))
    _x_scatter = np.asarray(_xs, dtype=float)
    _y_scatter = np.asarray(_ys, dtype=float)

    # -------------------
    # Grid za evaluaciju modela
    # -------------------
    _x_line = np.linspace(0.0, 1.0, 200, dtype=np.float32)
    _x_line_t = torch.from_numpy(_x_line).view(-1, 1).to(device)

    # -------------------
    # Widgeti
    # -------------------
    _psi0_slider = FloatSlider(
        value=0.202, min=-5.0, max=5.0, step=0.001,
        description="psi0:", readout_format=".3f",
        continuous_update=True, layout=Layout(width="95%")
    )
    _out = Output()

    # -------------------
    # Crtanje (handler)
    # -------------------
    def _redraw(_psi0: float):
        with _out:
            _out.clear_output(wait=True)

            # Model bez sigmoid
            _model_raw = model_1_3_1_b(psi0=_psi0, apply_sigmoid=False).to(device)
            with torch.no_grad():
                _pred_raw = _model_raw(_x_line_t).squeeze().detach().cpu().numpy()

            # Model sa sigmoid
            _model_sig = model_1_3_1_b(psi0=_psi0, apply_sigmoid=True).to(device)
            with torch.no_grad():
                _pred_sig = _model_sig(_x_line_t).squeeze().detach().cpu().numpy()

            # Plotovi
            _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(12, 4))

            # Bez sigmoid — zelena linija
            _ax1.scatter(_x_scatter, _y_scatter, alpha=0.8, label="Data")
            _ax1.plot(_x_line, _pred_raw, linewidth=2.0, label="Model (raw)", color="green")
            _ax1.set_title("Model Output (Without Sigmoid)")
            _ax1.set_xlabel(input_label + " (scaled 0–1 grid)")
            _ax1.set_ylabel(target_label)
            _ax1.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
            _ax1.legend()

            # Sa sigmoid — zelena linija + y=0.5
            _ax2.scatter(_x_scatter, _y_scatter, alpha=0.8, label="Data")
            _ax2.plot(_x_line, _pred_sig, linewidth=2.0, label="Model (sigmoid)", color="green")
            _ax2.axhline(0.5, linestyle="--", linewidth=1.2, alpha=0.9, label="y = 0.5")
            _ax2.set_title("Model Output (With Sigmoid)")
            _ax2.set_xlabel(input_label + " (scaled 0–1 grid)")
            _ax2.set_ylabel("Probability")
            # Poželjno za binarnu probu: fokus oko [0,1]
            _ax2.set_ylim(-0.05, 1.05)
            _ax2.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
            _ax2.legend()

            plt.tight_layout()
            plt.show()

    # Inicijalni prikaz
    _redraw(_psi0_slider.value)

    # Poveži slider
    def _on_change(_change):
        if _change["name"] == "value":
            _redraw(_change["new"])

    _psi0_slider.observe(_on_change, names="value")

    _ui = VBox([
        Label("Podesi psi0 i promatraj efekt na model (zelene linije). Desno: y=0.5."),
        _psi0_slider,
        _out
    ])
    display(_ui)


@torch.no_grad()
def plot_losses_b_over_psi0(path_to_csv: str,
                          normalizacija: bool = True,
                          psi0_min: float = -1.0,
                          psi0_max: float = 1.0,
                          psi0_steps: int = 401):
    """
    Sweepa psi0, računa Likelihood i Negative Log-Likelihood za Bernoulli model
    (sa sigmoid izlazom) i crta dvije krivulje. Target mora biti binaran (0/1).
    """
    assert os.path.exists(path_to_csv), f"CSV path ne postoji: {path_to_csv}"

    # --- Varijable skupa (očekuje se binarni target!) ---
    _var1, _var2 = "glucose_postprandial", "diagnosed_diabetes"  # promijeni ako koristiš drugi CSV
    _dataset = bikeRentalDataset(path_to_csv=path_to_csv,
                                 input_label=_var1,
                                 target_label=_var2,
                                 normalizacija=normalizacija)
    _dataloader = ordinary_dataloader(dataset=_dataset, batch_size=1)

    # --- Učitaj podatke ---
    _xs, _ys = [], []
    for _x_batch, _y_batch in _dataloader:
        _xs.append(float(_x_batch.squeeze().item()))
        _ys.append(float(_y_batch.squeeze().item()))

    _x = np.asarray(_xs, dtype=float)
    _y_train = np.asarray(_ys, dtype=float)

    # --- Provjera binarnosti targeta (0/1) ---
    _unique_targets = np.unique(_y_train)
    if not np.all(np.isin(_unique_targets, [0.0, 1.0])):
        raise ValueError(
            f"Target '{_var2}' mora biti binaran (0/1) za Bernoulli likelihood. "
            f"Pronađene vrijednosti: {_unique_targets}"
        )

    # --- Sweep po psi0 ---
    _psi0_values = np.linspace(psi0_min, psi0_max, psi0_steps, dtype=float)
    _lik_values, _nll_values = [], []

    # model ulazi su x∈[0,1], pa evaluiramo na odgovarajućem skaliranju x-a:
    # ovdje pretpostavljamo da je bikeRentalDataset već normalizirao input ako je normalizacija=True.
    _x_t = torch.from_numpy(_x.astype(np.float32)).view(-1, 1)

    for _psi0 in _psi0_values:
        # Model sa sigmoid da dobije λ∈[0,1]
        _model = model_1_3_1_b(psi0=float(_psi0), apply_sigmoid=True)
        _lambda_pred = _model(_x_t).cpu().numpy().squeeze()  # shape (N,)

        # Likelihood i NLL koriste isključivo tvoje funkcije
        _lik = compute_likelihood_b(_y_train, _lambda_pred)
        _nll = compute_negative_log_likelihood_b(_y_train, _lambda_pred)

        _lik_values.append(_lik)
        _nll_values.append(_nll)

    _lik_values = np.asarray(_lik_values, dtype=float)
    _nll_values = np.asarray(_nll_values, dtype=float)

    # --- Najbolje vrijednosti ---
    _best_psi_lik = _psi0_values[np.argmax(_lik_values)]
    _best_lik = np.max(_lik_values)
    _best_psi_nll = _psi0_values[np.argmin(_nll_values)]
    _best_nll = np.min(_nll_values)

    print("Best values (Bernoulli):")
    print(f"  Likelihood: best ψ₀ = {_best_psi_lik:.4f},  L   = {_best_lik:.6e}")
    print(f"  NLL       : best ψ₀ = {_best_psi_nll:.4f},  NLL = {_best_nll:.6f}")

    # --- Plot ---
    _fig, _ax = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)

    # Likelihood
    _ax[0].plot(_psi0_values, _lik_values)
    _ax[0].axhline(_best_lik, color="red", linestyle="--")
    _ax[0].axvline(_best_psi_lik, color="red", linestyle="--")
    _ax[0].set_title(f"Likelihood (best ψ₀={_best_psi_lik:.3f})")
    _ax[0].set_xlabel("ψ₀")
    _ax[0].set_ylabel("L")
    _ax[0].grid(True, linestyle="--", alpha=0.5)

    # NLL
    _ax[1].plot(_psi0_values, _nll_values)
    _ax[1].axhline(_best_nll, color="red", linestyle="--")
    _ax[1].axvline(_best_psi_nll, color="red", linestyle="--")
    _ax[1].set_title(f"NLL (best ψ₀={_best_psi_nll:.3f})")
    _ax[1].set_xlabel("ψ₀")
    _ax[1].set_ylabel("Loss")
    _ax[1].grid(True, linestyle="--", alpha=0.5)

    plt.show()