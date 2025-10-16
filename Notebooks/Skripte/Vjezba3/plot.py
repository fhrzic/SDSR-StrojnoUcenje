"""
Skripta za crtanje grafova
"""

# Knjižnica
import sys
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from ipywidgets import Dropdown, HBox, VBox, Label, Output, FloatText, ToggleButton, Button
from IPython.display import display, clear_output
from torch import  nn
from torch.utils.data import DataLoader
from matplotlib.gridspec import GridSpec

# Add project files
current_file_path = os.path.abspath(__file__)
main_dir =  os.path.dirname(current_file_path)
print(main_dir)
sys.path.append(main_dir)

from dataloader import *
from models import *

def create_bike_correlation_widget(path_to_csv: str = None):
    """
    Widget za crtanje podataka

    Args:
        path_to_csv, str, put do csv datoteke

    Returns:
        Top-level VBox widget.
    """

    # Učitajmo nazive stupaca 
    assert os.path.exists(path_to_csv), f"Path does not exist: {path_to_csv}"
    _columns = pd.read_csv(path_to_csv, nrows=0).columns.tolist() 
    # Maknimo string zbog plottanja
    if 'dteday' in _columns:
        _columns.remove('dteday')

    def plot_correlations(var1: str = None,
                        var2: str = None,
                        path_to_csv: str = None,
                        out: Output = None):
        """
        Funkcija koja crta odnos dviju varijabli te ispisuje
        njihovu korelaciju.

        Args:
            * var1, str, ime prve varijable -- X os
            * var2, str, ime druge varijable -- Y os
            * path_to_csv, str, put do podataka
            * out, Output, potrebno za dinamičko crtanje
        """
        # Stvorimo dataset
        # Dataset
        _dataset = bikeRentalDataset(path_to_csv = path_to_csv,
                                    input_label = var1,
                                    target_label = var2)

        # Dataloader
        _dataloader = ordinary_dataloader(dataset = _dataset,
                                        batch_size = 1)
        
        # Pokupimo sve podatke iz skupa dataloadera
        _xs, _ys = [], []
        for _x_batch, _y_batch in _dataloader:
            # Squeeze eliminira prvu dimenziju koja je nastala dobivanje batcha
            _xs.append(_x_batch.squeeze().item())
            _ys.append(_y_batch.squeeze().item())

        _x = np.asarray(_xs, dtype=float)
        _y = np.asarray(_ys, dtype=float)

        # Izračun korelacije
        if len(_x) < 2 or np.allclose(_x.std(), 0) or np.allclose(_y.std(), 0):
            _r = np.nan
        else:
            _r = np.corrcoef(_x, _y)[0, 1]

        # Crtanje podataka
        out.clear_output(wait=True)
        with out:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.scatter(_x, _y, alpha=0.8)
            ax.set_title(f"{var1} vs {var2}")
            ax.set_xlabel(var1)
            ax.set_ylabel(var2)
            ax.grid(True, linestyle="--", alpha=0.4)
            if np.isfinite(_r):
                fig.suptitle(f"Pearson r = {_r:.3f} (R² = {_r**2:.3f})", y=1.02, fontsize=10)
            else:
                fig.suptitle("Pearson r = NaN (insufficient variance or samples)", y=1.02, fontsize=10)
            fig.tight_layout()
            display(fig)
            plt.close(fig)  # <- prevents double rendering

    # Widgets
    _var1_dd = Dropdown(options = _columns, 
                        value = _columns[0], 
                        description="X (var1):", 
                        layout={"width": "300px"})
    _var2_dd = Dropdown(options = _columns, 
                        value= _columns[1], 
                        description="Y (var2):", 
                        layout={"width": "300px"})
    _status = Output()
    _plot_out = Output()

    def _update(*_):
        with _status:
            _status.clear_output(wait=True)
            
        plot_correlations(_var1_dd.value, 
                          _var2_dd.value, 
                          path_to_csv,
                          _plot_out)

    _var1_dd.observe(_update, names="value")
    _var2_dd.observe(_update, names="value")

    # Initial draw
    _update()

    ui = VBox([HBox([_var1_dd, _var2_dd]), _status, _plot_out])
    display(ui)


def plot_model_over_data(dataloader: DataLoader = None,
                         model: nn.Module = None):
    """
    Funkcija koja crta dani model nad podacima:

    Args:
        * dataloader, DataLoader, dataloader za podatake
        * model, nn.Module, model koji crta podatke 
    """
    # Storage
    _xs = []
    _predictions = []
    _ground_truth = []

    # Izracunajmo predikcije
    for _sample in dataloader:
        # Obtain
        _x, _y = _sample
        
        # Predict
        _y_hat = model(_x)

        # Store
        _xs.append(_x.squeeze().cpu().item())
        _ground_truth.append(_y.squeeze().cpu().item())
        _predictions.append(_y_hat.squeeze().cpu().item())

    _xs = np.asarray(_xs, dtype=float)
    _ground_truth = np.asarray(_ground_truth, dtype=float)
    _predictions = np.asarray(_predictions, dtype=float)

    # MSE
    _mse = np.mean((_ground_truth - _predictions) ** 2)

    # Sort by x for a clean prediction line
    _order = np.argsort(_xs)
    _xs_sorted = _xs[_order]
    _predictions_sorted = _predictions[_order]

    # Plot
    plt.figure(figsize=(7, 4))
    plt.scatter(_xs, _ground_truth, alpha=0.8, label="Ground truth")
    plt.scatter(_xs, _predictions, alpha=0.8, marker="x", label="Predictions")
    plt.plot(_xs_sorted, _predictions_sorted, linewidth=2, label="Prediction line")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(f"MSE = {_mse}")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.close()


def create_line_fit_with_activation_widget(csv_path: str = "Skripte/Vjezba3/day_bikes_rental.csv",
                                           input_label: str = "temp",
                                           target_label: str = "cnt",

                                           batch_size: int = 1,
                                           device: str = "cpu"):
    """
    Interaktivni widget za crtanje modela y = theta1 * x + theta0 nad podacima.
    Args:
        * input_label, str, ulazna labela - x os
        * target_label, str, izlazna labela - y os
        * batch_size, int, velicina batcha
        * theta0_init (bias)
        * theta1_init (slope)
    """

    assert os.path.exists(csv_path), f"CSV not found: {csv_path}"

    # --- Controls ---
    _v_theta0 = FloatText(description="theta0", value=0.0, layout={"width": "250px"})
    _v_theta1 = FloatText(description="theta1", value=0.0, layout={"width": "250px"})
    _t_norm   = ToggleButton(description="Normalize", value=False)
    _dd_act   = Dropdown(description="Activation",
                        options=["None", "ReLU", "Tanh", "Sigmoid"],
                        value="None")

    _out_status = Output()
    _out_plot = Output()

    def _update_plot(*_):
        _out_status.clear_output(wait=True)
        _out_plot.clear_output(wait=True)

        # 1) Dataset & dataloader
        _ds = bikeRentalDataset(path_to_csv=csv_path,
                                input_label=input_label,
                                target_label=target_label,
                                normalizacija=_t_norm.value)
        
        _dl = ordinary_dataloader(dataset=_ds, batch_size=batch_size)
        
        # 2) Model
        _model = model_1_1_1(theta0_init=_v_theta0.value, 
                             theta1_init=_v_theta1.value,
                             activation_function=_dd_act.value).to(device).eval()

        # 3) Plot (koristi vaš plotter)
        with _out_plot:
            plot_model_over_data(dataloader=_dl, model=_model)

        with _out_status:
            print(f"theta0={_v_theta0.value:.4f}, theta1={_v_theta1.value:.4f}, normalize={_t_norm.value}")

    # --- wire up events ---
    _v_theta0.observe(_update_plot, names="value")
    _v_theta1.observe(_update_plot, names="value")
    _t_norm.observe(_update_plot, names="value")
    _dd_act.observe(_update_plot, names="value")

    # --- initial draw ---
    _update_plot()

    ui = VBox([
        HBox([_v_theta0, _v_theta1, _t_norm, _dd_act]),
        _out_status,
        _out_plot
    ])
    display(ui)

def create_line_fit_widget(csv_path: str = "Skripte/Vjezba3/day_bikes_rental.csv",
                           input_label: str = "temp",
                           target_label: str = "cnt",
                           batch_size: int = 1,
                           device: str = "cpu"):
    """
    Interaktivni widget za crtanje modela y = theta1 * x + theta0 nad podacima.
    Args:
        * input_label, str, ulazna labela - x os
        * target_label, str, izlazna labela - y os
        * batch_size, int, velicina batcha
        * theta0_init (bias)
        * theta1_init (slope)
    """

    assert os.path.exists(csv_path), f"CSV not found: {csv_path}"

    # --- Controls ---
    _v_theta0 = FloatText(description="theta0", value=0.0, layout={"width": "250px"})
    _v_theta1 = FloatText(description="theta1", value=0.0, layout={"width": "250px"})
    _t_norm   = ToggleButton(description="Normalize", value=False)

    _out_status = Output()
    _out_plot = Output()

    def _update_plot(*_):
        _out_status.clear_output(wait=True)
        _out_plot.clear_output(wait=True)

        # 1) Dataset & dataloader
        _ds = bikeRentalDataset(path_to_csv=csv_path,
                                input_label=input_label,
                                target_label=target_label,
                                normalizacija=_t_norm.value)
        
        _dl = ordinary_dataloader(dataset=_ds, batch_size=batch_size)
        
        # 2) Model
        _model = model_1_1_1(theta0_init=_v_theta0.value, 
                             theta1_init=_v_theta1.value).to(device).eval()

        # 3) Plot (koristi vaš plotter)
        with _out_plot:
            plot_model_over_data(dataloader=_dl, model=_model)

        with _out_status:
            print(f"theta0={_v_theta0.value:.4f}, theta1={_v_theta1.value:.4f}, normalize={_t_norm.value}")

    # --- wire up events ---
    _v_theta0.observe(_update_plot, names="value")
    _v_theta1.observe(_update_plot, names="value")
    _t_norm.observe(_update_plot, names="value")

    # --- initial draw ---
    _update_plot()

    ui = VBox([
        HBox([_v_theta0, _v_theta1, _t_norm]),
        _out_status,
        _out_plot
    ])
    display(ui)


def create_model_1_3_1_widget(csv_path: str = "Skripte/Vjezba3/day_bikes_rental.csv",
                               input_label: str = "temp",
                               target_label: str = "cnt",
                               batch_size: int = 1,
                               device: str = "cpu"):
    """
    Interaktivni widget za:
      y = psi0 + psi1*a1(t10 + t11*x) + psi2*a2(t20 + t21*x) + psi3*a3(t30 + t31*x)
    Kontrole: θ10, θ11, θ20, θ21, θ30, θ31, ψ0..ψ3, Normalize, activation.
    """
    assert os.path.exists(csv_path), f"CSV not found: {csv_path}"

    # --- Controls (exact numeric inputs) ---
    _t10 = FloatText(description="θ10", value=0.0, layout={"width":"160px"})
    _t11 = FloatText(description="θ11", value=1.0, layout={"width":"160px"})
    _t20 = FloatText(description="θ20", value=0.0, layout={"width":"160px"})
    _t21 = FloatText(description="θ21", value=1.0, layout={"width":"160px"})
    _t30 = FloatText(description="θ30", value=0.0, layout={"width":"160px"})
    _t31 = FloatText(description="θ31", value=1.0, layout={"width":"160px"})

    _p0  = FloatText(description="ψ0", value=0.0, layout={"width":"160px"})
    _p1  = FloatText(description="ψ1", value=1.0, layout={"width":"160px"})
    _p2  = FloatText(description="ψ2", value=0.0, layout={"width":"160px"})
    _p3  = FloatText(description="ψ3", value=0.0, layout={"width":"160px"})

    _t_norm = ToggleButton(description="normalize", value=False, layout={"width":"140px"})
    _dd_act = Dropdown(description="activation",
                       options=["none","relu","tanh","sigmoid"],
                       value="none",
                       layout={"width":"200px"})

    _out_status = Output()
    _out_plot = Output()

    def _update_plot(*_):
        _out_status.clear_output(wait=True)
        _out_plot.clear_output(wait=True)

        # 1) Dataset & dataloader
        #   If your dataset uses a different arg (norm="standard"/"none"), adjust here.
        _ds = bikeRentalDataset(path_to_csv=csv_path,
                                input_label=input_label,
                                target_label=target_label,
                                normalizacija=_t_norm.value)

        _dl = ordinary_dataloader(dataset=_ds, batch_size=batch_size)

        # 2) Build init matrices from widgets
        _theta_init = np.array([
            [_t10.value, _t11.value],
            [_t20.value, _t21.value],
            [_t30.value, _t31.value],
        ], dtype=float)

        _psi_init = np.array([_p0.value, _p1.value, _p2.value, _p3.value], dtype=float)

        # 3) Model with chosen activation
        _act = None if _dd_act.value == "none" else _dd_act.value
        _model = model_1_3_1(theta_init_matrix=_theta_init,
                            psi_init_matrix=_psi_init,
                            activation_function=_act).to(device).eval()

        # 4) Plot
        with _out_plot:
            plot_model_over_data(dataloader=_dl, model=_model)

        with _out_status:
            print(
                f"θ10={_t10.value:.4f}, θ11={_t11.value:.4f} | "
                f"θ20={_t20.value:.4f}, θ21={_t21.value:.4f} | "
                f"θ30={_t30.value:.4f}, θ31={_t31.value:.4f} || "
                f"ψ0={_p0.value:.4f}, ψ1={_p1.value:.4f}, ψ2={_p2.value:.4f}, ψ3={_p3.value:.4f} | "
                f"normalize={_t_norm.value} | activation={_dd_act.value}"
            )

    # Wire up auto-update
    for _w in (_t10,_t11,_t20,_t21,_t30,_t31,_p0,_p1,_p2,_p3,_t_norm,_dd_act):
        _w.observe(_update_plot, names="value")

    # Initial draw
    _update_plot()

    _ui = VBox([
        HBox([_t10, _t11, _t20, _t21, _t30, _t31]),
        HBox([_p0, _p1, _p2, _p3, _t_norm, _dd_act]),
        _out_status,
        _out_plot
    ])
    display(_ui)


def create_model_1_3_1_component_plots_widget(device: str = "cpu"):
    """
    Interaktivni widget za model_1_3_1_for_plots:
      - Unos svih hiperparametara (theta i psi)
      - Odabir aktivacije (none/relu/tanh/sigmoid)
      - Unos intervala x (start, stop, step)
      - Crta 3x3 + 1 wide (y) grid s *fiksnim* x-osi i opcionalno *zaključanom* y-osi za sve grafove
    """

    # --- Controls (exact numeric inputs) ---
    t10 = FloatText(description="θ10", value=0.0, layout={"width":"140px"})
    t11 = FloatText(description="θ11", value=1.0, layout={"width":"140px"})
    t20 = FloatText(description="θ20", value=0.0, layout={"width":"140px"})
    t21 = FloatText(description="θ21", value=1.0, layout={"width":"140px"})
    t30 = FloatText(description="θ30", value=0.0, layout={"width":"140px"})
    t31 = FloatText(description="θ31", value=1.0, layout={"width":"140px"})

    p0  = FloatText(description="ψ0", value=0.0, layout={"width":"140px"})
    p1  = FloatText(description="ψ1", value=1.0, layout={"width":"140px"})
    p2  = FloatText(description="ψ2", value=0.0, layout={"width":"140px"})
    p3  = FloatText(description="ψ3", value=0.0, layout={"width":"140px"})

    dd_act = Dropdown(description="activation",
                      options=["none","relu","tanh","sigmoid"],
                      value="none", layout={"width":"200px"})

    x_start = FloatText(description="x start", value=0.0,  layout={"width":"140px"})
    x_stop  = FloatText(description="x stop",  value=1.0,  layout={"width":"140px"})
    x_step  = FloatText(description="x step",  value=0.01, layout={"width":"140px"})

    # Axis locking controls
    lock_y = ToggleButton(description="lock y", value=False, layout={"width":"100px"})
    y_min  = FloatText(description="y min", value=-5.0, layout={"width":"140px"})
    y_max  = FloatText(description="y max", value= 5.0, layout={"width":"140px"})

    btn_plot = Button(description="Plot", layout={"width":"120px"})
    out = Output()

    def _on_plot_clicked(_=None):
        out.clear_output(wait=True)
        with out:
            # Build x
            try:
                xs = np.arange(float(x_start.value), float(x_stop.value), float(x_step.value))
            except Exception as e:
                print(f"Invalid x range: {e}")
                return
            if xs.size == 0:
                print("x range is empty. Check start/stop/step.")
                return

            theta_init = np.array([
                [t10.value, t11.value],
                [t20.value, t21.value],
                [t30.value, t31.value],
            ], dtype=float)

            psi_init = np.array([p0.value, p1.value, p2.value, p3.value], dtype=float)
            act = None if dd_act.value == "none" else dd_act.value

            # Model
            model = model_1_3_1_for_plots(theta_init_matrix=theta_init,
                                          psi_init_matrix=psi_init,
                                          activation_function=act).to(device).eval()

            # Forward
            with torch.no_grad():
                x_t = torch.tensor(xs, dtype=torch.float32, device=device)
                y, z1, z2, z3, h1, h2, h3, w1, w2, w3 = model(x_t)

            to_np = lambda t: t.detach().cpu().numpy().astype(float)

            # Arrange series: 3x3 top, bottom wide
            series_grid = [
                ("z1 = t10 + t11·x", to_np(z1)),
                ("z2 = t20 + t21·x", to_np(z2)),
                ("z3 = t30 + t31·x", to_np(z3)),
                ("h1 = a1(z1)",      to_np(h1)),
                ("h2 = a2(z2)",      to_np(h2)),
                ("h3 = a3(z3)",      to_np(h3)),
                ("w1 = ψ1·h1",       to_np(w1)),
                ("w2 = ψ2·h2",       to_np(w2)),
                ("w3 = ψ3·h3",       to_np(w3)),
            ]
            series_wide = ("y  (final output)", to_np(y))

            # --- Determine axis limits ---
            x_lo, x_hi = float(x_start.value), float(x_stop.value)  # static x across all
            if lock_y.value:
                y_lo, y_hi = float(y_min.value), float(y_max.value)
            else:
                # global min/max across all series (including y)
                all_vals = np.concatenate([v for _, v in series_grid] + [series_wide[1]])
                if all_vals.size == 0:
                    y_lo, y_hi = -1.0, 1.0
                else:
                    y_lo, y_hi = np.min(all_vals), np.max(all_vals)
                    # small padding
                    pad = 0.05 * (y_hi - y_lo if y_hi > y_lo else 1.0)
                    y_lo, y_hi = y_lo - pad, y_hi + pad
                # update fields to show computed limits (optional)
                y_min.value, y_max.value = float(y_lo), float(y_hi)

            # --- Grid: 3x3 + 1 wide bottom ---
            fig = plt.figure(figsize=(12, 16))
            gs = GridSpec(4, 3, figure=fig)

            # Top 3x3
            for i, (title, yvals) in enumerate(series_grid):
                r, c = divmod(i, 3)
                ax = fig.add_subplot(gs[r, c])
                ax.plot(xs, yvals)
                ax.set_title(title, fontsize=10)
                ax.grid(True, linestyle="--", alpha=0.3)
                # static axes:
                ax.set_xlim(x_lo, x_hi)
                ax.set_ylim(y_lo, y_hi)
                if r == 2:
                    ax.set_xlabel("x")
                if c == 0:
                    ax.set_ylabel("value")

            # Bottom wide (y)
            ax_wide = fig.add_subplot(gs[3, :])
            ax_wide.plot(xs, series_wide[1])
            ax_wide.set_title(f"{series_wide[0]} (wide)", fontsize=11)
            ax_wide.set_xlabel("x")
            ax_wide.set_ylabel("value")
            ax_wide.grid(True, linestyle="--", alpha=0.3)
            ax_wide.set_xlim(x_lo, x_hi)
            ax_wide.set_ylim(y_lo, y_hi)

            fig.tight_layout()
            display(fig)
            plt.close(fig)

    btn_plot.on_click(_on_plot_clicked)

    # Initial draw
    _on_plot_clicked()

    ui = VBox([
        HBox([t10, t11, t20, t21, t30, t31]),
        HBox([p0, p1, p2, p3, dd_act]),
        HBox([x_start, x_stop, x_step, lock_y, y_min, y_max, btn_plot]),
        out
    ])
    display(ui)
