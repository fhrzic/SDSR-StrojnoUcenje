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
    _df = pd.read_csv(path_to_csv)

    # String update
    _df_numeric = _df.select_dtypes(include=[np.number])

    # String update
    _columns = _df_numeric.columns.tolist()
   
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
    _t10 = FloatText(description="θ10", value=0.0, layout={"width":"140px"})
    _t11 = FloatText(description="θ11", value=1.0, layout={"width":"140px"})
    _t20 = FloatText(description="θ20", value=0.0, layout={"width":"140px"})
    _t21 = FloatText(description="θ21", value=1.0, layout={"width":"140px"})
    _t30 = FloatText(description="θ30", value=0.0, layout={"width":"140px"})
    _t31 = FloatText(description="θ31", value=1.0, layout={"width":"140px"})

    _p0  = FloatText(description="ψ0", value=0.0, layout={"width":"140px"})
    _p1  = FloatText(description="ψ1", value=1.0, layout={"width":"140px"})
    _p2  = FloatText(description="ψ2", value=0.0, layout={"width":"140px"})
    _p3  = FloatText(description="ψ3", value=0.0, layout={"width":"140px"})

    _dd_act = Dropdown(description="activation",
                       options=["none","relu","tanh","sigmoid"],
                       value="none", layout={"width":"200px"})

    _x_start = FloatText(description="x start", value=0.0,  layout={"width":"140px"})
    _x_stop  = FloatText(description="x stop",  value=1.0,  layout={"width":"140px"})
    _x_step  = FloatText(description="x step",  value=0.01, layout={"width":"140px"})

    # Axis locking controls
    _lock_y = ToggleButton(description="lock y", value=False, layout={"width":"100px"})
    _y_min  = FloatText(description="y min", value=-5.0, layout={"width":"140px"})
    _y_max  = FloatText(description="y max", value= 5.0, layout={"width":"140px"})

    _btn_plot = Button(description="Plot", layout={"width":"120px"})
    _out = Output()

    def _on_plot_clicked(_=None):
        _out.clear_output(wait=True)
        with _out:
            # Build x
            try:
                _xs = np.arange(float(_x_start.value), float(_x_stop.value), float(_x_step.value))
            except Exception as _e:
                print(f"Invalid x range: {_e}")
                return
            if _xs.size == 0:
                print("x range is empty. Check start/stop/step.")
                return

            _theta_init = np.array([
                [_t10.value, _t11.value],
                [_t20.value, _t21.value],
                [_t30.value, _t31.value],
            ], dtype=float)

            _psi_init = np.array([_p0.value, _p1.value, _p2.value, _p3.value], dtype=float)
            _act = None if _dd_act.value == "none" else _dd_act.value

            # Model
            _model = model_1_3_1_for_plots(theta_init_matrix=_theta_init,
                                           psi_init_matrix=_psi_init,
                                           activation_function=_act).to(device).eval()

            # Forward
            with torch.no_grad():
                _x_t = torch.tensor(_xs, dtype=torch.float32, device=device)
                _y, _z1, _z2, _z3, _h1, _h2, _h3, _w1, _w2, _w3 = _model(_x_t)

            _to_np = lambda _t: _t.detach().cpu().numpy().astype(float)

            # Arrange series: 3x3 top, bottom wide
            _series_grid = [
                ("z1 = t10 + t11·x", _to_np(_z1)),
                ("z2 = t20 + t21·x", _to_np(_z2)),
                ("z3 = t30 + t31·x", _to_np(_z3)),
                ("h1 = a1(z1)",      _to_np(_h1)),
                ("h2 = a2(z2)",      _to_np(_h2)),
                ("h3 = a3(z3)",      _to_np(_h3)),
                ("w1 = ψ1·h1",       _to_np(_w1)),
                ("w2 = ψ2·h2",       _to_np(_w2)),
                ("w3 = ψ3·h3",       _to_np(_w3)),
            ]
            _series_wide = ("y  (final output)", _to_np(_y))

            # --- Determine axis limits ---
            _x_lo, _x_hi = float(_x_start.value), float(_x_stop.value)  # static x across all
            if _lock_y.value:
                _y_lo, _y_hi = float(_y_min.value), float(_y_max.value)
            else:
                # global min/max across all series (including y)
                _all_vals = np.concatenate([_v for _, _v in _series_grid] + [_series_wide[1]])
                if _all_vals.size == 0:
                    _y_lo, _y_hi = -1.0, 1.0
                else:
                    _y_lo, _y_hi = np.min(_all_vals), np.max(_all_vals)
                    # small padding
                    _pad = 0.05 * (_y_hi - _y_lo if _y_hi > _y_lo else 1.0)
                    _y_lo, _y_hi = _y_lo - _pad, _y_hi + _pad
                # update fields to show computed limits (optional)
                _y_min.value, _y_max.value = float(_y_lo), float(_y_hi)

            # --- Grid: 3x3 + 1 wide bottom ---
            _fig = plt.figure(figsize=(12, 16))
            _gs = GridSpec(4, 3, figure=_fig)

            # Top 3x3
            for _i, (_title, _yvals) in enumerate(_series_grid):
                _r, _c = divmod(_i, 3)
                _ax = _fig.add_subplot(_gs[_r, _c])
                _ax.plot(_xs, _yvals)
                _ax.set_title(_title, fontsize=10)
                _ax.grid(True, linestyle="--", alpha=0.3)
                # static axes:
                _ax.set_xlim(_x_lo, _x_hi)
                _ax.set_ylim(_y_lo, _y_hi)
                if _r == 2:
                    _ax.set_xlabel("x")
                if _c == 0:
                    _ax.set_ylabel("value")

            # Bottom wide (y)
            _ax_wide = _fig.add_subplot(_gs[3, 0])
            _ax_wide.plot(_xs, _series_wide[1])
            _ax_wide.set_title(f"{_series_wide[0]}", fontsize=11)
            _ax_wide.set_xlabel("x")
            _ax_wide.set_ylabel("value")
            _ax_wide.grid(True, linestyle="--", alpha=0.3)
            _ax_wide.set_xlim(_x_lo, _x_hi)
            _ax_wide.set_ylim(_y_lo, _y_hi)

            # Turn off unused axes in bottom row
            for _c in (1, 2):
                _ax_empty = _fig.add_subplot(_gs[3, _c])
                _ax_empty.axis("off")

            _fig.tight_layout()
            display(_fig)
            plt.close(_fig)

    _btn_plot.on_click(_on_plot_clicked)

    # Initial draw
    _on_plot_clicked()

    _ui = VBox([
        HBox([_t10, _t11, _t20, _t21, _t30, _t31]),
        HBox([_p0, _p1, _p2, _p3, _dd_act]),
        HBox([_x_start, _x_stop, _x_step, _lock_y, _y_min, _y_max, _btn_plot]),
        _out
    ])
    display(_ui)