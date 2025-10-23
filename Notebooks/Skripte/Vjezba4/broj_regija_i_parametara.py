# Knjižnice
import numpy as np
import matplotlib.pyplot as plt
import math
from ipywidgets import IntText, Dropdown, Button, HBox, VBox, Output, Layout, Checkbox
from IPython.display import display

def broj_regija(Di: int = 1, 
                D: int = 2) -> int:
    """
    # TODO
    HINT math.comb
    Izračun broja regija N = sum_{j=0}^{min(Di, D)} C(D, j).

    Args:
        *Di, int, Broj ulaznih neurona
        *D, int, Broj skrivenih neurona

    Returns:
        *int, broj linearnih regija
    """

    return 0

def broj_parametara(Di: int = 1, 
                    D: int = 1) -> int:
    """
    # TODO
    Izračun broj parametara N = (Di + 1) * D + D + 1 = Di*D + 2D + 1.

    Args:
        *Di, int, Broj ulaznih neurona
        *D, int, Broj skrivenih neurona

    Returns:
        *int, broj parametara 
    """
    return 0


def create_regions_widget_D():
    """
    Widget to plot two curves with fixed D and a sweep over D_i:
      - Left subplot: N_regions(D_i) = sum_{j=0}^{min(Di, D)} C(D, j)  (semilog-y)
      - Right subplot: N_params(D_i) = (Di + 1) * D + D + 1
    """
    # --- Controls ---
    _D      = IntText(description="D (fixed)", value=1000, layout=Layout(width="180px"))
    _Di_min = IntText(description="Dᵢ min", value=0,   layout=Layout(width="150px"))
    _Di_max = IntText(description="Dᵢ max", value=100, layout=Layout(width="150px"))
    _grid   = Checkbox(description="grid", value=True, layout=Layout(width="90px"))
    _btn    = Button(description="Plot", layout=Layout(width="120px", height="34px"))
    _out    = Output()

    _controls = HBox([_D, _Di_min, _Di_max, _grid, _btn])

    # --- Plot handler ---
    def _on_plot_clicked(_=None):
        _out.clear_output(wait=True)
        with _out:
            _d   = int(_D.value)
            _di0 = int(_Di_min.value)
            _di1 = int(_Di_max.value)
            if _d < 0 or _di0 < 0 or _di1 < 0:
                print("All values must be non-negative.")
                return
            if _di1 < _di0:
                print("Dᵢ max must be >= Dᵢ min.")
                return

            _Dis = np.arange(_di0, _di1 + 1, dtype=int)

            # Curves
            _regions = np.array([broj_regija(int(_Di), _d) for _Di in _Dis], dtype=float)
            _params  = np.array([broj_parametara(int(_Di), _d) for _Di in _Dis], dtype=float)

            # Figure with 2 subplots (side by side)
            _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(13, 4.8), sharex=False)

            # Left: Regions (semilog-y)
            _ax1.semilogy(_Dis, _regions, color='y', linewidth=2)
            _ax1.set_xlabel("Number of inputs, $D_i$")
            _ax1.set_ylabel("Number of regions, $N$")
            _ax1.set_title(fr"Regions vs $D_i$ (fixed $D$={_d})")
            if _grid.value:
                _ax1.grid(True, which="both", linestyle="--", alpha=0.3)
            if _di1 - _di0 >= 50:
                _ax1.set_xlim(_di0, _di1)
                _pos = _regions[_regions > 0]
                if _pos.size:
                    _ymin, _ymax = float(_pos.min()), float(_pos.max())
                    _ax1.set_ylim(max(1e0, _ymin), min(1e300, _ymax * 1.1))

            # Right: Parameters (linear y)
            _ax2.plot(_Dis, _params, color='b', linewidth=2)
            _ax2.set_xlabel("Number of inputs, $D_i$")
            _ax2.set_ylabel("Number of parameters, $N$")
            _ax2.set_title(fr"Parameters vs $D_i$ (fixed $D$={_d})")
            if _grid.value:
                _ax2.grid(True, linestyle="--", alpha=0.3)
            if _di1 - _di0 >= 50:
                _ax2.set_xlim(_di0, _di1)

            _fig.tight_layout()
            display(_fig)
            plt.close(_fig)

    _btn.on_click(_on_plot_clicked)

    # Initial draw
    _on_plot_clicked()
    display(VBox([_controls, _out]))
def create_regions_widget_Di():
    """
    Widget to plot two curves with fixed D_i and a sweep over D:
      - Left subplot: N_regions(D) = sum_{j=0}^{min(Di, D)} C(D, j)  (semilog-y)
      - Right subplot: N_params(D) = (Di + 1) * D + D + 1
    """
    # --- Controls ---
    _Di    = IntText(description="D\u2099 (inputs)", value=10, layout=Layout(width="180px"))
    _D_min = IntText(description="D min", value=0,   layout=Layout(width="150px"))
    _D_max = IntText(description="D max", value=1000,layout=Layout(width="150px"))
    _grid  = Checkbox(description="grid", value=True, layout=Layout(width="90px"))
    _btn   = Button(description="Plot", layout=Layout(width="120px", height="34px"))
    _out   = Output()

    _controls = HBox([_Di, _D_min, _D_max, _grid, _btn])

    # --- Plot handler ---
    def _on_plot_clicked(_=None):
        _out.clear_output(wait=True)
        with _out:
            _di = int(_Di.value)
            _d0 = int(_D_min.value)
            _d1 = int(_D_max.value)
            if _di < 0 or _d0 < 0 or _d1 < 0:
                print("All values must be non-negative.")
                return
            if _d1 < _d0:
                print("D max must be >= D min.")
                return

            _Ds = np.arange(_d0, _d1 + 1, dtype=int)

            # Curves
            _regions = np.array([broj_regija(_di, int(_D)) for _D in _Ds], dtype=float)
            _params  = np.array([broj_parametara(_di, int(_D)) for _D in _Ds], dtype=float)

            # Figure with 2 subplots (side by side)
            _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(13, 4.8), sharex=False)

            # Left: Regions (semilog-y)
            _ax1.semilogy(_Ds, _regions, color='y', linewidth=2)
            _ax1.set_xlabel("Number of hidden units, D")
            _ax1.set_ylabel("Number of regions, N")
            _ax1.set_title(fr"Regions vs D (fixed $D_i$={_di})")
            if _grid.value:
                _ax1.grid(True, which="both", linestyle="--", alpha=0.3)
            if _d1 - _d0 >= 100:
                _ax1.set_xlim(_d0, _d1)
                _pos = _regions[_regions > 0]
                if _pos.size:
                    _ymin, _ymax = float(_pos.min()), float(_pos.max())
                    _ax1.set_ylim(max(1e0, _ymin), min(1e300, _ymax * 1.1))

            # Right: Parameters (linear y)
            _ax2.plot(_Ds, _params, color='b', linewidth=2)
            _ax2.set_xlabel("Number of hidden units, D")
            _ax2.set_ylabel("Number of parameters, N")
            _ax2.set_title(fr"Parameters vs D (fixed $D_i$={_di})")
            if _grid.value:
                _ax2.grid(True, linestyle="--", alpha=0.3)
            if _d1 - _d0 >= 100:
                _ax2.set_xlim(_d0, _d1)

            _fig.tight_layout()
            display(_fig)
            plt.close(_fig)

    _btn.on_click(_on_plot_clicked)

    # Initial draw
    _on_plot_clicked()
    display(VBox([_controls, _out]))
