"""
Skirpta u kojoj se implementira linearna pretraga
"""

# Knjižnica
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RangeSlider
import os
import sys
from typing import Callable, Tuple, Optional, List, Dict
from matplotlib.lines import Line2D  # for legend entries

# Add project files
current_file_path = os.path.abspath(__file__)
main_dir =  os.path.dirname(current_file_path)
print(main_dir)
sys.path.append(main_dir)

# === ipywidgets verzija s točno traženim importima ===
from ipywidgets import Dropdown, Button, HBox, VBox, Output, Label, Layout, FloatSlider
from IPython.display import display


def bracketing_step(
    _loss_fn: Callable[[float], float],
    _a: float,
    _b: float,
    _c: float,
    _d: float,
) -> Tuple[float, float, float, float]:
    """
    Jedan 'bracketing' korak:
      - Evaluira L(b) i L(c)
      - Ako je L(b) > L(c): minimum je u [b, d]  → _a := _b
        Inače:            minimum je u [a, c]  → _d := _c
      - Ponovno postavlja unutarnje točke _b i _c na TREĆINE novog intervala

    Ulaz:
        _loss_fn : funkcija gubitka L(x)
        _a,_b,_c,_d : trenutni segment s uvjetom _a < _b < _c < _d

    Izlaz:
        (_a_new, _b_new, _c_new, _d_new)
    """
    # --- provjera urednog poretka ---
    if not (_a < _b < _c < _d):
        raise ValueError("Mora vrijediti _a < _b < _c < _d u bracketing_step.")

    # --- evaluacija ---
    _Lb = _loss_fn(_b)
    _Lc = _loss_fn(_c)

    # --- ažuriranje intervala ---
    if _Lb > _Lc:
        # minimum je u [b, d]
        _a_new, _d_new = _b, _d
    else:
        # minimum je u [a, c]
        _a_new, _d_new = _a, _c

    # --- nove unutarnje točke: trisekcija ---
    _b_new = _a_new + (_d_new - _a_new) / 3.0
    _c_new = _a_new + 2.0 * (_d_new - _a_new) / 3.0

    return _a_new, _b_new, _c_new, _d_new


def bracketing_min_search(
    _loss_fn: Callable[[float], float],
    _a: float,
    _b: float,
    _c: float,
    _d: float,
    _tolerance: float = 1e-3,
    _max_iter: int = 100,
    _return_history: bool = False,
) -> Tuple[float, float, Optional[List[Dict[str, float]]]]:
    """
    Iterativna varijanta koja poziva bracketing_step dok (d - a) <= _tolerance.

    Postupak:
      - Krene od zadanog [_a,_d] i točaka _b,_c (_a < _b < _c < _d)
      - U svakoj iteraciji pozove bracketing_step i sužava interval
      - Zaustavlja se kad je (_d - _a) <= _tolerance ili nakon _max_iter

    Vraća:
        (_x_star, _L_star, _history)
        gdje je _x_star = sredina [_a, _d], _L_star = L(_x_star),
        a _history (ako je tražen) je lista snimki po koraku.
    """
    if not (_a < _b < _c < _d):
        raise ValueError("Mora vrijediti _a < _b < _c < _d na početku.")
    if _tolerance <= 0:
        raise ValueError("_tolerance mora biti > 0.")
    if _max_iter <= 0:
        raise ValueError("_max_iter mora biti > 0.")

    _history: List[Dict[str, float]] = []

    for _ in range(_max_iter):
        if (_d - _a) <= _tolerance:
            break
        if _return_history:
            _history.append({"a": _a, "b": _b, "c": _c, "d": _d})

        _a, _b, _c, _d = bracketing_step(_loss_fn, _a, _b, _c, _d)

    _x_star = 0.5 * (_a + _d)
    _L_star = _loss_fn(_x_star)
    return (_x_star, _L_star, _history) if _return_history else (_x_star, _L_star, None)

 # -----------------------------
# Definicije funkcija (dvije Gaussove udoline)
# -----------------------------
def _loss_v1(_phi):
    # v1: primjer s minimumima oko 0.35 i 0.65 (kao u vašem uzorku)
    return 1.0 - 0.5*np.exp(-((_phi-0.65)*(_phi-0.65))/0.1) - 0.45*np.exp(-((_phi-0.35)*(_phi-0.35))/0.02)

def _loss_v2(_phi):
    # v2: uža udolina oko 0.2 i šira oko 0.8
    return 1.0 - 0.55*np.exp(-((_phi-0.20)*(_phi-0.20))/0.015) - 0.40*np.exp(-((_phi-0.80)*(_phi-0.80))/0.060)

def _loss_v3(_phi):
    # v3: dublja udolina oko 0.55 i vrlo uska oko 0.10
    return 1.0 - 0.50*np.exp(-((_phi-0.55)*(_phi-0.55))/0.030) - 0.45*np.exp(-((_phi-0.10)*(_phi-0.10))/0.008)

def _loss_v4(_phi):
    # v4: srednje široke udoline oko 0.35 i 0.90
    return 1.0 - 0.42*np.exp(-((_phi-0.35)*(_phi-0.35))/0.020) - 0.48*np.exp(-((_phi-0.90)*(_phi-0.90))/0.040)

def linear_search_widget():
    """
    Widget za bracketing line-search:
      - Uređuje se SAMO a i d (slideri); b i c se računaju.
      - 'Plot' resetira (a,b,c,d)=(0,1/3,2/3,1) i crta.
      - 'Next' poziva VAŠ bracketing_step za jedan korak.
      - 'Converge' iterira bracketing_step dok (d-a) ≤ tol (ili max_iter).
      - Legenda prikazuje i x i loss: a=..., L=..., b=..., L=..., ...
      - Automatski re-plot pri promjeni a ili d.
    Pretpostavlja postojanje: bracketing_step(_loss_fn, _a, _b, _c, _d) -> (a',b',c',d').
    """
    _FUNCTIONS = {
        "Mixture v1 (0.65 & 0.35)": _loss_v1,
        "Mixture v2 (0.20 & 0.80)": _loss_v2,
        "Mixture v3 (0.55 & 0.10)": _loss_v3,
        "Mixture v4 (0.35 & 0.90)": _loss_v4,
    }

    _N = 500
    _PHI_LO, _PHI_HI = 0.0, 1.0
    _colors = {'a': 'tab:blue', 'b': 'tab:orange', 'c': 'tab:green', 'd': 'tab:red'}

    _out = Output(layout=Layout(border="1px solid #ddd", width="900px", height="520px"))
    _dd = Dropdown(options=list(_FUNCTIONS.keys()), value="Mixture v1 (0.65 & 0.35)",
                   description="Funkcija:", layout=Layout(width="320px"))
    _btn_plot = Button(description="Plot", layout=Layout(width="90px"))
    _btn_next = Button(description="Next", layout=Layout(width="90px"))
    _btn_conv = Button(description="Converge", layout=Layout(width="110px"))
    _lbl = Label("ϕ ∈ [0, 1], N = 500")
    _status = Label("")

    _a = FloatSlider(value=0.0, min=0.0, max=0.99, step=0.001, description="a", layout=Layout(width="300px"))
    _d = FloatSlider(value=1.0, min=0.01, max=1.0, step=0.001, description="d", layout=Layout(width="300px"))

    # tolerancija i max_iter kao slideri (max_iter float → cast to int)
    _tol = FloatSlider(value=1e-3, min=1e-6, max=1e-1, step=1e-6, readout_format=".1e",
                       description="tol", layout=Layout(width="300px"))
    _max_iter = FloatSlider(value=100.0, min=1.0, max=100.0, step=1.0,
                            description="max_iter", layout=Layout(width="300px"))

    _b_val = 1/3
    _c_val = 2/3
    _b_label = Label(f"b = {_b_val:.3f}")
    _c_label = Label(f"c = {_c_val:.3f}")

    def _ensure_order_and_bounds():
        _eps = 1e-6
        if _a.value >= _d.value - _eps:
            _a.value = max(0.0, min(_d.value - 1e-3, 1.0))
        _a.max = max(0.0, _d.value - 1e-3)
        _d.min = min(1.0, _a.value + 1e-3)

    def _recompute_b_c_from_ad():
        nonlocal _b_val, _c_val
        _b_val = _a.value + (_d.value - _a.value)/3.0
        _c_val = _a.value + 2.0*(_d.value - _a.value)/3.0
        _b_label.value = f"b = {_b_val:.3f}"
        _c_label.value = f"c = {_c_val:.3f}"

    def _legend_with_values(_fn):
        _handles = [Line2D([0], [0], lw=2.0, label=_dd.value)]
        for _name, _x in [('a', _a.value), ('b', _b_val), ('c', _c_val), ('d', _d.value)]:
            _col = _colors[_name]
            _Lx = _fn(_x)
            _handles.append(Line2D([0], [0], color=_col, lw=1.8, linestyle='--',
                                   label=f"{_name}={_x:.3f}, L={_Lx:.3f}"))
        return _handles

    def _plot_now():
        with _out:
            _out.clear_output(wait=True)
            _phi = np.linspace(_PHI_LO, _PHI_HI, _N)
            _fn = _FUNCTIONS[_dd.value]
            _y = _fn(_phi)

            plt.figure(figsize=(9.2, 5.4))
            plt.plot(_phi, _y)

            for _name, _x in [('a', _a.value), ('b', _b_val), ('c', _c_val), ('d', _d.value)]:
                _col = _colors[_name]
                _Lx = _fn(_x)
                plt.axvline(_x, linestyle='--', linewidth=1.5, color=_col)
                plt.scatter([_x], [_Lx], s=40, color=_col, zorder=5)
                plt.text(_x, _Lx, f"  {_name}: L={_Lx:.3f}", color=_col, va='bottom', ha='left')

            plt.xlabel("ϕ"); plt.ylabel("Vrijednost"); plt.grid(True)
            plt.title(f"{_dd.value}  |  a={_a.value:.3f}, b={_b_val:.3f}, c={_c_val:.3f}, d={_d.value:.3f}")
            plt.legend(handles=_legend_with_values(_FUNCTIONS[_dd.value]), loc="best")
            plt.tight_layout(); plt.show()

    def _reset_to_initial_and_plot(_=None):
        _a.value = 0.0; _d.value = 1.0
        _ensure_order_and_bounds(); _recompute_b_c_from_ad()
        _plot_now()
        _status.value = "Resetirano na početne vrijednosti i iscrtano."

    def _on_next(_=None):
        nonlocal _b_val, _c_val
        try:
            _ensure_order_and_bounds()
            _fn = _FUNCTIONS[_dd.value]
            _a_new, _b_new, _c_new, _d_new = bracketing_step(_fn, _a.value, _b_val, _c_val, _d.value)
            _a.value, _d.value = _a_new, _d_new
            _b_val, _c_val = _b_new, _c_new
            _b_label.value = f"b = {_b_val:.3f}"; _c_label.value = f"c = {_c_val:.3f}"
            _ensure_order_and_bounds(); _plot_now()
            _status.value = ""
        except Exception as _e:
            _status.value = f"Greška u bracketing_step: {type(_e).__name__}: {_e}"

    def _on_converge(_=None):
        """Iteriraj bracketing_step dok (d-a) ≤ tol ili do max_iter, pa iscrtaj."""
        nonlocal _b_val, _c_val
        try:
            _ensure_order_and_bounds()
            _fn = _FUNCTIONS[_dd.value]
            _tol_val = float(_tol.value)
            _maxi = int(_max_iter.value)

            _a_loc, _b_loc, _c_loc, _d_loc = _a.value, _b_val, _c_val, _d.value
            _steps = 0
            while (_d_loc - _a_loc) > _tol_val and _steps < _maxi:
                _a_loc, _b_loc, _c_loc, _d_loc = bracketing_step(_fn, _a_loc, _b_loc, _c_loc, _d_loc)
                _steps += 1

            # ažuriraj prikaz s konvergiranim granicama
            _a.value, _d.value = _a_loc, _d_loc
            _b_val, _c_val = _b_loc, _c_loc
            _b_label.value = f"b = {_b_val:.3f}"; _c_label.value = f"c = {_c_val:.3f}"
            _ensure_order_and_bounds(); _plot_now()
            _status.value = f"Konvergirano u {_steps} koraka (tol={_tol_val:g})."
        except Exception as _e:
            _status.value = f"Greška u konvergenciji: {type(_e).__name__}: {_e}"

    # (Alternativa: ako želite striktno koristiti vašu bracketing_min_search,
    # dopunite je da VRAĆA i (a,b,c,d), npr. return (_x_star, _L_star, _a, _b, _c, _d, _history).)

    # Auto-plot pri promjeni a ili d
    def _on_a_change(_):
        _ensure_order_and_bounds(); _recompute_b_c_from_ad(); _plot_now()
    def _on_d_change(_):
        _ensure_order_and_bounds(); _recompute_b_c_from_ad(); _plot_now()

    _a.observe(_on_a_change, names='value')
    _d.observe(_on_d_change, names='value')
    _btn_plot.on_click(_reset_to_initial_and_plot)
    _btn_next.on_click(_on_next)
    _btn_conv.on_click(_on_converge)

    _row_top  = HBox([_dd, _btn_plot, _btn_next, _btn_conv, _lbl], layout=Layout(align_items="center"))
    _row_ad   = HBox([_a, _d], layout=Layout(align_items="center"))
    _row_ctl  = HBox([_tol, _max_iter], layout=Layout(align_items="center"))
    _row_bc   = HBox([Label("b i c se izračunavaju:"), _b_label, _c_label])
    _ui = VBox([_row_top, _row_ad, _row_ctl, _row_bc, _status, _out])

    _reset_to_initial_and_plot()
    display(_ui)