# Knjižnica
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from ipywidgets import Dropdown, HBox, VBox, Output, FloatText, Layout, Button
from IPython.display import display, clear_output
from torch import  nn
from matplotlib.gridspec import GridSpec

# Add project files
current_file_path = os.path.abspath(__file__)
main_dir =  os.path.dirname(current_file_path)
print(main_dir)
sys.path.append(main_dir)

from dataloader import *
from models import *


def draw_2D_function(ax, x1_mesh, x2_mesh, y):
    _pos = ax.contourf(x1_mesh, x2_mesh, y, levels=256, cmap='viridis', vmin=-10, vmax=10.0)
    ax.set_xlabel('x1'); ax.set_ylabel('x2')
    _levels = np.arange(-10, 10, 1.0)
    ax.contour(x1_mesh, x2_mesh, y, _levels, cmap='viridis')
    return _pos  # <-- return the mappable so we can make a colorbar


def create_model_2_3_1_component_plots_widget(device: str = "cpu"):
    # --- sizing knobs ---
    _FIELD_W = "200px"
    _DESC_W  = "115px"
    _style = {"description_width": _DESC_W}
    _box   = lambda: {"width": _FIELD_W}

    # ---- Initial values (your spec) ----
    # theta
    _init_t10, _init_t11, _init_t12 = -4.0,  0.9,  0.0
    _init_t20, _init_t21, _init_t22 =  5.0, -0.9, -0.5
    _init_t30, _init_t31, _init_t32 = -7.0,  0.5,  0.9

    # psi (use phi_10..phi_13 by default)
    _init_p0, _init_p1, _init_p2, _init_p3 = 0.0, -2.0, 2.0, 1.5
    # If you want the alternative set, uncomment the next line:
    # _init_p0, _init_p1, _init_p2, _init_p3 = -2.0, -1.0, -2.0, 0.8

    # --- Controls (exact numeric inputs) ---
    _t10 = FloatText(description="θ10", style=_style, value=_init_t10, layout=_box())
    _t11 = FloatText(description="θ11 (x1)", style=_style, value=_init_t11, layout=_box())
    _t12 = FloatText(description="θ12 (x2)", style=_style, value=_init_t12, layout=_box())

    _t20 = FloatText(description="θ20", style=_style, value=_init_t20, layout=_box())
    _t21 = FloatText(description="θ21 (x1)", style=_style, value=_init_t21, layout=_box())
    _t22 = FloatText(description="θ22 (x2)", style=_style, value=_init_t22, layout=_box())

    _t30 = FloatText(description="θ30", style=_style, value=_init_t30, layout=_box())
    _t31 = FloatText(description="θ31 (x1)", style=_style, value=_init_t31, layout=_box())
    _t32 = FloatText(description="θ32 (x2)", style=_style, value=_init_t32, layout=_box())

    _p0  = FloatText(description="ψ0", style=_style, value=_init_p0, layout=_box())
    _p1  = FloatText(description="ψ1", style=_style, value=_init_p1, layout=_box())
    _p2  = FloatText(description="ψ2", style=_style, value=_init_p2, layout=_box())
    _p3  = FloatText(description="ψ3", style=_style, value=_init_p3, layout=_box())

    _dd_act = Dropdown(description="activation", style=_style,
                       options=["none","relu","tanh","sigmoid"],
                       value="relu", layout={"width":"220px"})
    
    # Default x₁/x₂ ranges → 0.0 → 10.0 step 0.1
    _x1_start = FloatText(description="x1 start", style=_style, value=0.0, layout=_box())
    _x1_stop  = FloatText(description="x1 stop",  style=_style, value=10.0, layout=_box())
    _x1_step  = FloatText(description="x1 step",  style=_style, value=0.1, layout=_box())

    _x2_start = FloatText(description="x2 start", style=_style, value=0.0, layout=_box())
    _x2_stop  = FloatText(description="x2 stop",  style=_style, value=10.0, layout=_box())
    _x2_step  = FloatText(description="x2 step",  style=_style, value=0.1, layout=_box())

    _btn_plot = Button(description="Plot 2D", layout={"width":"160px", "height":"36px"})
    _out = Output(layout={"border": "1px solid #ddd"})

    # ---- draw_2D_function assumed defined above ----
    def _on_plot_clicked(_=None):
        _out.clear_output(wait=True)
        with _out:
            try:
                _x1s = np.arange(float(_x1_start.value), float(_x1_stop.value), float(_x1_step.value))
                _x2s = np.arange(float(_x2_start.value), float(_x2_stop.value), float(_x2_step.value))
            except Exception as _e:
                print(f"Invalid range(s): {_e}")
                return
            if _x1s.size == 0 or _x2s.size == 0:
                print("x1 or x2 range is empty. Check start/stop/step.")
                return

            _x1_mesh, _x2_mesh = np.meshgrid(_x1s, _x2s, indexing="xy")

            _theta_init = np.array([
                [_t10.value, _t11.value, _t12.value],
                [_t20.value, _t21.value, _t22.value],
                [_t30.value, _t31.value, _t32.value],
            ], dtype=float)

            _psi_init = np.array([_p0.value, _p1.value, _p2.value, _p3.value], dtype=float)
            _act = None if _dd_act.value == "none" else _dd_act.value

            _model = model_2_3_1_for_plots(theta_init_matrix=_theta_init,
                                           psi_init_matrix=_psi_init,
                                           activation_function=_act).to(device).eval()

            with torch.no_grad():
                _x1_t = torch.tensor(_x1_mesh, dtype=torch.float32, device=device)
                _x2_t = torch.tensor(_x2_mesh, dtype=torch.float32, device=device)
                _y, _z1, _z2, _z3, _h1, _h2, _h3, _w1, _w2, _w3 = _model(_x1_t, _x2_t)

            _to_np = lambda _t: _t.detach().cpu().numpy().astype(float)
            _grid_data = [
                ("z1 = θ10 + θ11·x1 + θ12·x2", _to_np(_z1)),
                ("z2 = θ20 + θ21·x1 + θ22·x2", _to_np(_z2)),
                ("z3 = θ30 + θ31·x1 + θ32·x2", _to_np(_z3)),
                ("h1 = a1(z1)",                _to_np(_h1)),
                ("h2 = a2(z2)",                _to_np(_h2)),
                ("h3 = a3(z3)",                _to_np(_h3)),
                ("w1 = ψ1·h1",                 _to_np(_w1)),
                ("w2 = ψ2·h2",                 _to_np(_w2)),
                ("w3 = ψ3·h3",                 _to_np(_w3)),
            ]
            _y_np = _to_np(_y)
            _psi0_np = np.full_like(_y_np, float(_p0.value))

            _fig = plt.figure(figsize=(13.5, 18))
            _gs = GridSpec(4, 3, figure=_fig)

            for _i, (_title, _vals) in enumerate(_grid_data):
                _r, _c = divmod(_i, 3)
                _ax = _fig.add_subplot(_gs[_r, _c])
                draw_2D_function(_ax, _x1_mesh, _x2_mesh, _vals)
                _ax.set_title(_title, fontsize=11)

            # Bottom row: left = ψ0, center = y, right = colorbar
            _ax_psi0 = _fig.add_subplot(_gs[3, 0])
            draw_2D_function(_ax_psi0, _x1_mesh, _x2_mesh, _psi0_np)
            _ax_psi0.set_title("w0 = ψ0 (bias)", fontsize=11)
            _ax_psi0.set_aspect('equal', adjustable='box')

            _ax_wide = _fig.add_subplot(_gs[3, 1])
            _pos = draw_2D_function(_ax_wide, _x1_mesh, _x2_mesh, _y_np)
            _ax_wide.set_title("y  (final output)", fontsize=12)
            _ax_wide.set_aspect('equal', adjustable='box')
            _ax_wide.set_anchor('C')  # <-- center the axes within its slot
            # heatmap legend (colorbar) on the last row
            _cbar = _fig.colorbar(_pos, ax=_ax_wide, orientation='vertical', fraction=0.046, pad=0.04)
            _cbar.set_label("value", rotation=90)
            _fig.tight_layout()
            display(_fig)
            plt.close(_fig)

    _btn_plot.on_click(_on_plot_clicked)
    _on_plot_clicked()

    _ui = VBox([
        HBox([_t10, _t11, _t12]),
        HBox([_t20, _t21, _t22]),
        HBox([_t30, _t31, _t32]),
        HBox([_p0, _p1, _p2, _p3, _dd_act]),
        HBox([_x1_start, _x1_stop, _x1_step]),
        HBox([_x2_start, _x2_stop, _x2_step, _btn_plot]),
        _out
    ], layout={"width": "100%"})
    display(_ui)


class model_1_3_2_for_plots(nn.Module):
    """
    Jednostavan model s 1 ulaznim neuronom, 3 skrivena neurona i 2 izlaza.

    Zajednički skriveni sloj:
      z1 = t10 + t11 * x
      z2 = t20 + t21 * x
      z3 = t30 + t31 * x
      h_i = a_i(z_i)

    Dva izlaza (različiti psi koeficijenti):
      y1 = psi0a + psi1a*h1 + psi2a*h2 + psi3a*h3
      y2 = psi0b + psi1b*h1 + psi2b*h2 + psi3b*h3
    """
    def __init__(self,
                 theta_init_matrix: np.ndarray = None,   # shape (3,2): [[t10,t11],[t20,t21],[t30,t31]]
                 psi_init_matrix:   np.ndarray = None,   # shape (2,4): [[psi0a,psi1a,psi2a,psi3a],
                                                          #               [psi0b,psi1b,psi2b,psi3b]]
                 activation_function: str = None):
        super().__init__()
        if theta_init_matrix is None:
            theta_init_matrix = np.zeros((3, 2), dtype=np.float32)
        if psi_init_matrix is None:
            psi_init_matrix = np.zeros((2, 4), dtype=np.float32)

        assert theta_init_matrix.shape == (3, 2), "theta_init_matrix must be (3,2)"
        assert psi_init_matrix.shape == (2, 4), "psi_init_matrix must be (2,4)"

        # Theta
        self.theta10 = nn.Parameter(torch.tensor(theta_init_matrix[0, 0], dtype=torch.float32))
        self.theta11 = nn.Parameter(torch.tensor(theta_init_matrix[0, 1], dtype=torch.float32))
        self.theta20 = nn.Parameter(torch.tensor(theta_init_matrix[1, 0], dtype=torch.float32))
        self.theta21 = nn.Parameter(torch.tensor(theta_init_matrix[1, 1], dtype=torch.float32))
        self.theta30 = nn.Parameter(torch.tensor(theta_init_matrix[2, 0], dtype=torch.float32))
        self.theta31 = nn.Parameter(torch.tensor(theta_init_matrix[2, 1], dtype=torch.float32))

        # Psi for y1 (suffix a)
        self.psi0a = nn.Parameter(torch.tensor(psi_init_matrix[0, 0], dtype=torch.float32))
        self.psi1a = nn.Parameter(torch.tensor(psi_init_matrix[0, 1], dtype=torch.float32))
        self.psi2a = nn.Parameter(torch.tensor(psi_init_matrix[0, 2], dtype=torch.float32))
        self.psi3a = nn.Parameter(torch.tensor(psi_init_matrix[0, 3], dtype=torch.float32))
        # Psi for y2 (suffix b)
        self.psi0b = nn.Parameter(torch.tensor(psi_init_matrix[1, 0], dtype=torch.float32))
        self.psi1b = nn.Parameter(torch.tensor(psi_init_matrix[1, 1], dtype=torch.float32))
        self.psi2b = nn.Parameter(torch.tensor(psi_init_matrix[1, 2], dtype=torch.float32))
        self.psi3b = nn.Parameter(torch.tensor(psi_init_matrix[1, 3], dtype=torch.float32))

        # Activation
        _act = nn.Identity
        if activation_function is not None:
            _af = activation_function.lower()
            if _af == "relu":    _act = nn.ReLU
            if _af == "tanh":    _act = nn.Tanh
            if _af == "sigmoid": _act = nn.Sigmoid
        self.activation1 = _act()
        self.activation2 = _act()
        self.activation3 = _act()

    def forward(self, x: torch.Tensor):
        """
        Returns:
          (y1, y2, z1, z2, z3, h1, h2, h3, w1a, w2a, w3a)
        """
        _x = torch.as_tensor(x, device=self.theta10.device, dtype=self.theta10.dtype)

        # affine
        _z1 = self.theta10 + self.theta11 * _x
        _z2 = self.theta20 + self.theta21 * _x
        _z3 = self.theta30 + self.theta31 * _x

        # activations
        _h1 = self.activation1(_z1)
        _h2 = self.activation2(_z2)
        _h3 = self.activation3(_z3)

        # weighted activations for y1 (a)
        _w1a = self.psi1a * _h1
        _w2a = self.psi2a * _h2
        _w3a = self.psi3a * _h3

        # outputs
        _y1 = self.psi0a + _w1a + _w2a + _w3a
        _y2 = self.psi0b + self.psi1b * _h1 + self.psi2b * _h2 + self.psi3b * _h3

        return _y1, _y2, _z1, _z2, _z3, _h1, _h2, _h3, _w1a, _w2a, _w3a


# ---------- WIDGET (z,h + merged wᵃ&wᵇ per unit + merged ψ0ᵃ&ψ0ᵇ + compact y1&y2) ----------
def create_model_1_3_2_component_plots_widget(device: str = "cpu"):
    """
    Interaktivni widget za model_1_3_2_for_plots:
      - Unos svih hiperparametara (theta i psi_a/psi_b)
      - Odabir aktivacije (none/relu/tanh/sigmoid)
      - Unos intervala x (start, stop, step)
      - Crta:
          row1: z1..z3
          row2: h1..h3
          row3: w1 (ψ1ᵃ·h1 & ψ1ᵇ·h1), w2 (ψ2ᵃ·h2 & ψ2ᵇ·h2), w3 (ψ3ᵃ·h3 & ψ3ᵇ·h3)
          row4: ψ0ᵃ & ψ0ᵇ (left), y₁ & y₂ outputs (middle), empty (right)
    """

    # --- Controls (predefined defaults) ---
    from ipywidgets import FloatText, Dropdown, ToggleButton, Button, Output, HBox, VBox
    import numpy as np
    import torch
    from matplotlib import pyplot as plt
    from matplotlib.gridspec import GridSpec
    from IPython.display import display

    _t10 = FloatText(description="θ10", value=-4.0, layout={"width":"140px"})
    _t11 = FloatText(description="θ11", value= 0.9, layout={"width":"140px"})
    _t20 = FloatText(description="θ20", value= 5.0, layout={"width":"140px"})
    _t21 = FloatText(description="θ21", value=-0.9, layout={"width":"140px"})
    _t30 = FloatText(description="θ30", value=-7.0, layout={"width":"140px"})
    _t31 = FloatText(description="θ31", value= 0.5, layout={"width":"140px"})

    _p0a = FloatText(description="ψ0ᵃ", value= 0.0, layout={"width":"140px"})
    _p1a = FloatText(description="ψ1ᵃ", value=-2.0, layout={"width":"140px"})
    _p2a = FloatText(description="ψ2ᵃ", value= 2.0, layout={"width":"140px"})
    _p3a = FloatText(description="ψ3ᵃ", value= 1.5, layout={"width":"140px"})

    _p0b = FloatText(description="ψ0ᵇ", value=-2.0, layout={"width":"140px"})
    _p1b = FloatText(description="ψ1ᵇ", value=-1.0, layout={"width":"140px"})
    _p2b = FloatText(description="ψ2ᵇ", value=-2.0, layout={"width":"140px"})
    _p3b = FloatText(description="ψ3ᵇ", value= 0.8, layout={"width":"140px"})

    _dd_act = Dropdown(description="activation",
                       options=["none","relu","tanh","sigmoid"],
                       value="relu", layout={"width":"200px"})

    _x_start = FloatText(description="x start", value=0.0,  layout={"width":"140px"})
    _x_stop  = FloatText(description="x stop",  value=1.0,  layout={"width":"140px"})
    _x_step  = FloatText(description="x step",  value=0.01, layout={"width":"140px"})

    _lock_y = ToggleButton(description="lock y", value=False, layout={"width":"100px"})
    _y_min  = FloatText(description="y min", value=-5.0, layout={"width":"200px"})
    _y_max  = FloatText(description="y max", value= 5.0, layout={"width":"200px"})

    _btn_plot = Button(description="Plot", layout={"width":"120px"})
    _out = Output()

    def _on_plot_clicked(_=None):
        _out.clear_output(wait=True)
        with _out:
            # x-range
            try:
                _xs = np.arange(float(_x_start.value), float(_x_stop.value), float(_x_step.value))
            except Exception as _e:
                print(f"Invalid x range: {_e}")
                return
            if _xs.size == 0:
                print("x range is empty. Check start/stop/step.")
                return

            # params
            _theta_init = np.array([
                [_t10.value, _t11.value],
                [_t20.value, _t21.value],
                [_t30.value, _t31.value],
            ], dtype=float)

            _psi_init = np.array([
                [_p0a.value, _p1a.value, _p2a.value, _p3a.value],  # y1
                [_p0b.value, _p1b.value, _p2b.value, _p3b.value],  # y2
            ], dtype=float)

            _act = None if _dd_act.value == "none" else _dd_act.value

            # model
            _model = model_1_3_2_for_plots(theta_init_matrix=_theta_init,
                                           psi_init_matrix=_psi_init,
                                           activation_function=_act).to(device).eval()

            # forward
            with torch.no_grad():
                _x_t = torch.tensor(_xs, dtype=torch.float32, device=device)
                _y1, _y2, _z1, _z2, _z3, _h1, _h2, _h3, _w1a, _w2a, _w3a = _model(_x_t)

            # numpy conversions
            _to_np = lambda _t: _t.detach().cpu().numpy().astype(float)
            _z1_np, _z2_np, _z3_np = _to_np(_z1), _to_np(_z2), _to_np(_z3)
            _h1_np, _h2_np, _h3_np = _to_np(_h1), _to_np(_h2), _to_np(_h3)
            _w1a_np, _w2a_np, _w3a_np = _to_np(_w1a), _to_np(_w2a), _to_np(_w3a)
            _y1_np, _y2_np = _to_np(_y1), _to_np(_y2)

            # φ-components (b) using model params
            _w1b_np = _to_np(_model.psi1b * _h1)
            _w2b_np = _to_np(_model.psi2b * _h2)
            _w3b_np = _to_np(_model.psi3b * _h3)

            # ψ0 lines
            _psi0a_np = np.full_like(_xs, float(_p0a.value))
            _psi0b_np = np.full_like(_xs, float(_p0b.value))

            # colors
            _col_y1, _col_y2 = "C0", "C1"

            # y-limits
            _x_lo, _x_hi = float(_x_start.value), float(_x_stop.value)
            if _lock_y.value:
                _y_lo, _y_hi = float(_y_min.value), float(_y_max.value)
            else:
                _all = np.concatenate([
                    _z1_np, _z2_np, _z3_np,
                    _h1_np, _h2_np, _h3_np,
                    _w1a_np, _w2a_np, _w3a_np,
                    _w1b_np, _w2b_np, _w3b_np,
                    _psi0a_np, _psi0b_np,
                    _y1_np, _y2_np
                ])
                _y_lo, _y_hi = (_all.min(), _all.max()) if _all.size else (-1.0, 1.0)
                _pad = 0.05 * (_y_hi - _y_lo if _y_hi > _y_lo else 1.0)
                _y_lo, _y_hi = _y_lo - _pad, _y_hi + _pad
                _y_min.value, _y_max.value = float(_y_lo), float(_y_hi)

            # figure: 4 rows × 3 cols
            _fig = plt.figure(figsize=(12, 18))
            _gs = GridSpec(4, 3, figure=_fig)

            # Row 1: z1..z3
            for _c, (_title, _vals) in enumerate([
                ("z1 = θ10 + θ11·x", _z1_np),
                ("z2 = θ20 + θ21·x", _z2_np),
                ("z3 = θ30 + θ31·x", _z3_np),
            ]):
                _ax = _fig.add_subplot(_gs[0, _c])
                _ax.plot(_xs, _vals)
                _ax.set_title(_title, fontsize=10)
                _ax.grid(True, linestyle="--", alpha=0.3)
                _ax.set_xlim(_x_lo, _x_hi); _ax.set_ylim(_y_lo, _y_hi)
                _ax.set_aspect('auto')
                if _c == 0: _ax.set_ylabel("value")

            # Row 2: h1..h3
            for _c, (_title, _vals) in enumerate([
                ("h1 = a1(z1)", _h1_np),
                ("h2 = a2(z2)", _h2_np),
                ("h3 = a3(z3)", _h3_np),
            ]):
                _ax = _fig.add_subplot(_gs[1, _c])
                _ax.plot(_xs, _vals)
                _ax.set_title(_title, fontsize=10)
                _ax.grid(True, linestyle="--", alpha=0.3)
                _ax.set_xlim(_x_lo, _x_hi); _ax.set_ylim(_y_lo, _y_hi)
                _ax.set_aspect('auto')
                if _c == 0: _ax.set_ylabel("value")

            # Row 3: merged wᵃ & wᵇ per hidden unit
            _ax_w1 = _fig.add_subplot(_gs[2, 0])
            _ax_w1.plot(_xs, _w1a_np, color=_col_y1, linewidth=2, label="ψ1ᵃ·h1")
            _ax_w1.plot(_xs, _w1b_np, color=_col_y2, linewidth=2, label="ψ1ᵇ·h1")
            _ax_w1.set_title("w1: (ψ1ᵃ·h1, ψ1ᵇ·h1)", fontsize=10)
            _ax_w1.grid(True, linestyle="--", alpha=0.3)
            _ax_w1.set_xlim(_x_lo, _x_hi); _ax_w1.set_ylim(_y_lo, _y_hi)
            _ax_w1.set_aspect('auto'); _ax_w1.set_xlabel("x"); _ax_w1.set_ylabel("value")
            _ax_w1.legend(loc="best", frameon=True)

            _ax_w2 = _fig.add_subplot(_gs[2, 1])
            _ax_w2.plot(_xs, _w2a_np, color=_col_y1, linewidth=2, label="ψ2ᵃ·h2")
            _ax_w2.plot(_xs, _w2b_np, color=_col_y2, linewidth=2, label="ψ2ᵇ·h2")
            _ax_w2.set_title("w2: (ψ2ᵃ·h2, ψ2ᵇ·h2)", fontsize=10)
            _ax_w2.grid(True, linestyle="--", alpha=0.3)
            _ax_w2.set_xlim(_x_lo, _x_hi); _ax_w2.set_ylim(_y_lo, _y_hi)
            _ax_w2.set_aspect('auto'); _ax_w2.set_xlabel("x")
            _ax_w2.legend(loc="best", frameon=True)

            _ax_w3 = _fig.add_subplot(_gs[2, 2])
            _ax_w3.plot(_xs, _w3a_np, color=_col_y1, linewidth=2, label="ψ3ᵃ·h3")
            _ax_w3.plot(_xs, _w3b_np, color=_col_y2, linewidth=2, label="ψ3ᵇ·h3")
            _ax_w3.set_title("w3: (ψ3ᵃ·h3, ψ3ᵇ·h3)", fontsize=10)
            _ax_w3.grid(True, linestyle="--", alpha=0.3)
            _ax_w3.set_xlim(_x_lo, _x_hi); _ax_w3.set_ylim(_y_lo, _y_hi)
            _ax_w3.set_aspect('auto'); _ax_w3.set_xlabel("x")
            _ax_w3.legend(loc="best", frameon=True)

            # Row 4: merged ψ0 (left), outputs y1&y2 (middle), empty (right)
            _ax_p0 = _fig.add_subplot(_gs[3, 0])
            _ax_p0.plot(_xs, _psi0a_np, color=_col_y1, linewidth=2, label="ψ0ᵃ")
            _ax_p0.plot(_xs, _psi0b_np, color=_col_y2, linewidth=2, label="ψ0ᵇ")
            _ax_p0.set_title("w0 = ψ0 (bias): y1 vs y2", fontsize=10)
            _ax_p0.grid(True, linestyle="--", alpha=0.3)
            _ax_p0.set_xlim(_x_lo, _x_hi); _ax_p0.set_ylim(_y_lo, _y_hi)
            _ax_p0.set_aspect('auto'); _ax_p0.set_xlabel("x"); _ax_p0.set_ylabel("value")
            _ax_p0.legend(loc="best", frameon=True)

            _ax_y = _fig.add_subplot(_gs[3, 1])
            _ax_y.plot(_xs, _y1_np, label="y1", color=_col_y1, linewidth=2)
            _ax_y.plot(_xs, _y2_np, label="y2", color=_col_y2, linewidth=2)
            _ax_y.set_title("y₁ & y₂ (outputs)", fontsize=10)
            _ax_y.grid(True, linestyle="--", alpha=0.3)
            _ax_y.set_xlim(_x_lo, _x_hi); _ax_y.set_ylim(_y_lo, _y_hi)
            _ax_y.set_aspect('auto'); _ax_y.set_xlabel("x")
            _ax_y.legend(loc="best", frameon=True)

            _ax_empty2 = _fig.add_subplot(_gs[3, 2]); _ax_empty2.axis("off")

            _fig.tight_layout()
            display(_fig)
            plt.close(_fig)

    _btn_plot.on_click(_on_plot_clicked)

    # Initial draw
    _on_plot_clicked()

    _ui = VBox([
        HBox([_t10, _t11, _t20]),
        HBox([_t21, _t30, _t31]),
        HBox([_p0a, _p1a, _p2a, _p3a]),
        HBox([_p0b, _p1b, _p2b, _p3b]),
        HBox([_dd_act, _x_start, _x_stop, _x_step]),
        HBox([_lock_y, _y_min, _y_max, _btn_plot]),
        _out
    ])
    display(_ui)


# --- Your model class must be defined above this cell ---
# class model_1_3_1(nn.Module): ...

def create_model_1_3_1_presets_widget(device: str = "cpu"):
    """
    Widget:
      - Activation is fixed to ReLU
      - Choose two presets (Left & Right) from: net1, net2, net3, net4
      - Each preset has a consistent color
      - Builds model_1_3_1 for both, evaluates on x ∈ [-1,1) with step 0.001
      - Plots three subplots:
          1) Left net output
          2) Right net output
          3) Composition: Right(Left(x))   i.e., net₂(net₁(x))
    """

    # --- Presets (θ and ψ) ---
    # net1
    _n1_theta = np.array([
        [ 0.00, -1.0],
        [ 0.00,  1.0],
        [-0.67,  1.0],
    ], dtype=float)
    _n1_phi = np.array([1.0, -2.0, -3.0, 9.3], dtype=float)

    # net2
    _n2_theta = np.array([
        [-0.60, -1.0],
        [ 0.20,  1.0],
        [-0.50,  1.0],
    ], dtype=float)
    _n2_phi = np.array([0.5, -1.0, -1.5, 2.0], dtype=float)

    # net3 = net2 with φ1 negated
    _n3_theta = _n2_theta.copy()
    _n3_phi   = _n2_phi.copy()
    _n3_phi[1] = -_n3_phi[1]   # -(-1.0) = +1.0

    # net4 = net1 with φ1 halved
    _n4_theta = _n1_theta.copy()
    _n4_phi   = _n1_phi.copy()
    _n4_phi[1] = _n4_phi[1] * 0.5  # -2.0 * 0.5 = -1.0

    _presets = {
        "net1": (_n1_theta, _n1_phi),
        "net2": (_n2_theta, _n2_phi),
        "net3 (net2 with φ1→-φ1)": (_n3_theta, _n3_phi),
        "net4 (net1 with φ1×0.5)":  (_n4_theta, _n4_phi),
    }

    # Consistent color per network
    _color_map = {
        "net1": "C0",
        "net2": "C1",
        "net3 (net2 with φ1→-φ1)": "C2",
        "net4 (net1 with φ1×0.5)":  "C3",
    }
    _comp_color = "C4"  # composition color

    # --- Controls ---
    _dd_left = Dropdown(
        description="Left",
        options=list(_presets.keys()),
        value="net1",
        layout=Layout(width="300px")
    )
    _dd_right = Dropdown(
        description="Right",
        options=list(_presets.keys()),
        value="net2",
        layout=Layout(width="300px")
    )
    _btn_plot = Button(description="Build & Plot", layout=Layout(width="140px", height="34px"))
    _out = Output()

    _controls = HBox([_dd_left, _dd_right, _btn_plot])

    def _build_model(preset_key: str):
        theta_init, psi_init = _presets[preset_key]
        model = model_1_3_1(
            theta_init_matrix=theta_init,
            psi_init_matrix=psi_init,
            activation_function="relu"  # fixed
        ).to(device).eval()
        return model, theta_init, psi_init

    def _on_plot_clicked(_=None):
        _out.clear_output(wait=True)
        with _out:
            left_key  = _dd_left.value
            right_key = _dd_right.value

            # Build models
            left_model,  th_l, ps_l  = _build_model(left_key)
            right_model, th_r, ps_r  = _build_model(right_key)

            # Input grid
            x = np.arange(-1.0, 1.0, 0.001, dtype=np.float32)
            x_t = torch.tensor(x, device=device)

            # Forward + composition
            with torch.no_grad():
                y_left_t  = left_model(x_t)
                y_right_t = right_model(x_t)
                y_comp_t  = right_model(y_left_t)  # composition Right(Left(x))

            y_left  = y_left_t.detach().cpu().numpy().astype(float)
            y_right = y_right_t.detach().cpu().numpy().astype(float)
            y_comp  = y_comp_t.detach().cpu().numpy().astype(float)

            # Colors
            col_l = _color_map[left_key]
            col_r = _color_map[right_key]

            # Common y-limits across all three panels (so nothing clips)
            y_min = float(min(y_left.min(), y_right.min(), y_comp.min()))
            y_max = float(max(y_left.max(), y_right.max(), y_comp.max()))
            pad = 0.05 * (y_max - y_min if y_max > y_min else 1.0)
            y_lo, y_hi = y_min - pad, y_max + pad

            # Plot three subplots
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True)

            ax1.plot(x, y_left, color=col_l, linewidth=2)
            ax1.set_title(f"{left_key} | activation=relu")
            ax1.set_xlabel("x")
            ax1.set_ylabel("y")
            ax1.grid(True, linestyle="--", alpha=0.3)
            ax1.set_ylim(y_lo, y_hi)

            ax2.plot(x, y_right, color=col_r, linewidth=2)
            ax2.set_title(f"{right_key} | activation=relu")
            ax2.set_xlabel("x")
            ax2.grid(True, linestyle="--", alpha=0.3)
            ax2.set_ylim(y_lo, y_hi)

            ax3.plot(x, y_comp, color=_comp_color, linewidth=2)
            ax3.set_title(f"Composition: {right_key}({left_key}(x))")
            ax3.set_xlabel("x")
            ax3.grid(True, linestyle="--", alpha=0.3)
            ax3.set_ylim(y_lo, y_hi)

            fig.tight_layout()
            display(fig)
            plt.close(fig)

            # Print the parameters underneath for reference
            print("LEFT preset:", left_key)
            print("θ_left =\n", th_l)
            print("ψ_left =\n", ps_l)
            print()
            print("RIGHT preset:", right_key)
            print("θ_right =\n", th_r)
            print("ψ_right =\n", ps_r)
            print()
            print("Composition shown is: Right(Left(x))")

    _btn_plot.on_click(_on_plot_clicked)

    # initial draw
    _on_plot_clicked()

    display(VBox([_controls, _out]))
