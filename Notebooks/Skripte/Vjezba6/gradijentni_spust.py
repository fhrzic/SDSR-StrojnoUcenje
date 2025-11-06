"""
Skripta u kojoj se implementira linearna pretraga
"""

# Knjižnica
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RangeSlider
import os
import sys
from typing import Callable, Tuple, Optional, List, Dict
from matplotlib.lines import Line2D  # for legend entries
from ipywidgets import FloatText, FloatSlider,IntSlider, HBox, VBox, Output, Label, Layout
from IPython.display import display
from matplotlib import cm
from ipywidgets import interact, widgets
from matplotlib.colors import ListedColormap

# Add project files
current_file_path = os.path.abspath(__file__)
main_dir =  os.path.dirname(os.path.dirname(current_file_path))
print(main_dir)
sys.path.append(main_dir)

from Vjezba3.dataloader import *

def gradijent_derivacija(y:np.array = None, 
                         x:np.array = None, 
                         theta:np.array = None)->np.array:
    """
    Metoda koja računa gradijent linearne regresije za dane ulazne podatke kako sljedi.

    Args:
        * y, np.array, ulazni podaci koji se predviđaju.
        * x, np.array, ulazni podaci koji temeljem kojih se predviđa y.
        * theta, np.array, theta0 i theta1 --> parametri pravca
    
    Returns:
        * np.array --> gradijent za theta0 i theta1 u obliku [theta0_grad, theta1_grad]
    """

    # --- pretvorbe i provjere ---
    _y = np.asarray(y, dtype=float).reshape(-1)
    _x = np.asarray(x, dtype=float).reshape(-1)
    _theta = np.asarray(theta, dtype=float).reshape(-1)

    _dtheta0, _dtheta1 = 0, 0
    return np.array([_dtheta0, _dtheta1], dtype=float)


def gradijent_aproksimacija(y: np.ndarray = None,
                           x: np.ndarray = None,
                           theta: np.ndarray = None,
                           delta: float = 1e-7) -> np.ndarray:
    """
    Numerička (finite-difference) aproksimacija gradijenta MSE-a za linearni model:
        y_hat = theta0 + theta1 * x

    Forward-difference aproksimacije:
        dL/dtheta0 ≈ ( MSE(y, y_hat(theta0+δ, theta1)) - MSE(y, y_hat(theta0, theta1)) ) / δ
        dL/dtheta1 ≈ ( MSE(y, y_hat(theta0, theta1+δ)) - MSE(y, y_hat(theta0, theta1)) ) / δ

    Args:
        * _y (np.ndarray): ciljne vrijednosti, oblik (N,)
        * _x (np.ndarray): ulazne vrijednosti, oblik (N,)
        * _theta (np.ndarray): parametri [theta0, theta1]
        * _delta (float): mali pomak δ ( > 0 )

    Returns:
        * np.ndarray: gradijent [dtheta0, dtheta1]
    """
    # --- provjere / priprema ---
    _y = np.asarray(y, dtype=float).reshape(-1)
    _x = np.asarray(x, dtype=float).reshape(-1)
    _theta = np.asarray(theta, dtype=float).reshape(-1)

   
    _dtheta0, _dtheta1 = 0, 0

    return np.array([_dtheta0, _dtheta1], dtype=float)

def mse(_y: np.ndarray, _y_hat: np.ndarray) -> float:
    """
    Računa MSE (mean squared error) izravno iz _y i _y_hat.

    Args:
        * _y (np.ndarray): stvarne ciljne vrijednosti, oblik (N,)
        * _y_hat (np.ndarray): predikcije modela, oblik (N,)

    Returns:
        * float: MSE = (1/N) * sum( (y_hat - y)^2 )
    """
    _y = np.asarray(_y, dtype=float).reshape(-1)
    _y_hat = np.asarray(_y_hat, dtype=float).reshape(-1)

    if _y.shape[0] != _y_hat.shape[0]:
        raise ValueError("Duljine _y i _y_hat moraju biti jednake.")

    _res = _y_hat - _y
    _mse = np.mean(_res ** 2)
    return float(_mse)


def plot_bike_side_by_side(
    _path_to_csv: str = os.path.join(main_dir, "Vjezba3", "day_bikes_rental.csv"),
):
    """
    Tri interaktivna prikaza:
      (Lijevo)  Scatter temp→cnt + linearni model (crvena linija) i MSE
      (Sredina) 3D MSE površina + putanja gradijentnog spusta
      (Desno)   2D konture MSE (custom colormap) + ista putanja

    Dodatno:
      - Skriven “Figure XX” header i toolbar
      - Prikaz konačnog gradijenta (na zadnjoj GD točki)
      - Ručni unos θ₀, θ₁ preko FloatText polja (sinkronizirano sa sliderima)
    """
    # --- Tvrdo zadani stupci ---
    _var1, _var2 = "temp", "cnt"

    # --- Učitavanje podataka preko dataset/dataloader-a ---
    _dataset = bikeRentalDataset(
        path_to_csv=_path_to_csv,
        input_label=_var1,
        target_label=_var2,
        normalizacija=True,
    )
    _dataloader = ordinary_dataloader(dataset=_dataset, batch_size=1)

    _xs, _ys = [], []
    for _x_batch, _y_batch in _dataloader:
        _xs.append(_x_batch.squeeze().item())
        _ys.append(_y_batch.squeeze().item())
    _x = np.asarray(_xs, dtype=float).reshape(-1)
    _y = np.asarray(_ys, dtype=float).reshape(-1)

    # --- Kontrole ---
    _theta0 = FloatSlider(value=1.0, min=-20.0, max=20.0, step=0.01,
                          description="θ₀", layout=Layout(width="260px"), continuous_update=True)
    _theta1 = FloatSlider(value=1.0, min=-20.0, max=20.0, step=0.01,
                          description="θ₁", layout=Layout(width="260px"), continuous_update=True)
    _theta0_txt = FloatText(value=1.0, description="θ₀ (txt)", layout=Layout(width="160px"))
    _theta1_txt = FloatText(value=1.0, description="θ₁ (txt)", layout=Layout(width="160px"))

    _alpha  = FloatSlider(value=0.10, min=0.0, max=0.5, step=0.005,
                          description="α", layout=Layout(width="260px"), continuous_update=False)
    _iters  = IntSlider(value=10, min=0, max=200, step=1,
                        description="iter", layout=Layout(width="260px"), continuous_update=False)

    _mse_label = Label("MSE: —")
    _final_info = Label("Final θ & grad: —")  # prikaz konačnog theta i gradijenta

    # --- Output paneli: lijevo/sredina/desno ---
    _out_left   = Output(layout=Layout(border="1px solid #ddd", width="600px", height="520px"))
    _out_middle = Output(layout=Layout(border="1px solid #ddd", width="600px", height="520px"))
    _out_right  = Output(layout=Layout(border="1px solid #ddd", width="600px", height="520px"))

    # --- Precompute 3D/2D MSE površine (jednom) ---
    _theta0_grid = np.linspace(-20.0, 20.0, 150)
    _theta1_grid = np.linspace(-20.0, 20.0, 150)
    _t0_mesh, _t1_mesh = np.meshgrid(_theta0_grid, _theta1_grid)  # (100,100)

    def _linear(_xvec, _t0, _t1):
        return _t1 * _xvec + _t0

    _Z_vals = np.empty_like(_t0_mesh, dtype=float)
    for _i, _t0 in enumerate(_theta0_grid):
        for _j, _t1 in enumerate(_theta1_grid):
            _yhat_grid = _linear(_x, _t0, _t1)
            _Z_vals[_j, _i] = mse(_y, _yhat_grid)  # mesh indexing (row=j, col=i)

    _norm   = plt.Normalize(_Z_vals.min(), _Z_vals.max())
    _colors = cm.viridis(_norm(_Z_vals))

    # --- Custom colormap (kao u draw_loss_function) ---
    _hex_vals = (
        '2a0902','2b0a03','2c0b04','2d0c05','2e0c06','2f0d07','300d08','310e09','320f0a','330f0b','34100b',
        '35110c','36110d','37120e','38120f','39130f','3a1410','3b1411','3c1511','3d1612','3e1613','3f1713',
        '401714','411814','421915','431915','451a16','461b16','471b17','481c17','491d18','4a1d18','4b1e19',
        '4c1f19','4d1f1a','4e201b','50211b','51211c','52221c','53231d','54231d','55241e','56251e','57261f',
        '58261f','592720','5b2821','5c2821','5d2922','5e2a22','5f2b23','602b23','612c24','622d25','632e25',
        '652e26','662f26','673027','683027','693128','6a3229','6b3329','6c342a','6d342a','6f352b','70362c',
        '71372c','72372d','73382e','74392e','753a2f','763a2f','773b30','783c31','7a3d31','7b3e32','7c3e33',
        '7d3f33','7e4034','7f4134','804235','814236','824336','834437','854538','864638','874739','88473a',
        '89483a','8a493b','8b4a3c','8c4b3c','8d4c3d','8e4c3e','8f4d3f','904e3f','924f40','935041','945141',
        '955242','965343','975343','985444','995545','9a5646','9b5746','9c5847','9d5948','9e5a49','9f5a49',
        'a05b4a','a15c4b','a35d4b','a45e4c','a55f4d','a6604e','a7614e','a8624f','a96350','aa6451','ab6552',
        'ac6552','ad6653','ae6754','af6855','b06955','b16a56','b26b57','b36c58','b46d59','b56e59','b66f5a',
        'b7705b','b8715c','b9725d','ba735d','bb745e','bc755f','bd7660','be7761','bf7862','c07962','c17a63',
        'c27b64','c27c65','c37d66','c47e67','c57f68','c68068','c78169','c8826a','c9836b','ca846c','cb856d',
        'cc866e','cd876f','ce886f','ce8970','cf8a71','d08b72','d18c73','d28d74','d38e75','d48f76','d59077',
        'd59178','d69279','d7937a','d8957b','d9967b','da977c','da987d','db997e','dc9a7f','dd9b80','de9c81',
        'de9d82','df9e83','e09f84','e1a185','e2a286','e2a387','e3a488','e4a589','e5a68a','e5a78b','e6a88c',
        'e7aa8d','e7ab8e','e8ac8f','e9ad90','eaae91','eaaf92','ebb093','ecb295','ecb396','edb497','eeb598',
        'eeb699','efb79a','efb99b','f0ba9c','f1bb9d','f1bc9e','f2bd9f','f2bfa1','f3c0a2','f3c1a3','f4c2a4',
        'f5c3a5','f5c5a6','f6c6a7','f6c7a8','f7c8aa','f7c9ab','f8cbac','f8ccad','f8cdae','f9ceb0','f9d0b1',
        'fad1b2','fad2b3','fbd3b4','fbd5b6','fbd6b7','fcd7b8','fcd8b9','fcdaba','fddbbc','fddcbd','fddebe',
        'fddfbf','fee0c1','fee1c2','fee3c3','fee4c5','ffe5c6','ffe7c7','ffe8c9','ffe9ca','ffebcb','ffeccd',
        'ffedce','ffefcf','fff0d1','fff2d2','fff3d3','fff4d5','fff6d6','fff7d8','fff8d9','fffada','fffbdc',
        'fffcdd','fffedf','ffffe0'
    )
    _hex_dec = np.array([int(h, 16) for h in _hex_vals])
    _r = np.floor(_hex_dec / (256*256))
    _g = np.floor((_hex_dec - _r*256*256) / 256)
    _b = np.floor(_hex_dec - _r*256*256 - _g*256)
    _my_colormap = ListedColormap(np.vstack((_r, _g, _b)).T / 255.0)

    # --- Helper: GD putanja + konačni gradijent ---
    def _compute_gd_path_and_final_grad():
        _theta = np.array([float(_theta0.value), float(_theta1.value)], dtype=float)
        _thetas, _scores = [], []
        for _ in range(int(_iters.value)):
            _yhat = _theta[0] + _theta[1] * _x
            _scores.append(mse(_y, _yhat))
            _thetas.append(_theta.copy())
            _g = gradijent_derivacija(y=_y, x=_x, theta=_theta)
            _theta = _theta - float(_alpha.value) * _g
        # završna točka
        _yhat = _theta[0] + _theta[1] * _x
        _scores.append(mse(_y, _yhat))
        _thetas.append(_theta.copy())
        _g_final = gradijent_derivacija(y=_y, x=_x, theta=_theta)  # grad na finalnoj točki

        _t0_path = np.array([t[0] for t in _thetas])
        _t1_path = np.array([t[1] for t in _thetas])
        _scores_arr = np.array(_scores)
        return _t0_path, _t1_path, _scores_arr, _theta, _g_final

    # --- Lijevi graf ---
    def _draw_left():
        with _out_left:
            _out_left.clear_output(wait=True)
            fig = plt.figure(figsize=(8.0, 5.6))
            try:
                fig.canvas.header_visible = False  # ipympl
                fig.canvas.toolbar_visible = False
            except Exception:
                pass

            plt.scatter(_x, _y, s=25, alpha=0.9, label=f"{_var1} vs {_var2}")
            _xline = np.linspace(float(_x.min()), float(_x.max()), 300)
            _yline = _theta0.value + _theta1.value * _xline
            plt.plot(_xline, _yline, linewidth=2.0, color="red",
                     label=r"$\hat{y}=\theta_0+\theta_1 x$")

            _y_hat_pts = _theta0.value + _theta1.value * _x
            _mse_val = mse(_y, _y_hat_pts)
            _mse_label.value = f"MSE: {_mse_val:.5f}"

            plt.xlabel(_var1); plt.ylabel(_var2)
            plt.title(f"{_var1} vs {_var2}  |  θ₀={_theta0.value:.2f}, θ₁={_theta1.value:.2f}")
            plt.grid(True, linestyle="--", linewidth=0.6, alpha=0.7)
            plt.legend(loc="best")
            plt.tight_layout()
            plt.show()

    # --- Srednji graf ---
    def _draw_middle(_t0_path, _t1_path, _scores_arr):
        with _out_middle:
            _out_middle.clear_output(wait=True)
            fig = plt.figure(figsize=(8.0, 5.6))
            try:
                fig.canvas.header_visible = False
                fig.canvas.toolbar_visible = False
            except Exception:
                pass

            ax = fig.add_subplot(111, projection='3d')
            surf = ax.plot_surface(_t1_mesh, _t0_mesh, _Z_vals, rcount=20, ccount=20,
                                   facecolors=_colors, shade=False)
            surf.set_facecolor((0, 0, 0, 0))
            ax.plot(_t1_path, _t0_path, _scores_arr, 'x-', linewidth=2.0)

            ax.set_xlabel('Theta1'); ax.set_ylabel('Theta0'); ax.set_zlabel('Cijena (MSE)')
            ax.set_title('Funkcija cijene + putanja gradijentnog spusta')
            plt.tight_layout(); plt.show()

    # --- Desni graf ---
    def _draw_right(_t0_path, _t1_path):
        with _out_right:
            _out_right.clear_output(wait=True)
            fig, ax = plt.subplots(figsize=(8.0, 5.6))
            try:
                fig.canvas.header_visible = False
                fig.canvas.toolbar_visible = False
            except Exception:
                pass

            csf = ax.contourf(_t0_mesh, _t1_mesh, _Z_vals, 512, cmap=_my_colormap)
            ax.contour(_t0_mesh, _t1_mesh, _Z_vals, 60, colors=['#80808080'])
            ax.plot(_t0_path, _t1_path, 'go-', linewidth=1.8, markersize=4)

            ax.set_xlabel(r'Offset $\phi_{0}$ (theta0)')
            ax.set_ylabel(r'Frequency $\phi_{1}$ (theta1)')
            ax.set_title('Konture MSE + GD putanja')
            plt.tight_layout(); plt.show()

    # --- Refresh: izračun putanje + update sva tri grafa + tekst o finalnim vrijednostima ---
    def _refresh_all(_=None):
        _draw_left()
        _t0_path, _t1_path, _scores_arr, _theta_final, _g_final = _compute_gd_path_and_final_grad()
        _draw_middle(_t0_path, _t1_path, _scores_arr)
        _draw_right(_t0_path, _t1_path)
        _final_info.value = (
            f"Final θ: [θ₀={_theta_final[0]:.4f}, θ₁={_theta_final[1]:.4f}]   "
            f"∇MSE(final): [dθ₀={_g_final[0]:.4e}, dθ₁={_g_final[1]:.4e}]"
        )

    # --- Sync sliders <-> text polja (dvostruka sinkronizacija) ---
    _is_syncing = {"t0": False, "t1": False}

    def _on_t0_slider(chg):
        if _is_syncing["t0"]: return
        _is_syncing["t0"] = True
        _theta0_txt.value = float(_theta0.value)
        _is_syncing["t0"] = False
        _refresh_all()

    def _on_t1_slider(chg):
        if _is_syncing["t1"]: return
        _is_syncing["t1"] = True
        _theta1_txt.value = float(_theta1.value)
        _is_syncing["t1"] = False
        _refresh_all()

    def _on_t0_text(chg):
        if _is_syncing["t0"]: return
        _is_syncing["t0"] = True
        try:
            _theta0.value = float(_theta0_txt.value)
        finally:
            _is_syncing["t0"] = False

    def _on_t1_text(chg):
        if _is_syncing["t1"]: return
        _is_syncing["t1"] = True
        try:
            _theta1.value = float(_theta1_txt.value)
        finally:
            _is_syncing["t1"] = False

    _theta0.observe(_on_t0_slider, names="value")
    _theta1.observe(_on_t1_slider, names="value")
    _theta0_txt.observe(_on_t0_text, names="value")
    _theta1_txt.observe(_on_t1_text, names="value")
    _alpha.observe(_refresh_all,  names="value")
    _iters.observe(_refresh_all,  names="value")

    # --- Layout (tri grafa) ---
    _controls_row1 = HBox([_theta0, _theta1, _alpha, _iters],
                          layout=Layout(align_items="center"))
    _controls_row2 = HBox([_theta0_txt, _theta1_txt, _mse_label, _final_info],
                          layout=Layout(align_items="center"))
    _plots_row     = HBox([_out_left, _out_middle, _out_right],
                          layout=Layout(align_items="center"))
    _ui = VBox([_controls_row1, _controls_row2, _plots_row])

    display(_ui)
    _refresh_all()
