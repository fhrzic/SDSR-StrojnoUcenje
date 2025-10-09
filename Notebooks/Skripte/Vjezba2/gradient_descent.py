# Učitavanje potrebnih knjižica
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from ipywidgets import interact, widgets
from sklearn.metrics import mean_squared_error
import sys
import os

# Add project files
current_file_path = os.path.abspath(__file__)
main_dir =  os.path.dirname(current_file_path)
print(main_dir)
sys.path.append(main_dir)

# Set your CSV path (two columns: x,y; no header)
def plot_interactive(csv_path: str =  os.path.join(main_dir, "ledoSladoledi.csv")):
    """
    Interactive 3D MSE surface for y = theta1*x + theta0 with a gradient-descent path.
    Returns the ipywidgets widget created by `interact`.
    """

    # ---- Load data
    data = pd.read_csv(os.path.join(main_dir, "data.csv"), engine="python", sep=r"[,\t\s]+", header=None)
    x = data[0]
    y = data[1]

    # ---- Precompute mesh & surface (once)
    theta0_grid = np.linspace(-10, 10, 100)
    theta1_grid = np.linspace(-10, 10, 100)
    xs, ys = np.meshgrid(theta0_grid, theta1_grid)

    def linear(xvec, t0, t1):
        return t1 * xvec + t0

    score_vals = []
    for t0 in theta0_grid:
        for t1 in theta1_grid:
            yhat = linear(x, t0, t1)
            score_vals.append(mean_squared_error(y, yhat))
    Z = np.asarray(score_vals).reshape(xs.shape)

    norm = plt.Normalize(Z.min(), Z.max())
    colors = cm.viridis(norm(Z))

    # Design matrix for gradient descent: X = [1, x]
    X = np.column_stack([np.ones_like(x), x])
    n = len(x)

    # ---- Callback (inner function) uses precomputed closure vars
    def _funkcijaMSE(theta0=0.0, theta1=0.0, alpha=0.1, brojIteracija=10):
        theta = np.array([theta0, theta1], dtype=float)
        thetas, scores = [], []

        for _ in range(int(brojIteracija)):
            yhat = X @ theta
            scores.append(mean_squared_error(y, yhat))
            thetas.append(theta.copy())
            grad = (X.T @ (yhat - y)) / n
            theta = theta - alpha * grad

        # Final point
        yhat = X @ theta
        scores.append(mean_squared_error(y, yhat))
        thetas.append(theta.copy())

        t0_path = np.array([t[0] for t in thetas])
        t1_path = np.array([t[1] for t in thetas])
        scores_arr = np.array(scores)

        # --- Draw (ensure surface is rendered)
        plt.close('all')
        fig = plt.figure("Funkcija cijene", figsize=(7, 7))
        ax = fig.add_subplot(111, projection='3d')

        surf = ax.plot_surface(xs, ys, Z, rcount=20, ccount=20,
                               facecolors=colors, shade=False)
        surf.set_facecolor((0, 0, 0, 0))  # make colormap visible

        ax.plot(t1_path, t0_path, scores_arr, 'bx-', linewidth=2.0)

        ax.set_xlabel('Theta1')
        ax.set_ylabel('Theta0')
        ax.set_zlabel('Cijena (MSE)')
        ax.set_title('Funkcija cijene + putanja gradijentnog spusta')

        plt.show()  # <- critical to ensure rendering with %matplotlib widget

    # ---- Build and return the widget
    return interact(
        _funkcijaMSE,
        theta0=widgets.FloatSlider(min=-10, max=10, step=0.5, value=0.0, continuous_update=False),
        theta1=widgets.FloatSlider(min=-10, max=10, step=0.5, value=0.0, continuous_update=False),
        alpha=widgets.FloatSlider(min=0.0, max=0.2, step=0.005, value=0.1, continuous_update=False),
        brojIteracija=widgets.IntSlider(min=0, max=50, step=1, value=10, continuous_update=False),
    )