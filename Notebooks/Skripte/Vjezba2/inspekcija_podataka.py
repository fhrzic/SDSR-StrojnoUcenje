# Učitavanje programskih knjižica
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, root_mean_squared_error

# Add project files
current_file_path = os.path.abspath(__file__)
main_dir =  os.path.dirname(current_file_path)
print(main_dir)
sys.path.append(main_dir)

def plot_data():
    """
    Function for plotting the data for ledo_sladoledi.csv
    """

    # Učitavanje podataka u dataframe 
    _data = pd.read_csv(os.path.join(main_dir, "ledoSladoledi.csv"), header=None)
    _x = _data[0]
    _y = _data[1]

    # Crtanje podataka
    plt.figure('Podaci')
    plt.plot(_x, _y, 'bx')
    plt.ylabel("Profit u 10,000$")
    plt.xlabel("Populacija u 10,000 stanovnika")
    plt.grid()
    plt.show()

def inspect_data():
    """
    Inspect data properties
    """

    # Učitavanje podataka u dataframe 
    _data = pd.read_csv(os.path.join(main_dir, "ledoSladoledi.csv"), header=None)
    _x = _data[0]
    _y = _data[1]

    #TODO inspect data

def draw_linear(theta0: float = None,
                theta1: float = None):
    """
    Method which draws linear function based on the given parameters theta0 and theta1:
        y = theta0 + theta1*x

        Args:
        * theta0, int, bias
        * theta1, int, slope
    """

    # Učitavanje podataka u dataframe 
    _data = pd.read_csv(os.path.join(main_dir, "ledoSladoledi.csv"), header=None)
    _x = _data[0]
    _y = _data[1]

    # Izračun pravca
    _x_axis = np.linspace(np.min(_x),np.max(_x),len(_x))
    _f = lambda x, theta0, theta1: x * theta1 + theta0


    # Crtanje podataka i pravca
    plt.figure('Podaci')
    plt.plot(_x, _y, 'bx')
    plt.plot(_x_axis, _f(_x_axis, theta0, theta1),'g', label = "y = " + str(round(theta1, 2))+" * x + "+str(round(theta0,2)))

    plt.ylabel("Profit u 10,000$")
    plt.xlabel("Populacija u 10,000 stanovnika")
    plt.grid()
    plt.show()

def evaluate_metrics(theta0: float = None,
                theta1: float = None):
    """
    Method which draws linear function based on the given parameters theta0 and theta1:
        y = theta0 + theta1 * x

    Also computes regression evaluation metrics.

    Args:
    * theta0: float, bias (intercept)
    * theta1: float, slope
    """

    # Učitavanje podataka u dataframe 
    _data = pd.read_csv(os.path.join(main_dir, "ledoSladoledi.csv"), header=None)
    _x = _data[0]
    _y = _data[1]

    # Predikcija na osnovu modela
    _y_pred = theta0 + theta1 * _x

    # Izračunavanje metrika #TODO
    mse = 0
    rmse = 0
    mae = 0
    r2 = 0

    print(f"MSE:  {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"R²:   {r2:.4f}")

    # Izračun pravca za crtanje
    _x_axis = np.linspace(np.min(_x), np.max(_x), len(_x))
    _f = lambda x, theta0, theta1: x * theta1 + theta0

    # Crtanje podataka i pravca
    plt.figure('Podaci')
    plt.plot(_x, _y, 'bx', label='Podaci')
    plt.plot(_x_axis, _f(_x_axis, theta0, theta1), 'g',
             label=f"y = {round(theta1, 2)} * x + {round(theta0, 2)}")

    plt.ylabel("Profit u 10,000$")
    plt.xlabel("Populacija u 10,000 stanovnika")
    plt.legend()
    plt.grid()
    plt.show()


def grid_search():
    """
    Method which implements grid search
    """
    # Učitavanje podataka u dataframe 
    _data = pd.read_csv(os.path.join(main_dir, "ledoSladoledi.csv"), header=None)
    _x = _data[0]
    _y = _data[1]

    # Kreirajmo funkciju
    _f = lambda _x, _theta0, _theta1: _x * _theta1 + _theta0

    # Generirajmo thete #TODO HINT linspace i meshgrid
    _theta0 = 0
    _theta1 = 0

    # Izračunajmo za svaku kombinaciju parametara theta cijenu
    _score = [0]
    
    
    _score = np.asarray(_score)
    _xs, _ys = np.meshgrid(_theta0, _theta1)

    # Prikažimo grafički rezultat
    fig = plt.figure("Funkcija cijene", figsize=(7,7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(_xs, _ys, _score.reshape(_xs.shape), cmap='hot')
    ax.set_xlabel('Theta0')
    ax.set_ylabel('Theta1')
    ax.set_zlabel('Cijena')
    plt.show()