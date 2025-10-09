import numpy as np
# Imports plotting library
import matplotlib.pyplot as plt


# Linearna funkcija s jednom varijablom, x
def linear_function_1D(x, beta, omega):
    # TODO -- implementirajte 1D linearnu funkciju
    y = x*omega + beta
    return y

# Linearna funkcija s dvije varijable, x1 and x2
def linear_function_2D(x1, x2, beta, omega1, omega2):
    # TODO -- implementirajte 2D linearnu funkciju 
    y = beta + x1*omega1 + x2*omega2
    return y

# Linearna funkcija s tri varijable, x1, x2, and x_3
def linear_function_3D(x1, x2, x3, beta, omega1, omega2, omega3):
    # TODO -- rimplementirajte 3D linearnu funkciju
    y = x1
    return y

# Plot the 1D linear function
def plot_1D(x, y):
    fig, ax = plt.subplots()
    ax.plot(x,y,'r-')
    ax.set_ylim([0,10])
    ax.set_xlim([0,10])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title("1D plot")
    plt.show()

# Code to draw 2D function -- read it so you know what is going on, but you don't have to change it
def plot_2D(x1_mesh, x2_mesh, y):
    fig, ax = plt.subplots()
    fig.set_size_inches(7,7)
    pos = ax.contourf(x1_mesh, x2_mesh, y, levels=256 ,cmap = 'hot', vmin=-50,vmax=50.0)
    fig.colorbar(pos, ax=ax)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    levels = np.arange(-50,50,1.0)
    ax.contour(x1_mesh, x2_mesh, y, levels, cmap='winter')
    ax.set_title("2D plot")
    plt.show()
