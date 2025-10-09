# Učitavanje knjižnica
from __future__ import print_function
import ipywidgets as widgets
from IPython.display import display, clear_output
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # only if you need later
# %matplotlib widget  # run this once per notebook (keep it in a cell above)

# Global container (list of [x, y])
point_list = []

def generator():
    # --- Widgets ---
    x = widgets.BoundedFloatText(value=0, min=-10, max=10, step=0.5, description='X:')
    y = widgets.BoundedFloatText(value=0, min=-10, max=10, step=0.5, description='Y:')
    btn_add = widgets.Button(description="Add Point")
    btn_print = widgets.Button(description="Print points")
    btn_calc = widgets.Button(description="Calculate")
    out_text = widgets.Output()
    out_plot = widgets.Output()

    # --- Helpers ---
    def redraw_scatter():
        with out_plot:
            out_plot.clear_output(wait=True)
            fig, ax = plt.subplots(figsize=(5, 5))
            if point_list:
                x_cor = [p[0] for p in point_list]
                y_cor = [p[1] for p in point_list]
                ax.scatter(x_cor, y_cor, alpha=0.9, marker='x')
            ax.grid(True)
            ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_title('Graf')
            ax.set_xlim(-10, 10); ax.set_ylim(-10, 10)
            plt.show()

    # --- Callbacks (capture widgets via closure) ---
    def on_add(_b):
        point_list.append([x.value, y.value])
        with out_text:
            out_text.clear_output()
            print(f"Added point: ({x.value}, {y.value})")
        redraw_scatter()

    def on_print(_b):
        with out_text:
            out_text.clear_output()
            if not point_list:
                print("No points yet.")
            else:
                for px, py in point_list:
                    print(f"Point: ({px}, {py})")

    def on_calc(_b):
        if len(point_list) < 2:
            with out_text:
                out_text.clear_output()
                print("Need at least 2 points to fit a line.")
            redraw_scatter()
            return

        x_data = np.asarray([p[0] for p in point_list], dtype=float)
        y_data = np.asarray([p[1] for p in point_list], dtype=float)

        # Linear fit: y = m*x + b (np.polyfit returns [m, b])
        m, b = np.polyfit(x_data, y_data, 1)
        f = lambda q: m * q + b

        with out_text:
            out_text.clear_output()
            sum_x  = float(np.sum(x_data))
            sum_y  = float(np.sum(y_data))
            sum_xx = float(np.sum(x_data * x_data))
            sum_xy = float(np.sum(x_data * y_data))
            print(f"Sum X:  {sum_x:.4f}")
            print(f"Sum Y:  {sum_y:.4f}")
            print(f"Sum X2: {sum_xx:.4f}")
            print(f"Sum XY: {sum_xy:.4f}")
            print(f"y = {m:.4f} x + {b:.4f}")

        with out_plot:
            out_plot.clear_output(wait=True)
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.scatter(x_data, y_data, alpha=0.9, marker='x')

            # Proper axis for line
            xAxis = np.linspace(-10, 10, 201)
            ax.plot(xAxis, f(xAxis))

            ax.grid(True)
            ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_title('Graf')
            ax.set_xlim(-10, 10); ax.set_ylim(-10, 10)
            plt.show()

    # --- Wire events ---
    btn_add.on_click(on_add)
    btn_print.on_click(on_print)
    btn_calc.on_click(on_calc)

    # --- Layout & display ---
    controls = widgets.VBox([x, y, widgets.HBox([btn_add, btn_print, btn_calc])])
    layout = widgets.VBox([widgets.HBox([controls, out_text]), out_plot])
    display(layout)


