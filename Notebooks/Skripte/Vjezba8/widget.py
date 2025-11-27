"""
Widget wrapper za interaktivno biranje hiperparametara i treniranje DeepMLPNet-a.
"""

# Učitavanje knjižnica
from typing import Optional

import torch
import ipywidgets as widgets
from ipywidgets import HBox, VBox, Layout, Output
import os, sys

# Add project files
current_file_path = os.path.abspath(__file__)
main_dir =  os.path.dirname(current_file_path)
sys.path.append(main_dir)



from dataloader import mnist_dataset, ordinary_dataloader
from models import DeepMLPNet
from training import DeepMLPTrainer


def deep_mlp_training_widget():
    """
    Funkcija koja stvara interaktivni widget za:
      - odabir hiperparametara DeepMLPNet mreže
      - treniranje modela pomoću DeepMLPTrainer klase
      - vizualizaciju normi gradijenata (vanishing/exploding demo)

    Kontrole:
        * hidden_dim
        * num_layers
        * init_scale
        * activation
        * init_type
        * init_constant_value
        * learning_rate
        * train batch size
        * valid/train split ratio
        * num_epochs
    """

    # =========================
    # 1. Definicija kontrola
    # =========================
    _hidden_dim = widgets.IntSlider(
        description="Hidden dim",
        value=256,
        min=2,
        max=1024,
        step=16,
        continuous_update=False,
        layout=Layout(width="250px"),
    )

    _num_layers = widgets.IntSlider(
        description="# slojeva",
        value=10,
        min=1,
        max=50,
        step=1,
        continuous_update=False,
        layout=Layout(width="250px"),
    )

    _init_scale = widgets.FloatLogSlider(
        description="Init scale",
        value=0.01,
        base=10,
        min=-4,   # 1e-4
        max=1,    # 1e0
        step=0.25,
        continuous_update=False,
        layout=Layout(width="260px"),
    )

    _activation = widgets.Dropdown(
        description="Aktivacija",
        options=["sigmoid", "relu", "tanh"],
        value="sigmoid",
        layout=Layout(width="220px"),
    )

    _init_type = widgets.Dropdown(
        description="Init tip",
        options=[
            "xavier_uniform",
            "xavier_normal",
            "kaiming_uniform",
            "zeros",
            "constant",
        ],
        value="xavier_uniform",
        layout=Layout(width="260px"),
    )

    _init_constant_value = widgets.FloatText(
        description="Const val",
        value=0.01,
        layout=Layout(width="220px"),
    )

    _learning_rate = widgets.FloatLogSlider(
        description="LR",
        value=1e-3,
        base=10,
        min=-5,   # 1e-5
        max=-1,   # 1e-1
        step=0.25,
        continuous_update=False,
        layout=Layout(width="260px"),
    )

    _train_batch_size = widgets.IntSlider(
        description="Train BS",
        value=128,
        min=16,
        max=512,
        step=16,
        continuous_update=False,
        layout=Layout(width="250px"),
    )

    _valid_ratio = widgets.FloatSlider(
        description="Valid ratio",
        value=0.1,
        min=0.05,
        max=0.3,
        step=0.05,
        readout_format=".2f",
        continuous_update=False,
        layout=Layout(width="260px"),
    )

    _num_epochs = widgets.IntSlider(
        description="# epoha",
        value=8,
        min=1,
        max=30,
        step=1,
        continuous_update=False,
        layout=Layout(width="250px"),
    )

    _max_train_batches = widgets.IntSlider(
        description="Max batch/ep",
        value=150,
        min=10,
        max=500,
        step=10,
        continuous_update=False,
        layout=Layout(width="260px"),
    )

    _train_button = widgets.Button(
        description="Pokreni treniranje",
        button_style="success",
        icon="play",
        layout=Layout(width="220px", height="40px"),
    )

    _output = Output(layout=Layout(border="1px solid #ccc"))

    # =========================
    # 2. Callback za gumb
    # =========================
    def _on_train_clicked(_b):
        with _output:
            _output.clear_output(wait=True)

            print("==== Postavke treniranja ====")
            print(f"hidden_dim          = {_hidden_dim.value}")
            print(f"num_layers          = {_num_layers.value}")
            print(f"init_scale          = {_init_scale.value:.5f}")
            print(f"activation          = {_activation.value}")
            print(f"init_type           = {_init_type.value}")
            print(f"init_constant_value = {_init_constant_value.value}")
            print(f"learning_rate       = {_learning_rate.value:.5f}")
            print(f"train batch size    = {_train_batch_size.value}")
            print(f"valid/train split   = {_valid_ratio.value:.2f}")
            print(f"num_epochs          = {_num_epochs.value}")
            print(f"max_train_batches   = {_max_train_batches.value}")
            print("==============================\n")

            # 1) Dataset-i (train/valid) s istim valid_ratio
            _train_dataset = mnist_dataset(
                subset="train",
                valid_ratio=_valid_ratio.value,
            )
            _valid_dataset = mnist_dataset(
                subset="valid",
                valid_ratio=_valid_ratio.value,
            )

            # 2) Dataloader-i
            _train_loader = ordinary_dataloader(
                dataset=_train_dataset,
                batch_size=_train_batch_size.value,
            )
            # valid BS može biti veći (brže prolazi)
            _valid_loader = ordinary_dataloader(
                dataset=_valid_dataset,
                batch_size=max(_train_batch_size.value, 256),
            )

            # 3) Model
            _model = DeepMLPNet(
                input_dim=28 * 28,
                hidden_dim=_hidden_dim.value,
                num_layers=_num_layers.value,
                num_classes=10,
                init_scale=float(_init_scale.value),
                activation=_activation.value,
                init_type=_init_type.value,
                init_constant_value=float(_init_constant_value.value),
            )

            # 4) Trainer
            _trainer = DeepMLPTrainer(
                model=_model,
                train_loader=_train_loader,
                valid_loader=_valid_loader,
                num_epochs=_num_epochs.value,
                learning_rate=float(_learning_rate.value),
                device=None,  # automatski bira 'cuda' ako je dostupno
                max_train_batches=_max_train_batches.value,
                track_gradients=True,
            )

            # 5) Treniranje + crtanje gradijenata
            _history = _trainer.train(plot_gradients=True)

            print("\nGotovo treniranje.")
            print("Train loss po epohi:", _trainer.train_loss_history)
            if len(_trainer.valid_loss_history) > 0:
                print("Valid loss po epohi:", _trainer.valid_loss_history)
            if len(_trainer.valid_f1_history) > 0:
                print("Valid macro F1 po epohi:", _trainer.valid_f1_history)

    _train_button.on_click(_on_train_clicked)

    # =========================
    # 3. Layout widgeta
    # =========================
    _row1 = HBox(
        [
            _hidden_dim,
            _num_layers,
            _num_epochs,
        ],
        layout=Layout(justify_content="flex-start", gap="10px"),
    )

    _row2 = HBox(
        [
            _init_scale,
            _activation,
            _init_type,
        ],
        layout=Layout(justify_content="flex-start", gap="10px"),
    )

    _row3 = HBox(
        [
            _init_constant_value,
            _learning_rate,
            _train_batch_size,
            _valid_ratio,
        ],
        layout=Layout(justify_content="flex-start", gap="10px"),
    )

    _row4 = HBox(
        [
            _max_train_batches,
            _train_button,
        ],
        layout=Layout(justify_content="flex-start", gap="10px"),
    )

    _ui = VBox(
        [
            _row1,
            _row2,
            _row3,
            _row4,
            _output,
        ],
        layout=Layout(gap="12px"),
    )

    return _ui