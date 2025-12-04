"""
Widget wrapper za interaktivno biranje hiperparametara i treniranje DeepMLPNet-a.
"""

from typing import Optional

import torch
import ipywidgets as widgets
from ipywidgets import HBox, VBox, Layout, Output
import matplotlib.pyplot as plt
import os, sys

# Add project files
current_file_path = os.path.abspath(__file__)
main_dir = os.path.dirname(current_file_path)
sys.path.append(main_dir)

from dataloader import mnist_dataset, ordinary_dataloader
from models import DeepMLPNet
from training import DeepMLPTrainer


def deep_mlp_training_widget():
    """
    Widget za treniranje DeepMLPNet mreže na MNIST-u s:
      - L1/L2 regularizacijom
      - dropoutom
      - opcionalnom augmentacijom train skupa
      - box-plot aktivacija po slojevima
      - box-plot gradijenata po slojevima
      - valid macro F1 krivuljom + test macro F1 točkom
    """

    _trainer_holder = {"trainer": None}

    # ---------- Kontrole ----------
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

    _activation_label = widgets.HTML(
        value="<b>Aktivacija:</b> ReLU",
        layout=Layout(width="180px"),
    )
    _init_type_label = widgets.HTML(
        value="<b>Init:</b> Kaiming (He, uniform)",
        layout=Layout(width="220px"),
    )
    _init_scale_label = widgets.HTML(
        value="<b>Init scale:</b> 1.0",
        layout=Layout(width="150px"),
    )

    _learning_rate = widgets.FloatLogSlider(
        description="LR",
        value=1e-3,
        base=10,
        min=-5,
        max=-1,
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

    _dropout = widgets.FloatSlider(
        description="Dropout",
        value=0.0,
        min=0.0,
        max=0.8,
        step=0.05,
        readout_format=".2f",
        continuous_update=False,
        layout=Layout(width="260px"),
    )

    _augment = widgets.Checkbox(
        description="Augment train",
        value=False,
        layout=Layout(width="300px"),
    )

    _l1_lambda = widgets.FloatText(
        description="L1 λ",
        value=0.0,
        layout=Layout(width="180px"),
    )

    _l2_lambda = widgets.FloatText(
        description="L2 λ",
        value=0.0,
        layout=Layout(width="180px"),
    )

    _train_button = widgets.Button(
        description="Pokreni treniranje",
        button_style="success",
        icon="play",
        layout=Layout(width="220px", height="40px"),
    )

    _layer_dropdown = widgets.Dropdown(
        description="Sloj (akt)",
        options=[],
        layout=Layout(width="400px"),
    )
    _plot_layer_button = widgets.Button(
        description="Plot aktivacije sloja",
        button_style="info",
        icon="bar-chart",
        layout=Layout(width="220px", height="40px"),
    )

    _grad_layer_dropdown = widgets.Dropdown(
        description="Sloj (grad)",
        options=[],
        layout=Layout(width="400px"),
    )
    _plot_grad_button = widgets.Button(
        description="Plot gradijente sloja",
        button_style="warning",
        icon="area-chart",
        layout=Layout(width="220px", height="40px"),
    )

    _output = Output(layout=Layout(border="1px solid #ccc"))

    # ---------- Test macro F1 helper ----------
    def _evaluate_test_macro_f1(_model: torch.nn.Module,
                                _device: torch.device,
                                batch_size: int = 256) -> float:
        _test_dataset = mnist_dataset(subset="test")
        _test_loader = ordinary_dataloader(
            dataset=_test_dataset,
            batch_size=batch_size,
        )

        _model.eval()
        _all_preds = []
        _all_targets = []

        with torch.no_grad():
            for _images, _labels in _test_loader:
                _images = _images.to(_device)
                _labels = _labels.to(_device)

                _logits = _model(_images)
                _pred = torch.argmax(_logits, dim=1)

                _all_preds.append(_pred.cpu())
                _all_targets.append(_labels.cpu())

        if len(_all_preds) == 0:
            return float("nan")

        _all_preds = torch.cat(_all_preds, dim=0)
        _all_targets = torch.cat(_all_targets, dim=0)

        _macro_f1 = DeepMLPTrainer._compute_macro_f1(_all_targets, _all_preds)
        return float(_macro_f1)

    # ---------- Callback: treniranje ----------
    def _on_train_clicked(_b):
        with _output:
            _output.clear_output(wait=True)

            _train_dataset = mnist_dataset(
                subset="train",
                valid_ratio=_valid_ratio.value,
                use_augmentation=_augment.value,  # <--- AUGMENT
            )
            _valid_dataset = mnist_dataset(
                subset="valid",
                valid_ratio=_valid_ratio.value,
                use_augmentation=False,           # valid bez augmentacije
            )

            _train_loader = ordinary_dataloader(
                dataset=_train_dataset,
                batch_size=_train_batch_size.value,
            )
            _eval_bs = max(_train_batch_size.value, 256)
            _valid_loader = ordinary_dataloader(
                dataset=_valid_dataset,
                batch_size=_eval_bs,
            )

            _model = DeepMLPNet(
                input_dim=28 * 28,
                hidden_dim=_hidden_dim.value,
                num_layers=_num_layers.value,
                num_classes=10,
                init_scale=1.0,
                activation="relu",
                init_type="kaiming_uniform",
                init_constant_value=0.01,
                dropout_prob=float(_dropout.value),
            )

            _trainer = DeepMLPTrainer(
                model=_model,
                train_loader=_train_loader,
                valid_loader=_valid_loader,
                num_epochs=_num_epochs.value,
                learning_rate=float(_learning_rate.value),
                device=None,
                max_train_batches=_max_train_batches.value,
                track_gradients=True,
                l1_lambda=float(_l1_lambda.value),
                l2_lambda=float(_l2_lambda.value),
                verbose=True,
            )

            _trainer.train(plot_gradients=True)
            _trainer_holder["trainer"] = _trainer

            _layer_names = sorted(_trainer.activation_history.keys())
            _layer_dropdown.options = _layer_names
            if _layer_names:
                _layer_dropdown.value = _layer_names[0]

            _grad_layer_names = sorted(_trainer.grad_distribution_history.keys())
            _grad_layer_dropdown.options = _grad_layer_names
            if _grad_layer_names:
                _grad_layer_dropdown.value = _grad_layer_names[0]

            has_valid = len(_trainer.valid_f1_history) > 0
            if has_valid:
                _epochs_axis = range(1, len(_trainer.valid_f1_history) + 1)
                _test_macro_f1 = _evaluate_test_macro_f1(
                    _model,
                    _trainer.device,
                    batch_size=_eval_bs,
                )

                plt.figure(figsize=(6, 4))
                plt.plot(
                    _epochs_axis,
                    _trainer.valid_f1_history,
                    marker="o",
                    linewidth=2,
                    label="Valid macro F1",
                )

                if _test_macro_f1 is not None:
                    _last_epoch = len(_trainer.valid_f1_history)
                    plt.scatter(
                        [_last_epoch],
                        [_test_macro_f1],
                        marker="*",
                        s=160,
                        color="red",
                        edgecolors="black",
                        linewidths=0.8,
                        label=f"Test macro F1 = {_test_macro_f1:.3f}",
                    )

                plt.xlabel("Epoh")
                plt.ylabel("Macro F1")
                plt.title("Macro F1: valid krivulja + test točka")
                plt.grid(True, linestyle="--", linewidth=0.4)
                plt.ylim(0.0, 1.0)
                plt.legend()
                plt.tight_layout()
                plt.show()

    _train_button.on_click(_on_train_clicked)

    # ---------- Callback: box-plot aktivacija ----------
    def _on_plot_layer_clicked(_b):
        with _output:
            _trainer = _trainer_holder.get("trainer", None)
            if _trainer is None:
                print("Nema treniranog modela. Najprije pokreni treniranje.")
                return

            _layer_name = _layer_dropdown.value
            if not _layer_name:
                print("Nije odabran sloj za crtanje aktivacija.")
                return

            if _layer_name not in _trainer.activation_history:
                print(f"Nema spremljenih aktivacija za sloj: {_layer_name}")
                return

            _epoch_activations = _trainer.activation_history[_layer_name]
            if len(_epoch_activations) == 0:
                print("Nema aktivacija za crtanje.")
                return

            _box_data = [t.numpy() for t in _epoch_activations]
            _epochs = list(range(1, len(_box_data) + 1))

            plt.figure(figsize=(8, 5))
            plt.boxplot(
                _box_data,
                positions=_epochs,
                showfliers=False,
            )
            plt.xlabel("Epoh")
            plt.ylabel("Aktivacije")
            plt.title(f"Distribucija aktivacija po epohama\nSloj: {_layer_name}")
            plt.grid(True, axis="y", linestyle="--", linewidth=0.4)
            plt.tight_layout()
            plt.show()

    _plot_layer_button.on_click(_on_plot_layer_clicked)

    # ---------- Callback: box-plot gradijenata ----------
    def _on_plot_grad_clicked(_b):
        with _output:
            _trainer = _trainer_holder.get("trainer", None)
            if _trainer is None:
                print("Nema treniranog modela. Najprije pokreni treniranje.")
                return

            _layer_name = _grad_layer_dropdown.value
            if not _layer_name:
                print("Nije odabran sloj za crtanje gradijenata.")
                return

            if _layer_name not in _trainer.grad_distribution_history:
                print(f"Nema spremljenih gradijenata za sloj: {_layer_name}")
                return

            _epoch_grads = _trainer.grad_distribution_history[_layer_name]
            if len(_epoch_grads) == 0:
                print("Nema gradijenata za crtanje.")
                return

            _box_data = [t.numpy() for t in _epoch_grads]
            _epochs = list(range(1, len(_box_data) + 1))

            plt.figure(figsize=(8, 5))
            plt.boxplot(
                _box_data,
                positions=_epochs,
                showfliers=False,
            )
            plt.xlabel("Epoh")
            plt.ylabel("Gradijenti")
            plt.title(f"Distribucija gradijenata po epohama\nSloj: {_layer_name}")
            plt.grid(True, axis="y", linestyle="--", linewidth=0.4)
            plt.tight_layout()
            plt.show()

    _plot_grad_button.on_click(_on_plot_grad_clicked)

    # ---------- Layout ----------
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
            _activation_label,
            _init_type_label,
            _init_scale_label,
        ],
        layout=Layout(justify_content="flex-start", gap="10px"),
    )

    _row3 = HBox(
        [
            _learning_rate,
            _train_batch_size,
            _valid_ratio,
            _dropout,
        ],
        layout=Layout(justify_content="flex-start", gap="10px"),
    )

    _row4 = HBox(
        [
            _max_train_batches,
            _l1_lambda,
            _l2_lambda,
            _augment,   # checkbox za augmentaciju
        ],
        layout=Layout(justify_content="flex-start", gap="10px"),
    )

    _row5 = HBox(
        [
            _train_button,
        ],
        layout=Layout(justify_content="flex-start", gap="10px"),
    )

    _row6 = HBox(
        [
            _layer_dropdown,
            _plot_layer_button,
        ],
        layout=Layout(justify_content="flex-start", gap="10px"),
    )

    _row7 = HBox(
        [
            _grad_layer_dropdown,
            _plot_grad_button,
        ],
        layout=Layout(justify_content="flex-start", gap="10px"),
    )

    _ui = VBox(
        [
            _row1,
            _row2,
            _row3,
            _row4,
            _row5,
            _row6,
            _row7,
            _output,
        ],
        layout=Layout(gap="12px"),
    )

    return _ui
