"""
Jednostavni widgeti za treniranje MLP i Conv2D CNN modela na MNIST-u.

Zajedničke značajke:
  - biraju se: learning rate, broj epoha, batch size, augmentacija (DA/NE)
  - dodatno:
      * MLP: broj neurona u skrivenom sloju (hidden_dim) i broj slojeva
      * Conv2D: broj Conv slojeva, broj filtera i veličina jezgre
  - aktivacija je uvijek ReLU
  - inicijalizacija je Kaiming He
  - grafički se prikazuje score (macro F1) na validacijskom skupu + test score
  - na početku svakog treninga ispisuje se model summary (print_model_summary)
"""

import os
import sys
import time
from typing import Dict, List

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

import ipywidgets as widgets
from ipywidgets import HBox, VBox, Layout, Output
from IPython.display import display  # nije nužno, ali korisno u notebooku

# Dodaj projektni direktorij u sys.path (ako već nije)
current_file_path = os.path.abspath(__file__)
main_dir = os.path.dirname(current_file_path)
if main_dir not in sys.path:
    sys.path.append(main_dir)

from dataloader import mnist_dataset, ordinary_dataloader
from models import DeepMLPNet, SimpleConv2DNet, print_model_summary


# ======================================================================
# Pomoćne funkcije: macro F1 i treniranje
# ======================================================================

def _compute_macro_f1(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """
    Računa macro F1 mjeru za višeklasnu klasifikaciju.

    y_true: (N,)
    y_pred: (N,)
    """
    y_true = y_true.view(-1)
    y_pred = y_pred.view(-1)

    if y_true.numel() == 0:
        return 0.0

    num_classes = int(max(int(y_true.max()), int(y_pred.max()))) + 1
    f1_per_class: List[float] = []

    for c in range(num_classes):
        tp = ((y_pred == c) & (y_true == c)).sum().item()
        fp = ((y_pred == c) & (y_true != c)).sum().item()
        fn = ((y_pred != c) & (y_true == c)).sum().item()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        if (precision + recall) > 0:
            f1 = 2.0 * precision * recall / (precision + recall)
        else:
            f1 = 0.0

        f1_per_class.append(f1)

    if len(f1_per_class) == 0:
        return 0.0

    macro_f1 = float(sum(f1_per_class) / len(f1_per_class))
    return macro_f1


def _evaluate_macro_f1(
    model: nn.Module,
    data_loader,
    device: torch.device,
) -> float:
    """
    Pomoćna funkcija za evaluaciju macro F1 na danom loaderu.
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            pred = torch.argmax(logits, dim=1)

            all_preds.append(pred.cpu())
            all_targets.append(labels.cpu())

    if len(all_preds) == 0:
        return float("nan")

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    macro_f1 = _compute_macro_f1(all_targets, all_preds)
    return float(macro_f1)


def _train_with_macro_f1(
    model: nn.Module,
    train_loader,
    valid_loader,
    test_loader,
    num_epochs: int,
    learning_rate: float,
    device: torch.device,
) -> Dict[str, List[float]]:
    """
    Jednostavna petlja treniranja:
      - optimizator: Adam
      - loss: CrossEntropy
      - prati se train loss
      - računa se macro F1 na valid skupu po epohama
      - nakon treninga računa se macro F1 na test skupu (samo zadnji model)

    Vraća:
      history = {
        "train_loss": [...],
        "valid_macro_f1": [...],
        "test_macro_f1_last": [float],
        "epoch_time_sec": [...]
      }
    """

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    train_loss_hist: List[float] = []
    valid_f1_hist: List[float] = []
    epoch_time_hist: List[float] = []

    for epoch in range(1, num_epochs + 1):
        start_time = time.time()

        model.train()
        running_loss = 0.0
        num_batches = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            num_batches += 1

        avg_loss = running_loss / max(1, num_batches)
        train_loss_hist.append(avg_loss)

        # Validacija (macro F1)
        valid_macro_f1 = _evaluate_macro_f1(model, valid_loader, device)
        valid_f1_hist.append(valid_macro_f1)

        epoch_time = time.time() - start_time
        epoch_time_hist.append(epoch_time)

        print(
            f"[Epoh {epoch:02d}] "
            f"Train loss: {avg_loss:.4f} | "
            f"Valid macro F1: {valid_macro_f1:.4f} | "
            f"Vrijeme: {epoch_time:.2f} s"
        )

    # Test macro F1 za zadnji model
    test_macro_f1_last = _evaluate_macro_f1(model, test_loader, device)

    history: Dict[str, List[float]] = {
        "train_loss": train_loss_hist,
        "valid_macro_f1": valid_f1_hist,
        "test_macro_f1_last": [test_macro_f1_last],
        "epoch_time_sec": epoch_time_hist,
    }

    # Plot: valid macro F1 po epohama + test F1 (zvjezdica)
    epochs_axis = range(1, num_epochs + 1)
    plt.figure(figsize=(6, 4))
    plt.plot(
        epochs_axis,
        valid_f1_hist,
        marker="o",
        linewidth=2,
        label="Valid macro F1",
    )

    plt.scatter(
        [num_epochs],
        [test_macro_f1_last],
        marker="*",
        s=160,
        color="red",
        edgecolors="black",
        linewidths=0.8,
        label=f"Test macro F1 = {test_macro_f1_last:.3f}",
    )

    plt.xlabel("Epoh")
    plt.ylabel("Macro F1")
    plt.title("Valid vs Test macro F1")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    print(f"\nZadnji test macro F1: {test_macro_f1_last:.4f}")
    print("Vrijeme po epohama (s):", [f"{t:.2f}" for t in epoch_time_hist])

    return history


# ======================================================================
# MLP widget (simple, s izborom neurona i slojeva)
# ======================================================================

def mlp_simple_widget():
    """
    Jednostavan MLP widget:

      - NEMA regularizacije u sučelju
      - korisnik bira:
          * learning rate
          * broj epoha
          * batch size
          * augmentacija (DA/NE)
          * broj neurona u skrivenim slojevima (hidden_dim)
          * broj skrivenih slojeva (num_layers)
      - model: DeepMLPNet
        * aktivacija: ReLU
        * inicijalizacija: Kaiming He
      - grafički prikazuje:
        * macro F1 na validacijskom skupu po epohama
        * test macro F1 kao zvjezdicu za zadnji model
      - na početku treninga ispisuje se model summary (print_model_summary)
    """

    output = Output(layout=Layout(border="1px solid #ccc"))

    # ---- Kontrole ----
    lr_slider = widgets.FloatLogSlider(
        description="LR",
        value=1e-3,
        base=10,
        min=-5,
        max=-1,
        step=0.25,
        continuous_update=False,
        layout=Layout(width="260px"),
        style={"description_width": "initial"},
    )

    epochs_slider = widgets.IntSlider(
        description="# epoha",
        value=5,
        min=1,
        max=50,
        step=1,
        continuous_update=False,
        layout=Layout(width="260px"),
        style={"description_width": "initial"},
    )

    batch_slider = widgets.IntSlider(
        description="Batch size",
        value=32,
        min=8,
        max=256,
        step=8,
        continuous_update=False,
        layout=Layout(width="260px"),
        style={"description_width": "initial"},
    )

    hidden_dim_slider = widgets.IntSlider(
        description="Hidden dim",
        value=256,
        min=32,
        max=1024,
        step=32,
        continuous_update=False,
        layout=Layout(width="260px"),
        style={"description_width": "initial"},
    )

    num_layers_slider = widgets.IntSlider(
        description="# slojeva",
        value=4,
        min=1,
        max=20,
        step=1,
        continuous_update=False,
        layout=Layout(width="260px"),
        style={"description_width": "initial"},
    )

    augment_checkbox = widgets.Checkbox(
        description="Augmentacija (train)",
        value=False,
        layout=Layout(width="260px"),
    )

    train_button = widgets.Button(
        description="Treniraj MLP",
        button_style="success",
        icon="play",
        layout=Layout(width="220px", height="40px"),
    )

    # ---- Callback ----
    def _on_train_clicked(_b):
        with output:
            output.clear_output(wait=True)

            batch_size = int(batch_slider.value)
            lr = float(lr_slider.value)
            num_epochs = int(epochs_slider.value)
            hidden_dim = int(hidden_dim_slider.value)
            num_layers = int(num_layers_slider.value)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Uređaj: {device}")

            print("Priprema MNIST train/valid/test skupa...")
            train_ds = mnist_dataset(
                subset="train",
                use_augmentation=augment_checkbox.value,
                verbose=False,
            )
            valid_ds = mnist_dataset(
                subset="valid",
                use_augmentation=False,
                verbose=False,
            )
            test_ds = mnist_dataset(
                subset="test",
                use_augmentation=False,
                verbose=False,
            )

            train_loader = ordinary_dataloader(train_ds, batch_size=batch_size)
            valid_loader = ordinary_dataloader(valid_ds, batch_size=batch_size)
            test_loader = ordinary_dataloader(test_ds, batch_size=batch_size)

            # MLP model – ReLU + Kaiming He
            model = DeepMLPNet(
                input_dim=28 * 28,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                num_classes=10,
                init_scale=1.0,
                activation="relu",
                init_type="kaiming_uniform",
                init_constant_value=0.01,
                dropout_prob=0.0,
            )

            print("\n=== Model summary (MLP) ===")
            summary_obj = print_model_summary(
                model=model.to(device),
                device=str(device),
                input_dim=(1, 28 * 28),
            )
            # eksplicitno ispisati summary objekt
            print(summary_obj)

            print("\n=== Pokrećem treniranje MLP-a ===")
            _ = _train_with_macro_f1(
                model=model,
                train_loader=train_loader,
                valid_loader=valid_loader,
                test_loader=test_loader,
                num_epochs=num_epochs,
                learning_rate=lr,
                device=device,
            )

            print("\nTrening MLP-a završen.")

    train_button.on_click(_on_train_clicked)

    # Layout
    row1 = HBox([lr_slider, epochs_slider, batch_slider],
                layout=Layout(gap="10px"))
    row2 = HBox([hidden_dim_slider, num_layers_slider, augment_checkbox],
                layout=Layout(gap="10px"))
    row3 = HBox([train_button],
                layout=Layout(gap="10px"))

    ui = VBox([row1, row2, row3, output], layout=Layout(gap="12px"))
    return ui


# ======================================================================
# Conv2D CNN widget (simple, s izborom slojeva/filtera/jezgre)
# ======================================================================

def conv2d_simple_widget():
    """
    Jednostavan Conv2D CNN widget:

      - korisnik bira:
          * learning rate
          * broj epoha
          * batch size
          * augmentacija (DA/NE)
          * broj Conv slojeva (num_layers)
          * broj filtera po sloju (filters_per_layer)
          * veličinu jezgre (kernel_size)
      - model: SimpleConv2DNet
        * Conv2d slojevi s ReLU aktivacijom
        * Kaiming He inicijalizacija
      - grafički prikazuje:
        * macro F1 na validacijskom skupu po epohama
        * test macro F1 kao zvjezdicu za zadnji model
      - na početku treninga ispisuje se model summary (print_model_summary)
    """

    output = Output(layout=Layout(border="1px solid #ccc"))

    # Kontrole
    lr_slider = widgets.FloatLogSlider(
        description="LR",
        value=1e-3,
        base=10,
        min=-5,
        max=-1,
        step=0.25,
        continuous_update=False,
        layout=Layout(width="260px"),
        style={"description_width": "initial"},
    )

    epochs_slider = widgets.IntSlider(
        description="# epoha",
        value=5,
        min=1,
        max=50,
        step=1,
        continuous_update=False,
        layout=Layout(width="260px"),
        style={"description_width": "initial"},
    )

    batch_slider = widgets.IntSlider(
        description="Batch size",
        value=32,
        min=8,
        max=256,
        step=8,
        continuous_update=False,
        layout=Layout(width="260px"),
        style={"description_width": "initial"},
    )

    num_layers_slider = widgets.IntSlider(
        description="# Conv slojeva",
        value=3,
        min=1,
        max=6,
        step=1,
        continuous_update=False,
        layout=Layout(width="260px"),
        style={"description_width": "initial"},
    )

    filters_slider = widgets.IntSlider(
        description="# filtera / sloj",
        value=16,
        min=4,
        max=64,
        step=4,
        continuous_update=False,
        layout=Layout(width="260px"),
        style={"description_width": "initial"},
    )

    kernel_dropdown = widgets.Dropdown(
        description="Kernel size",
        options=[("3×3", 3), ("5×5", 5), ("7×7", 7)],
        value=3,
        layout=Layout(width="260px"),
        style={"description_width": "initial"},
    )

    augment_checkbox = widgets.Checkbox(
        description="Augmentacija (train)",
        value=False,
        layout=Layout(width="260px"),
    )

    train_button = widgets.Button(
        description="Treniraj Conv2D",
        button_style="success",
        icon="play",
        layout=Layout(width="220px", height="40px"),
    )

    def _on_train_clicked(_b):
        with output:
            output.clear_output(wait=True)

            batch_size = int(batch_slider.value)
            lr = float(lr_slider.value)
            num_epochs = int(epochs_slider.value)
            num_layers = int(num_layers_slider.value)
            filters = int(filters_slider.value)
            kernel_size = int(kernel_dropdown.value)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Uređaj: {device}")

            print("Priprema MNIST train/valid/test skupa...")
            train_ds = mnist_dataset(
                subset="train",
                use_augmentation=augment_checkbox.value,
                verbose=False,
            )
            valid_ds = mnist_dataset(
                subset="valid",
                use_augmentation=False,
                verbose=False,
            )
            test_ds = mnist_dataset(
                subset="test",
                use_augmentation=False,
                verbose=False,
            )

            train_loader = ordinary_dataloader(train_ds, batch_size=batch_size)
            valid_loader = ordinary_dataloader(valid_ds, batch_size=batch_size)
            test_loader = ordinary_dataloader(test_ds, batch_size=batch_size)

            # CNN model – SimpleConv2DNet s odabranim parametrima
            model = SimpleConv2DNet(
                in_channels=1,
                num_classes=10,
                num_layers=num_layers,
                filters_per_layer=filters,
                kernel_size=kernel_size,
                input_height=28,
                input_width=28,
            )

            print("\n=== Model summary (Conv2D CNN) ===")
            summary_obj = print_model_summary(
                model=model.to(device),
                device=str(device),
                input_dim=(1, 1, 28, 28),
            )
            print(summary_obj)

            print("\n=== Pokrećem treniranje Conv2D mreže ===")
            _ = _train_with_macro_f1(
                model=model,
                train_loader=train_loader,
                valid_loader=valid_loader,
                test_loader=test_loader,
                num_epochs=num_epochs,
                learning_rate=lr,
                device=device,
            )

            print("\nTrening Conv2D mreže završen.")

    train_button.on_click(_on_train_clicked)

    row1 = HBox([lr_slider, epochs_slider, batch_slider],
                layout=Layout(gap="10px"))
    row2 = HBox([num_layers_slider, filters_slider, kernel_dropdown],
                layout=Layout(gap="10px"))
    row3 = HBox([augment_checkbox, train_button],
                layout=Layout(gap="10px"))

    ui = VBox([row1, row2, row3, output], layout=Layout(gap="12px"))
    return ui


# ======================================================================
# Tab koji spaja oba widgeta
# ======================================================================

def unified_mlp_conv2d_widget():
    """
    Tab sučelje:
      - jedan tab za MLP
      - drugi tab za Conv2D CNN
    """
    mlp_ui = mlp_simple_widget()
    cnn_ui = conv2d_simple_widget()

    tab = widgets.Tab(children=[mlp_ui, cnn_ui])
    tab.set_title(0, "MLP (fully connected)")
    tab.set_title(1, "Conv2D CNN")
    return tab
