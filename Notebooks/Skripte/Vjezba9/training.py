"""
Klasa za treniranje DeepMLPNet mreže na MNIST skupu podataka.

Korištene komponente:
    * DeepMLPNet  -- duboki MLP s odabirljivom aktivacijom i inicijalizacijom
    * mnist_dataset -- klasa za MNIST skup s train/valid/test podskupovima
    * ordinary_dataloader -- jednostavan PyTorch DataLoader wrapper

Funkcionalnosti:
    * L1 i L2 regularizacija preko težina
    * praćenje train/valid loss-a
    * praćenje valid macro F1 po epohama
    * norme gradijenata po slojevima (za nestajanje/eksplodiranje)
    * distribucije aktivacija po slojevima (po epohama)
    * distribucije gradijenata po slojevima (po epohama)
"""

from typing import Optional, Dict, List
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import os, sys

# Add project files
current_file_path = os.path.abspath(__file__)
main_dir = os.path.dirname(current_file_path)
sys.path.append(main_dir)

from models import DeepMLPNet
from dataloader import mnist_dataset, ordinary_dataloader


class DeepMLPTrainer:
    """
    Klasa koja enkapsulira proces treniranja modela (npr. DeepMLPNet)
    na danim train/valid dataloaderima.

    Args:
        * model, nn.Module, instanca mreže
        * train_loader, DataLoader, dataloader za trening skup
        * valid_loader, DataLoader ili None, dataloader za validacijski skup
        * num_epochs, int, broj epoha u treniranju
        * learning_rate, float, početni learning rate (Adam)
        * device, torch.device ili None, uređaj ('cuda' ili 'cpu'); ako je None,
          automatski se detektira
        * max_train_batches, int ili None, maksimalan broj batch-eva po epohi
        * track_gradients, bool, ako je True, prati norme gradijenata po slojevima
        * l1_lambda, float, jakost L1 regularizacije
        * l2_lambda, float, jakost L2 regularizacije
        * verbose, bool, ako je True, ispisuje loss/F1 po epohama

    Spremamo:
        * train_loss_history
        * valid_loss_history
        * valid_f1_history
        * grad_history[param_name] -> L2 norm gradijenta po epohama
        * activation_history[layer_name][epoch] -> 1D tensor aktivacija (prvi batch)
        * grad_distribution_history[param_name][epoch] -> 1D tensor gradijenata (prvi batch)
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        valid_loader: Optional[DataLoader] = None,
        num_epochs: int = 8,
        learning_rate: float = 1e-3,
        device: Optional[torch.device] = None,
        max_train_batches: Optional[int] = 150,
        track_gradients: bool = True,
        l1_lambda: float = 0.0,
        l2_lambda: float = 0.0,
        verbose: bool = True,
    ):
        # --- Device ---
        _device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Korišteni uređaj:", _device)

        self.device = _device
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.max_train_batches = max_train_batches
        self.track_gradients = track_gradients
        self.l1_lambda = float(l1_lambda)
        self.l2_lambda = float(l2_lambda)
        self.verbose = verbose

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.learning_rate)

        # Povijesti
        self.train_loss_history: List[float] = []
        self.valid_loss_history: List[float] = []
        self.valid_f1_history: List[float] = []
        self.grad_history: Dict[str, List[float]] = defaultdict(list)

        # Aktivacije / gradijenti (distribucije po epohama)
        self.activation_history: Dict[str, List[torch.Tensor]] = defaultdict(list)
        self._current_epoch_idx: int = -1
        self._record_activation_this_epoch: bool = False
        self.max_activation_samples: int = 2000

        self.grad_distribution_history: Dict[str, List[torch.Tensor]] = defaultdict(list)
        self.max_grad_samples: int = 2000

        # Hookovi za aktivacije
        self._register_activation_hooks()

        # ==== Ispis svih postavki za run ====
        print("==== Postavke modela ====")
        if isinstance(self.model, DeepMLPNet):
            if len(self.model.hidden_layers) > 0:
                _first = self.model.hidden_layers[0]
                print(f"  input_dim        = {_first.in_features}")
                print(f"  hidden_dim       = {_first.out_features}")
                print(f"  num_layers       = {len(self.model.hidden_layers)}")
            else:
                print("  (model nema skrivenih slojeva?)")
            print(f"  num_classes      = {self.model.output_layer.out_features}")
            print(f"  activation       = {self.model.activation}")
            print(f"  init_type        = {self.model.init_type}")
            print(f"  init_scale       = {self.model.init_scale}")
            dropout_prob = getattr(self.model, "dropout_prob", 0.0)
            print(f"  dropout_prob     = {dropout_prob}")
        else:
            print(f"  model type       = {type(self.model)} (detalji nisu specificirani)")

        print("==== Postavke treniranja ====")
        print(f"  num_epochs       = {self.num_epochs}")
        print(f"  learning_rate    = {self.learning_rate}")
        print(f"  max_train_batches= {self.max_train_batches}")
        print(f"  track_gradients  = {self.track_gradients}")
        print(f"  l1_lambda        = {self.l1_lambda}")
        print(f"  l2_lambda        = {self.l2_lambda}")
        print(f"  verbose          = {self.verbose}")
        print("================================\n")

    # ---------- Hookovi za aktivacije ----------
    def _register_activation_hooks(self) -> None:
        """
        Registrira forward hookove na Linear slojeve (skrivene + izlazni)
        kako bi se pratile distribucije aktivacija po epohama.
        """
        for _name, _module in self.model.named_modules():
            if isinstance(_module, nn.Linear) and (
                "hidden_layers" in _name or "output_layer" in _name
            ):
                _module.register_forward_hook(self._make_activation_hook(_name))

    def _make_activation_hook(self, layer_name: str):
        def hook(module, inputs, output):
            if not self._record_activation_this_epoch:
                return
            _act = output
            if isinstance(_act, tuple):
                _act = _act[0]
            _act = _act.detach().view(-1)
            if _act.numel() > self.max_activation_samples:
                _idx = torch.randperm(_act.numel(), device=_act.device)[: self.max_activation_samples]
                _act = _act[_idx]
            self.activation_history[layer_name].append(_act.cpu())
        return hook

    # ---------- Pomoćne metode ----------
    def _record_grad_norms(self) -> None:
        """
        Sprema L2 norme gradijenata za sve težinske parametre (po epohama).
        """
        for _name, _param in self.model.named_parameters():
            if _param.grad is not None and "weight" in _name:
                self.grad_history[_name].append(_param.grad.norm().item())

    def _record_grad_distributions_first_batch(self) -> None:
        """
        Sprema distribucije gradijenata težina za PRVI batch u epohi
        (1D tensor, eventualno subsamplan).
        """
        for _name, _param in self.model.named_parameters():
            if _param.grad is not None and "weight" in _name:
                _g = _param.grad.detach().view(-1)
                if _g.numel() > self.max_grad_samples:
                    _idx = torch.randperm(_g.numel(), device=_g.device)[: self.max_grad_samples]
                    _g = _g[_idx]
                self.grad_distribution_history[_name].append(_g.cpu())

    def _compute_regularization_loss(self) -> torch.Tensor:
        """
        Računa L1 i L2 regularizacijsku kaznu preko težina (weight parametara).

        Returns:
            * torch.Tensor, skalar:
              l1_lambda * sum|w| + l2_lambda * sum(w^2)
        """
        if self.l1_lambda == 0.0 and self.l2_lambda == 0.0:
            return torch.tensor(0.0, device=self.device)

        _l1_penalty = torch.tensor(0.0, device=self.device)
        _l2_penalty = torch.tensor(0.0, device=self.device)

        for _name, _param in self.model.named_parameters():
            if _param.requires_grad and "weight" in _name:
                if self.l1_lambda != 0.0:
                    _l1_penalty = _l1_penalty + _param.abs().sum()
                if self.l2_lambda != 0.0:
                    _l2_penalty = _l2_penalty + (_param ** 2).sum()

        _reg_loss = self.l1_lambda * _l1_penalty + self.l2_lambda * _l2_penalty
        return _reg_loss

    def _train_one_epoch(self, _epoch_idx: int) -> float:
        """
        Treniranje modela kroz jednu epohu na train_loader-u.
        """
        self.model.train()
        _running_loss = 0.0

        self._current_epoch_idx = _epoch_idx - 1
        self._record_activation_this_epoch = True

        for _batch_idx, (_images, _labels) in enumerate(self.train_loader):
            _images = _images.to(self.device)
            _labels = _labels.to(self.device)

            self.optimizer.zero_grad()
            _logits = self.model(_images)

            _loss_data = self.criterion(_logits, _labels)
            _loss_reg = self._compute_regularization_loss()
            _loss = _loss_data + _loss_reg

            _loss.backward()

            if self.track_gradients and _batch_idx == 0:
                self._record_grad_norms()
                self._record_grad_distributions_first_batch()

            self.optimizer.step()
            _running_loss += _loss.item()

            # aktivacije snimamo samo za prvi batch u epohi
            if self._record_activation_this_epoch:
                self._record_activation_this_epoch = False

            if (self.max_train_batches is not None) and \
               (_batch_idx + 1 >= self.max_train_batches):
                break

        _avg_train_loss = _running_loss / (_batch_idx + 1)
        return _avg_train_loss

    @staticmethod
    def _compute_macro_f1(_y_true: torch.Tensor,
                          _y_pred: torch.Tensor) -> float:
        """
        Računa macro F1 mjeru za višeklasnu klasifikaciju.
        """
        _y_true = _y_true.view(-1)
        _y_pred = _y_pred.view(-1)

        if _y_true.numel() == 0:
            return 0.0

        _num_classes = int(max(int(_y_true.max()), int(_y_pred.max()))) + 1
        _f1_per_class = []

        for _c in range(_num_classes):
            _tp = ((_y_pred == _c) & (_y_true == _c)).sum().item()
            _fp = ((_y_pred == _c) & (_y_true != _c)).sum().item()
            _fn = ((_y_pred != _c) & (_y_true == _c)).sum().item()

            _precision = _tp / (_tp + _fp) if (_tp + _fp) > 0 else 0.0
            _recall = _tp / (_tp + _fn) if (_tp + _fn) > 0 else 0.0

            if (_precision + _recall) > 0:
                _f1 = 2.0 * _precision * _recall / (_precision + _recall)
            else:
                _f1 = 0.0

            _f1_per_class.append(_f1)

        if len(_f1_per_class) == 0:
            return 0.0

        _macro_f1 = float(sum(_f1_per_class) / len(_f1_per_class))
        return _macro_f1

    def _validate_one_epoch(self) -> (float, float):
        """
        Evaluacija modela na valid_loader-u (ako postoji).

        Returns:
            * float, prosječni valid loss za ovu epohu
            * float, macro F1 na validacijskom skupu
        """
        if self.valid_loader is None:
            return float("nan"), float("nan")

        self.model.eval()
        _valid_running_loss = 0.0
        _num_valid_batches = 0

        _all_preds = []
        _all_targets = []

        with torch.no_grad():
            for _v_images, _v_labels in self.valid_loader:
                _v_images = _v_images.to(self.device)
                _v_labels = _v_labels.to(self.device)

                _v_logits = self.model(_v_images)
                _v_loss = self.criterion(_v_logits, _v_labels)

                _valid_running_loss += _v_loss.item()
                _num_valid_batches += 1

                _v_pred = torch.argmax(_v_logits, dim=1)

                _all_preds.append(_v_pred.cpu())
                _all_targets.append(_v_labels.cpu())

        _avg_valid_loss = _valid_running_loss / max(1, _num_valid_batches)

        if len(_all_preds) > 0:
            _all_preds = torch.cat(_all_preds, dim=0)
            _all_targets = torch.cat(_all_targets, dim=0)
            _macro_f1 = self._compute_macro_f1(_all_targets, _all_preds)
        else:
            _macro_f1 = float("nan")

        return _avg_valid_loss, _macro_f1

    def _plot_gradients(self) -> None:
        """
        Crtanje normi gradijenata po slojevima kroz epohe (log skala).
        """
        if not self.track_gradients or len(self.grad_history) == 0:
            print("Nema spremljenih gradijenata za crtanje.")
            return

        _all_weight_names = [name for name in self.grad_history.keys()]
        print("Praćeni slojevi:", _all_weight_names)

        _num_layers_total = sum(1 for _n in _all_weight_names
                                if "hidden_layers" in _n)
        _num_samples = min(5, _num_layers_total) if _num_layers_total > 0 else 0

        _layers_to_plot: List[str] = []
        if _num_samples > 0:
            _sample_indices = np.linspace(0, _num_layers_total - 1,
                                          _num_samples, dtype=int)
            for _idx in _sample_indices:
                _layer_name = f"hidden_layers.{_idx}.weight"
                if _layer_name in self.grad_history:
                    _layers_to_plot.append(_layer_name)

        if "output_layer.weight" in self.grad_history:
            _layers_to_plot.append("output_layer.weight")

        print("Slojevi za crtanje:", _layers_to_plot)

        _epochs_axis = range(1, len(self.train_loss_history) + 1)

        plt.figure(figsize=(9, 5))
        for _layer_name in _layers_to_plot:
            if _layer_name in self.grad_history:
                plt.plot(
                    _epochs_axis,
                    self.grad_history[_layer_name],
                    marker="o",
                    label=_layer_name,
                )

        plt.yscale("log")
        plt.xlabel("Epoh")
        plt.ylabel("Norma gradijenta (log skala)")
        plt.title("Norme gradijenata kroz epohe")
        plt.legend(fontsize=8)
        plt.grid(True, which="both", linestyle="--", linewidth=0.3)
        plt.tight_layout()
        plt.show()

    def train(self, plot_gradients: bool = True) -> Dict[str, List[float]]:
        """
        Glavna metoda koja pokreće treniranje kroz zadani broj epoha
        i (opcionalno) crta norme gradijenata te Train vs Valid loss graf.

        Returns:
            * dict s listama:
                - "train_loss": lista prosječnih train loss-eva po epohi
                - "valid_loss": lista prosječnih valid loss-eva po epohi (ako postoji)
                - "valid_macro_f1": lista macro F1 vrijednosti po epohi (ako postoji)
        """
        for _epoch in range(1, self.num_epochs + 1):
            _avg_train_loss = self._train_one_epoch(_epoch)
            self.train_loss_history.append(_avg_train_loss)

            _avg_valid_loss, _macro_f1 = float("nan"), float("nan")
            if self.valid_loader is not None:
                _avg_valid_loss, _macro_f1 = self._validate_one_epoch()
                self.valid_loss_history.append(_avg_valid_loss)
                self.valid_f1_history.append(_macro_f1)

            if self.verbose:
                if self.valid_loader is not None:
                    print(
                        f"Epoch {_epoch:02d} | "
                        f"train loss: {_avg_train_loss:.4f} | "
                        f"valid loss: {_avg_valid_loss:.4f} | "
                        f"valid macro F1: {_macro_f1:.4f}"
                    )
                else:
                    print(
                        f"Epoch {_epoch:02d} | "
                        f"train loss: {_avg_train_loss:.4f}"
                    )

        if plot_gradients:
            self._plot_gradients()

        if len(self.train_loss_history) > 0:
            _epochs_axis = range(1, len(self.train_loss_history) + 1)

            plt.figure(figsize=(8, 5))
            plt.plot(
                _epochs_axis,
                self.train_loss_history,
                marker="o",
                linewidth=2,
                label="Train loss",
            )

            if len(self.valid_loss_history) > 0:
                plt.plot(
                    _epochs_axis,
                    self.valid_loss_history,
                    marker="s",
                    linewidth=2,
                    label="Valid loss",
                )

            plt.xlabel("Epoh")
            plt.ylabel("Loss")
            plt.title("Train vs Valid Loss kroz epohe")
            plt.grid(True, linestyle="--", linewidth=0.4)
            plt.legend()
            plt.tight_layout()
            plt.show()

        _history_dict: Dict[str, List[float]] = {"train_loss": self.train_loss_history}
        if self.valid_loader is not None:
            _history_dict["valid_loss"] = self.valid_loss_history
            _history_dict["valid_macro_f1"] = self.valid_f1_history

        return _history_dict
