"""
Main class for deep MLP model on MNIST with selectable activation function,
various weight initialisation schemes and optional dropout.
"""

# Učitavanje knjižnica
from typing import Literal
import torchinfo
import torch
import torch.nn as nn


def print_model_summary(model: torch.nn.Module = None,
                        device: str = 'cpu',
                        input_dim=None):
    """
    Funkcija koja ispisuje sažetak modela.

    Argumenti:
        * model, pytorch model.
        * device, str, gdje se nalaze model i podaci, zadano je: cpu.
        * input_dim, tupple, ulazne dimenzije modela.

    Izlaz:
        * vraća ispisivi sažetak modela
    """
    return torchinfo.summary(
        model=model,
        input_size=input_dim,
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=16,
        device=device,
        row_settings=["var_names"]
    )


class DeepMLPNet(nn.Module):
    """
    Duboka potpuno povezana mreža (MLP) s:
      - odabirljivom aktivacijom (ReLU, sigmoid ili tanh)
      - odabirljivom inicijalizacijom težina:
          * 'xavier_uniform'
          * 'xavier_normal'
          * 'kaiming_uniform' (He)
          * 'zeros'          (sve težine = 0)
          * 'constant'       (sve težine = isti broj)
      - opcionalnim dropoutom u skrivenim slojevima

    Arhitektura:
        - Ulaz: spljoštena 28x28 MNIST slika (zadano)
        - `num_layers` skrivenih slojeva sa `hidden_dim` neurona
        - iza svakog skrivenog sloja:
            Linear -> aktivacija -> (opcijski Dropout)
        - Linearni izlazni sloj s `num_classes` neurona
    """

    def __init__(
        self,
        input_dim: int = 28 * 28,
        hidden_dim: int = 256,
        num_layers: int = 20,
        num_classes: int = 10,
        init_scale: float = 1.0,
        activation: Literal["relu", "sigmoid", "tanh"] = "sigmoid",
        init_type: Literal[
            "xavier_uniform",
            "xavier_normal",
            "kaiming_uniform",
            "zeros",
            "constant",
        ] = "xavier_uniform",
        init_constant_value: float = 0.01,
        dropout_prob: float = 0.0,
    ):
        super().__init__()

        self.init_scale = init_scale
        self.activation = activation.lower()
        self.init_type = init_type.lower()
        self.init_constant_value = init_constant_value
        self.dropout_prob = float(dropout_prob)

        assert self.activation in ["relu", "sigmoid", "tanh"], \
            f"activation mora biti 'relu', 'sigmoid' ili 'tanh', dobiveno: {activation}"

        assert self.init_type in [
            "xavier_uniform",
            "xavier_normal",
            "kaiming_uniform",
            "zeros",
            "constant",
        ], (
            "init_type mora biti jedan od: "
            "'xavier_uniform', 'xavier_normal', 'kaiming_uniform', 'zeros', 'constant' "
            f"(dobiveno: {init_type})"
        )

        assert 0.0 <= self.dropout_prob < 1.0, \
            f"dropout_prob mora biti u [0, 1), dobiveno: {dropout_prob}"

        _layers = []
        _last_dim = input_dim

        for _ in range(num_layers):
            _fc = nn.Linear(_last_dim, hidden_dim)
            self._init_weights(_fc)
            _layers.append(_fc)
            _last_dim = hidden_dim

        self.hidden_layers = nn.ModuleList(_layers)

        self.output_layer = nn.Linear(_last_dim, num_classes)
        self._init_weights(self.output_layer)

        # Dropout modul (ako je p=0, Dropout se efektivno ne primjenjuje)
        if self.dropout_prob > 0.0:
            self.dropout = nn.Dropout(p=self.dropout_prob)
        else:
            self.dropout = nn.Identity()

    def _init_weights(self, _layer: nn.Linear) -> None:
        """
        Pomoćna metoda za inicijalizaciju težina jednog Linear sloja
        prema odabranom `init_type` i `init_scale`.
        """
        if self.init_type == "xavier_uniform":
            nn.init.xavier_uniform_(_layer.weight)
            nn.init.zeros_(_layer.bias)
        elif self.init_type == "xavier_normal":
            nn.init.xavier_normal_(_layer.weight)
            nn.init.zeros_(_layer.bias)
        elif self.init_type == "kaiming_uniform":
            nn.init.kaiming_uniform_(
                _layer.weight,
                nonlinearity="relu",
            )
            nn.init.zeros_(_layer.bias)
        elif self.init_type == "zeros":
            nn.init.zeros_(_layer.weight)
            nn.init.zeros_(_layer.bias)
        elif self.init_type == "constant":
            nn.init.constant_(_layer.weight, self.init_constant_value)
            nn.init.zeros_(_layer.bias)
        else:
            raise ValueError(f"Nepodržan init_type: {self.init_type}")

        if self.init_type not in ["zeros", "constant"]:
            _layer.weight.data *= self.init_scale

    def _act_fn(self, _x: torch.Tensor) -> torch.Tensor:
        if self.activation == "sigmoid":
            return torch.sigmoid(_x)
        elif self.activation == "relu":
            return torch.relu(_x)
        elif self.activation == "tanh":
            return torch.tanh(_x)
        raise ValueError(f"Nepodržana aktivacija: {self.activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Standardni forward prolaz kroz mrežu:
        - spljoštavanje slike ako je potrebno
        - prolazak kroz skrivene slojeve + aktivacija + (opcijski dropout)
        - izlazni linearni sloj bez aktivacije (logiti)
        """
        if x.dim() > 2:
            _x = x.view(x.size(0), -1)
        else:
            _x = x

        for _layer in self.hidden_layers:
            _x = self._act_fn(_layer(_x))
            _x = self.dropout(_x)

        _x = self.output_layer(_x)
        return _x
