"""
Skripta u kojoj se implementiraju modeli za vježbu 4
"""

# Knjižnice
import torch
from torch import nn
import torchinfo
from torchview import draw_graph
from IPython.display import display, SVG
import numpy as np
from typing import Tuple

def plot_graph(model: torch.nn.Module = None, 
               device:str = 'cpu',
               input_dim = None):
    """
    Funkcija koja crta graf modela.

    Argumenti:
        * model, pytorch model
        * device, str, gdje se nalaze model i podaci, zadano je: cpu
        * input_dim, tupple, ulazne dimenzije modela
    """
    model = model.to(device).eval()

    _graph = draw_graph(
        model,
        input_size= input_dim,   # e.g. (1, 3, 224, 224)
        expand_nested=True,
        graph_name="Model",
        device=device,
    )

    # Option A: display directly from the Graphviz object
    _svg_bytes = _graph.visual_graph.pipe(format="svg")  # no temp file
    display(SVG(_svg_bytes))

def print_model_summary(model: torch.nn.Module = None, 
                        device:str = 'cpu',
                        input_dim = None):
    """
    Funkcija koja ispisuje sažetak modela.

    Argumenti:
        * model, pytorch model.
        * device, str, gdje se nalaze model i podaci, zadano je: cpu.
        * input_dim, tupple, ulazne dimenzije modela.

    Izlaz:
        * vraća ispisivi sažetak modela
    """
    return torchinfo.summary(model=model, 
            input_size= input_dim, # make sure this is "input_size", not "input_shape"
            col_names=["input_size", "output_size", "num_params", "trainable"],
            col_width=16,
            device = device,
            row_settings=["var_names"]) 

class model_2_3_1(nn.Module):
    """
    Jednostavan model s 2 ulazna neurona, 3 skrivena neurona i 1 izlazom.

    y = psi0 + psi1 * a1(t10 + t11*x1 + t12*x2) + psi2 * a2(t20 + t21*x1 + t22*x2)
        + psi3 * a3(t30 + t31*x1 + t32*x2)

    Args:
        *theta_init_matrix, np.ndarray oblika (3, 3),
            redovi: po jedan za svaki skriveni neuron (1..3)
            stupci: [theta_i0, theta_i1, theta_i2] (bias, w.r.t x1, w.r.t x2)
        *psi_init_matrix, np.ndarray, oblika (4,) [psi0, psi1, psi2, psi3]
        *activation_function, str, aktivacijska funkcija {"relu", "tanh", "sigmoid"} ili None (Identity)
    """
    def __init__(self,
                 theta_init_matrix: np.ndarray = None,
                 psi_init_matrix: np.ndarray = None,
                 activation_function: str = None):
        super().__init__()

        # ---- Safety / defaults (optional, but handy) ----
        if theta_init_matrix is None:
            theta_init_matrix = np.zeros((3, 3), dtype=np.float32)
        if psi_init_matrix is None:
            psi_init_matrix = np.zeros((4,), dtype=np.float32)

        # očekujemo (3,3)
        assert theta_init_matrix.shape == (3, 3), "theta_init_matrix must be shape (3,3): [[t10,t11,t12],[t20,t21,t22],[t30,t31,t32]]"
        assert psi_init_matrix.shape == (4,), "psi_init_matrix must be shape (4,): [psi0,psi1,psi2,psi3]"

        # --- Theta parameters (po skrivenom neuronu: bias + 2 težine) ---
        # Hidden 1
        self.theta10 = nn.Parameter(torch.tensor(theta_init_matrix[0, 0], dtype=torch.float32))  # bias
        self.theta11 = nn.Parameter(torch.tensor(theta_init_matrix[0, 1], dtype=torch.float32))  # w.r.t x1
        self.theta12 = nn.Parameter(torch.tensor(theta_init_matrix[0, 2], dtype=torch.float32))  # w.r.t x2
        # Hidden 2
        self.theta20 = nn.Parameter(torch.tensor(theta_init_matrix[1, 0], dtype=torch.float32))
        self.theta21 = nn.Parameter(torch.tensor(theta_init_matrix[1, 1], dtype=torch.float32))
        self.theta22 = nn.Parameter(torch.tensor(theta_init_matrix[1, 2], dtype=torch.float32))
        # Hidden 3
        self.theta30 = nn.Parameter(torch.tensor(theta_init_matrix[2, 0], dtype=torch.float32))
        self.theta31 = nn.Parameter(torch.tensor(theta_init_matrix[2, 1], dtype=torch.float32))
        self.theta32 = nn.Parameter(torch.tensor(theta_init_matrix[2, 2], dtype=torch.float32))

        # --- Psi parameters ---
        self.psi0 = nn.Parameter(torch.tensor(psi_init_matrix[0], dtype=torch.float32))
        self.psi1 = nn.Parameter(torch.tensor(psi_init_matrix[1], dtype=torch.float32))
        self.psi2 = nn.Parameter(torch.tensor(psi_init_matrix[2], dtype=torch.float32))
        self.psi3 = nn.Parameter(torch.tensor(psi_init_matrix[3], dtype=torch.float32))

        # --- Activation selection ---
        _act_map = {
            None: nn.Identity,
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "sigmoid": nn.Sigmoid
        }
        _act_key = None if activation_function is None else activation_function.lower()
        _act_cls = _act_map.get(_act_key, nn.Identity)

        self.activation1 = _act_cls()
        self.activation2 = _act_cls()
        self.activation3 = _act_cls()

    def forward(self, 
                x1: torch.Tensor = None,
                x2: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x1: Tensor za prvi ulaz (može imati bilo koji batch shape).
            x2: Tensor za drugi ulaz (isti ili broadcast-kompatibilan batch shape kao x1).

        Returns:
            y: skalar po uzorku s istim (broadcastanim) batch shapeom.
        """
        if x1 is None or x2 is None:
            raise ValueError("Both x1 and x2 must be provided.")

        _x1 = torch.as_tensor(x1, device=self.theta10.device, dtype=self.theta10.dtype)
        _x2 = torch.as_tensor(x2, device=self.theta10.device, dtype=self.theta10.dtype)

        # Osiguraj kompatibilne dimenzije (broadcast ako je potrebno)
        _x1, _x2 = torch.broadcast_tensors(_x1, _x2)

        # tri afine transformacije
        z1 = self.theta10 + self.theta11 * _x1 + self.theta12 * _x2
        z2 = self.theta20 + self.theta21 * _x1 + self.theta22 * _x2
        z3 = self.theta30 + self.theta31 * _x1 + self.theta32 * _x2

        # aktivacije
        h1 = self.activation1(z1)
        h2 = self.activation2(z2)
        h3 = self.activation3(z3)

        # ponderiranje psijima i izlaz
        y = self.psi0 + self.psi1 * h1 + self.psi2 * h2 + self.psi3 * h3
        return y

class model_1_3_2(nn.Module):
    """
    Jednostavan model s 1 ulaznim neuronom, 3 skrivena neurona i 2 izlaza.

    Zajednički skriveni sloj:
      z1 = t10 + t11 * x
      z2 = t20 + t21 * x
      z3 = t30 + t31 * x
      h_i = a_i(z_i)

    Dva izlaza dijele iste aktivacije, ali imaju zasebne psi koeficijente:
      y1 = psi0a + psi1a*h1 + psi2a*h2 + psi3a*h3
      y2 = psi0b + psi1b*h1 + psi2b*h2 + psi3b*h3

    Args:
        theta_init_matrix: np.ndarray oblika (3, 2)
            redovi: po jedan za svaki skriveni neuron (1..3)
            stupci: [theta_i0 (bias), theta_i1 (w.r.t x)]
        psi_init_matrix: np.ndarray oblika (2, 4)
            redovi: po jedan za svaki izlaz (y1, y2)
            stupci: [psi0, psi1, psi2, psi3] (bias + težine za h1..h3)
        activation_function: {"relu", "tanh", "sigmoid"} ili None (Identity)
    """
    def __init__(self,
                 theta_init_matrix: np.ndarray = None,
                 psi_init_matrix:   np.ndarray = None,
                 activation_function: str = None):
        super().__init__()

        # ---- Defaults / checks ----
        if theta_init_matrix is None:
            theta_init_matrix = np.zeros((3, 2), dtype=np.float32)
        if psi_init_matrix is None:
            psi_init_matrix = np.zeros((2, 4), dtype=np.float32)

        assert theta_init_matrix.shape == (3, 2), \
            "theta_init_matrix must be (3,2): [[t10,t11],[t20,t21],[t30,t31]]"
        assert psi_init_matrix.shape == (2, 4), \
            "psi_init_matrix must be (2,4): [[psi0a,psi1a,psi2a,psi3a],[psi0b,psi1b,psi2b,psi3b]]"

        # --- Theta parameters (po skrivenom neuronu: bias + 1 težina) ---
        # Hidden 1
        self.theta10 = nn.Parameter(torch.tensor(theta_init_matrix[0, 0], dtype=torch.float32))  # bias
        self.theta11 = nn.Parameter(torch.tensor(theta_init_matrix[0, 1], dtype=torch.float32))  # w.r.t x
        # Hidden 2
        self.theta20 = nn.Parameter(torch.tensor(theta_init_matrix[1, 0], dtype=torch.float32))
        self.theta21 = nn.Parameter(torch.tensor(theta_init_matrix[1, 1], dtype=torch.float32))
        # Hidden 3
        self.theta30 = nn.Parameter(torch.tensor(theta_init_matrix[2, 0], dtype=torch.float32))
        self.theta31 = nn.Parameter(torch.tensor(theta_init_matrix[2, 1], dtype=torch.float32))

        # --- Psi parameters (dva seta) ---
        # Za y1:
        self.psi0a = nn.Parameter(torch.tensor(psi_init_matrix[0, 0], dtype=torch.float32))
        self.psi1a = nn.Parameter(torch.tensor(psi_init_matrix[0, 1], dtype=torch.float32))
        self.psi2a = nn.Parameter(torch.tensor(psi_init_matrix[0, 2], dtype=torch.float32))
        self.psi3a = nn.Parameter(torch.tensor(psi_init_matrix[0, 3], dtype=torch.float32))
        # Za y2:
        self.psi0b = nn.Parameter(torch.tensor(psi_init_matrix[1, 0], dtype=torch.float32))
        self.psi1b = nn.Parameter(torch.tensor(psi_init_matrix[1, 1], dtype=torch.float32))
        self.psi2b = nn.Parameter(torch.tensor(psi_init_matrix[1, 2], dtype=torch.float32))
        self.psi3b = nn.Parameter(torch.tensor(psi_init_matrix[1, 3], dtype=torch.float32))

        # --- Activation selection ---
        _act_map = {
            None: nn.Identity,
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "sigmoid": nn.Sigmoid
        }
        _act_key = None if activation_function is None else activation_function.lower()
        _act_cls = _act_map.get(_act_key, nn.Identity)

        self.activation1 = _act_cls()
        self.activation2 = _act_cls()
        self.activation3 = _act_cls()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Tensor ulaza (može imati bilo koji batch shape)

        Returns:
            (y1, y2): dva izlaza s istim batch shapeom kao x
        """
        _x = torch.as_tensor(x, device=self.theta10.device, dtype=self.theta10.dtype)

        # tri afine transformacije
        _z1 = self.theta10 + self.theta11 * _x
        _z2 = self.theta20 + self.theta21 * _x
        _z3 = self.theta30 + self.theta31 * _x

        # aktivacije
        _h1 = self.activation1(_z1)
        _h2 = self.activation2(_z2)
        _h3 = self.activation3(_z3)

        # izlaz 1
        _y1 = self.psi0a + self.psi1a * _h1 + self.psi2a * _h2 + self.psi3a * _h3
        # izlaz 2
        _y2 = self.psi0b + self.psi1b * _h1 + self.psi2b * _h2 + self.psi3b * _h3

        return _y1, _y2