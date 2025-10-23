"""
Skripta u kojoj se implementiraju modeli za vježbu 3
"""

# Knjižnice
import torch
from torch import nn
import numpy as np
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple

class model_1_3_1(nn.Module):
    """
    #NUŽNA METODA#
    Jednostavan model koji implementira pravac.
    Dakle 1 ulazna podatak, dva parametra, jedan izlazan:

    y = psi0 + psi1*(a1(t10 + t11*x)) + psi2*(a2(t20 + t21*x)) + psi3*(a3(t30 + t31*x))
    """
    def __init__(self, 
                 theta_init_matrix: np.array = None,
                 psi_init_matrix: np.array = None,
                 activation_function: str = None):
        """
        Init metoda u kojoj defirinamo slojeve neuronske mreže i po potrebi dodatne parametre

        Args:
            * theta_init_matrix, 2D nd.array, 6 broja gdje prvi redak predstavlja vrijednosti theta za  h1, 
            drugi za h2, i treći za h3. Isto tako uvijek se prvo navodi theta0 pa theta1 (t10 pa t11, t20 pa t21,
            i na kraju t30 pa t31)
            * psi_init_matrix, 1d nd.array, 4 broja koja prikazaju 4 
            * activation_function, str, vrsta aktivacijske funckije. Može biti
                - tanh: tangens hiperbolni
                - relu: rectified linear unit
                - sigmoid: za sigmoid funkciju.
        """
        super().__init__()
        # Set parameters
        self.theta10 = nn.Parameter(torch.tensor(theta_init_matrix[0,0], dtype=torch.float32))
        self.theta11 = nn.Parameter(torch.tensor(theta_init_matrix[0,1], dtype=torch.float32))
        self.theta20 = nn.Parameter(torch.tensor(theta_init_matrix[1,0], dtype=torch.float32))
        self.theta21 = nn.Parameter(torch.tensor(theta_init_matrix[1,1], dtype=torch.float32))
        self.theta30 = nn.Parameter(torch.tensor(theta_init_matrix[2,0], dtype=torch.float32))
        self.theta31 = nn.Parameter(torch.tensor(theta_init_matrix[2,1], dtype=torch.float32))

        self.psi0 = nn.Parameter(torch.tensor(psi_init_matrix[0], dtype=torch.float32))
        self.psi1 = nn.Parameter(torch.tensor(psi_init_matrix[1], dtype=torch.float32))
        self.psi2 = nn.Parameter(torch.tensor(psi_init_matrix[2], dtype=torch.float32))
        self.psi3 = nn.Parameter(torch.tensor(psi_init_matrix[3], dtype=torch.float32))


        # Set activation function
        self.activation = None
        if activation_function is not None:
            _activation_function = activation_function.lower()
            if _activation_function == "relu": _act = nn.ReLU
            if _activation_function == "tanh": _act = nn.Tanh
            if _activation_function == "sigmoid": _act = nn.Sigmoid
        else:
            # Free pass - placeholder
            _act = nn.Identity
        self.activation1 = _act()
        self.activation2 = _act()
        self.activation3 = _act()


    def forward(self, x: torch.Tensor)->torch.Tensor:
        """
        #NUŽNA METODA#
        Forward pass. Služi učenju neuronskih mreža.

        Args:
            * x, torch.Tensor, input data
        
        Returns:
            y = psi0 + psi1*(a1(t10 + t11*x)) + psi2*(a2(t20 + t21*x)) + psi3*(a3(t30 + t31*x))

        """ 
        _x = torch.as_tensor(x, device=self.theta10.device, dtype=self.theta10.dtype)

        # three affine branches
        _z1 = self.theta10 + self.theta11 * _x
        _z2 = self.theta20 + self.theta21 * _x
        _z3 = self.theta30 + self.theta31 * _x

        # activations
        _h1 = self.activation1(_z1)
        _h2 = self.activation2(_z2)
        _h3 = self.activation3(_z3)
        
        # Multiplications
        _w1 = self.psi1 * _h1
        _w2 = self.psi2 * _h2
        _w3 = self.psi3 * _h3

        # linear combination with psi's
        _y = self.psi0 + _w1 + _w2 + _w3 
        return _y



class model_2_3_1_for_plots(nn.Module):
    """
    #NUŽNA METODA#
    Jednostavan model s 2 ulazna podatka, 3 skrivena neurona i 1 izlazom:

    y = psi0
        + psi1 * a1(t10 + t11*x1 + t12*x2)
        + psi2 * a2(t20 + t21*x1 + t22*x2)
        + psi3 * a3(t30 + t31*x1 + t32*x2)
    """
    def __init__(self, 
                 theta_init_matrix: np.ndarray = None,
                 psi_init_matrix:   np.ndarray = None,
                 activation_function: str = None):
        """
        Init metoda u kojoj definiramo slojeve i parametre.

        Args:
            * theta_init_matrix, 2D np.ndarray oblika (3,3):
              redovi -> skriveni neuroni (1..3)
              stupci -> [theta_i0 (bias), theta_i1 (x1), theta_i2 (x2)]
            * psi_init_matrix, 1D np.ndarray oblika (4,): [psi0, psi1, psi2, psi3]
            * activation_function, str u {"tanh","relu","sigmoid"} ili None (Identity)
        """
        super().__init__()

        # Defaults / provjere
        if theta_init_matrix is None:
            theta_init_matrix = np.zeros((3, 3), dtype=np.float32)
        if psi_init_matrix is None:
            psi_init_matrix = np.zeros((4,), dtype=np.float32)

        assert theta_init_matrix.shape == (3, 3), \
            "theta_init_matrix must be shape (3,3): [[t10,t11,t12],[t20,t21,t22],[t30,t31,t32]]"
        assert psi_init_matrix.shape == (4,), \
            "psi_init_matrix must be shape (4,): [psi0,psi1,psi2,psi3]"

        # --- Theta parametri (po skrivenom neuronu: bias + 2 težine) ---
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

        # --- Psi parametri ---
        self.psi0 = nn.Parameter(torch.tensor(psi_init_matrix[0], dtype=torch.float32))
        self.psi1 = nn.Parameter(torch.tensor(psi_init_matrix[1], dtype=torch.float32))
        self.psi2 = nn.Parameter(torch.tensor(psi_init_matrix[2], dtype=torch.float32))
        self.psi3 = nn.Parameter(torch.tensor(psi_init_matrix[3], dtype=torch.float32))

        # --- Aktivacija ---
        _act = nn.Identity
        if activation_function is not None:
            _activation_function = activation_function.lower()
            if _activation_function == "relu":    _act = nn.ReLU
            if _activation_function == "tanh":    _act = nn.Tanh
            if _activation_function == "sigmoid": _act = nn.Sigmoid
        self.activation1 = _act()
        self.activation2 = _act()
        self.activation3 = _act()

    def forward(self, 
                x1: torch.Tensor,
                x2: torch.Tensor):
        """
        #NUŽNA METODA#
        Forward pass s dva ulaza.

        Args:
            * x1, torch.Tensor, prvi ulaz
            * x2, torch.Tensor, drugi ulaz (broadcast-kompatibilan s x1)

        Returns:
            (y, z1, z2, z3, h1, h2, h3, w1, w2, w3)
        """
        _x1 = torch.as_tensor(x1, device=self.theta10.device, dtype=self.theta10.dtype)
        _x2 = torch.as_tensor(x2, device=self.theta10.device, dtype=self.theta10.dtype)

        # Broadcast (ako je potrebno)
        _x1, _x2 = torch.broadcast_tensors(_x1, _x2)

        # tri afine grane
        _z1 = self.theta10 + self.theta11 * _x1 + self.theta12 * _x2
        _z2 = self.theta20 + self.theta21 * _x1 + self.theta22 * _x2
        _z3 = self.theta30 + self.theta31 * _x1 + self.theta32 * _x2

        # aktivacije
        _h1 = self.activation1(_z1)
        _h2 = self.activation2(_z2)
        _h3 = self.activation3(_z3)

        # ponderiranja
        _w1 = self.psi1 * _h1
        _w2 = self.psi2 * _h2
        _w3 = self.psi3 * _h3

        # izlaz
        _y = self.psi0 + _w1 + _w2 + _w3

        return _y, _z1, _z2, _z3, _h1, _h2, _h3, _w1, _w2, _w3


class model_1_3_2_for_plots(nn.Module):
    """
    Jednostavan model s 1 ulaznim neuronom, 3 skrivena neurona i 2 izlaza.

    Zajednički skriveni sloj:
      z1 = t10 + t11 * x
      z2 = t20 + t21 * x
      z3 = t30 + t31 * x
      h_i = a_i(z_i)

    Dva izlaza (različiti psi koeficijenti):
      y1 = psi0a + psi1a*h1 + psi2a*h2 + psi3a*h3
      y2 = psi0b + psi1b*h1 + psi2b*h2 + psi3b*h3
    """
    def __init__(self,
                 theta_init_matrix: np.ndarray = None,   
                 psi_init_matrix: np.ndarray = None,                                        
                 activation_function: str = None):
        super().__init__()
        if theta_init_matrix is None:
            theta_init_matrix = np.zeros((3, 2), dtype=np.float32)
        if psi_init_matrix is None:
            psi_init_matrix = np.zeros((2, 4), dtype=np.float32)

        assert theta_init_matrix.shape == (3, 2), "theta_init_matrix must be (3,2)"
        assert psi_init_matrix.shape == (2, 4), "psi_init_matrix must be (2,4)"

        # Theta
        self.theta10 = nn.Parameter(torch.tensor(theta_init_matrix[0, 0], dtype=torch.float32))
        self.theta11 = nn.Parameter(torch.tensor(theta_init_matrix[0, 1], dtype=torch.float32))
        self.theta20 = nn.Parameter(torch.tensor(theta_init_matrix[1, 0], dtype=torch.float32))
        self.theta21 = nn.Parameter(torch.tensor(theta_init_matrix[1, 1], dtype=torch.float32))
        self.theta30 = nn.Parameter(torch.tensor(theta_init_matrix[2, 0], dtype=torch.float32))
        self.theta31 = nn.Parameter(torch.tensor(theta_init_matrix[2, 1], dtype=torch.float32))

        # Psi for y1 (suffix a)
        self.psi0a = nn.Parameter(torch.tensor(psi_init_matrix[0, 0], dtype=torch.float32))
        self.psi1a = nn.Parameter(torch.tensor(psi_init_matrix[0, 1], dtype=torch.float32))
        self.psi2a = nn.Parameter(torch.tensor(psi_init_matrix[0, 2], dtype=torch.float32))
        self.psi3a = nn.Parameter(torch.tensor(psi_init_matrix[0, 3], dtype=torch.float32))
        # Psi for y2 (suffix b)
        self.psi0b = nn.Parameter(torch.tensor(psi_init_matrix[1, 0], dtype=torch.float32))
        self.psi1b = nn.Parameter(torch.tensor(psi_init_matrix[1, 1], dtype=torch.float32))
        self.psi2b = nn.Parameter(torch.tensor(psi_init_matrix[1, 2], dtype=torch.float32))
        self.psi3b = nn.Parameter(torch.tensor(psi_init_matrix[1, 3], dtype=torch.float32))

        # Activation
        _act = nn.Identity
        if activation_function is not None:
            _af = activation_function.lower()
            if _af == "relu":    _act = nn.ReLU
            if _af == "tanh":    _act = nn.Tanh
            if _af == "sigmoid": _act = nn.Sigmoid
        self.activation1 = _act()
        self.activation2 = _act()
        self.activation3 = _act()

    def forward(self, x: torch.Tensor):
        """
        Returns:
          (y1, y2, z1, z2, z3, h1, h2, h3, w1a, w2a, w3a)
        """
        _x = torch.as_tensor(x, device=self.theta10.device, dtype=self.theta10.dtype)

        # affine
        _z1 = self.theta10 + self.theta11 * _x
        _z2 = self.theta20 + self.theta21 * _x
        _z3 = self.theta30 + self.theta31 * _x

        # activations
        _h1 = self.activation1(_z1)
        _h2 = self.activation2(_z2)
        _h3 = self.activation3(_z3)

        # weighted activations for y1 (a)
        _w1a = self.psi1a * _h1
        _w2a = self.psi2a * _h2
        _w3a = self.psi3a * _h3

        # outputs
        _y1 = self.psi0a + _w1a + _w2a + _w3a
        _y2 = self.psi0b + self.psi1b * _h1 + self.psi2b * _h2 + self.psi3b * _h3

        return _y1, _y2, _z1, _z2, _z3, _h1, _h2, _h3, _w1a, _w2a, _w3a