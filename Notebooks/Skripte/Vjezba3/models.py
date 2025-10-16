"""
Skripta u kojoj se implementiraju modeli za vježbu 3
"""

# Knjižnice
import torch
from torch import nn
import torchinfo
from torchview import draw_graph
from IPython.display import display, SVG
import numpy as np

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
        input_size=[input_dim],   # e.g. (1, 3, 224, 224)
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
   
class model_1_1_1(nn.Module):
    """
    #NUŽNA METODA#
    Jednostavan model koji implementira pravac.
    Dakle 1 ulazna podatak, dva parametra, jedan izlazan:

    y = theta1 * x + theta0
    """
    def __init__(self, 
                 theta0_init: float = 0.0, 
                 theta1_init: float = 1.0,
                 activation_function: str = None):
        """
        Init metoda u kojoj defirinamo slojeve neuronske mreže i po potrebi dodatne parametre

        Args:
            * theta0_init, float, inicijalna vrijednost težine theta0
            * theta1_init, float, inicijalna vrijednost težine theta1,
            * activation_function, str, vrsta aktivacijske funckije. Može biti
                - tanh: tangens hiperbolni
                - relu: rectified linear unit
                - sigmoid: za sigmoid funkciju.
        """
        super().__init__()
        # Set parameters
        self.theta0 = nn.Parameter(torch.tensor(theta0_init, dtype=torch.float32))
        self.theta1 = nn.Parameter(torch.tensor(theta1_init, dtype=torch.float32))

        # Set activation function
        self.activation = None
        if activation_function is not None:
            _activation_function = activation_function.lower()
            if _activation_function == "relu": self.activation = nn.ReLU()
            if _activation_function == "tanh": self.activation = nn.Tanh()
            if _activation_function == "sigmoid": self.activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor)->torch.Tensor:
        """
        #NUŽNA METODA#
        Forward pass. Služi učenju neuronskih mreža.

        Args:
            * x, torch.Tensor, input data
        
        Returns:
            * rezultat pravca: x * theta0 + theta1
        """ 
        _x = torch.as_tensor(x, device=self.theta0.device, dtype=self.theta0.dtype)
        _b0 = self.theta0 
        _b1 = self.theta1 
        _z1 = _b1 * _x + _b0

        # Apply activation function
        if self.activation is not None:
             _z1 = self.activation(_z1)

        return  _z1
    
class model_1_3_1_for_plots(nn.Module):
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

        return _y, _z1, _z2, _z3, _h1, _h2, _h3, _w1, _w2, _w3
    
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
        _z1 = 0
        _z2 = 0
        _z3 = 0

        # activations
        _h1 = 0
        _h2 = 0
        _h3 = 0
        
        # Multiplications
        _w1 = 0
        _w2 = 0
        _w3 = 0

        # linear combination with psi's
        _y = 0
        return _y