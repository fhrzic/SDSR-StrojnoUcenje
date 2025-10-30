# Knjižnice
import torch
from torch import nn

class model_1_3_1(nn.Module):
    """
    1 → 3 → 1 shallow model with hardcoded thetas and (psi1..psi3).
    The ONLY configurable value is psi0 provided in __init__.
    """
    def __init__(self, psi0: float = 0.202, 
                 device: str = "cpu"):
        """
        Init metoda koja gradi model.

        Args:
            * psi0, float, vrijednost biasa ispred ulaznog neurona
            * device, str, ime urđaja na kojem egzistira neuronska mreža
        """
        super().__init__()

        # ====== EDIT THESE CONSTANTS IF NEEDED ======
        # thetas: z_k = theta_k0 + theta_k1 * x  (k=1..3)
        theta10, theta11 = 0.2,  -1.3
        theta20, theta21 = -0.25, 1.1
        theta30, theta31 = -1.6, 2.1

        # fixed output weights (psi1..psi3)
        psi1, psi2, psi3 = -1, 1.1, -0.65
        # ===========================================

        # store as buffers (non-trainable, move with .to(device))
        self.register_buffer("theta10", torch.tensor(theta10, dtype=torch.float32))
        self.register_buffer("theta11", torch.tensor(theta11, dtype=torch.float32))
        self.register_buffer("theta20", torch.tensor(theta20, dtype=torch.float32))
        self.register_buffer("theta21", torch.tensor(theta21, dtype=torch.float32))
        self.register_buffer("theta30", torch.tensor(theta30, dtype=torch.float32))
        self.register_buffer("theta31", torch.tensor(theta31, dtype=torch.float32))

        self.register_buffer("psi1", torch.tensor(psi1, dtype=torch.float32))
        self.register_buffer("psi2", torch.tensor(psi2, dtype=torch.float32))
        self.register_buffer("psi3", torch.tensor(psi3, dtype=torch.float32))

        # only free knob
        self.register_buffer("psi0", torch.tensor(float(psi0), dtype=torch.float32))

        # activation
        self.activation = nn.ReLU()

        # device move
        self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            * x, torch.Tensor, ulaz u neuronsku mrežu

        Returns:
            * torch.Tensor, izlaz iz neuronske mreže
        """
        x = torch.as_tensor(x, dtype=torch.float32, device=self.psi0.device)

        z1 = self.theta10 + self.theta11 * x
        z2 = self.theta20 + self.theta21 * x
        z3 = self.theta30 + self.theta31 * x

        h1 = self.activation(z1)
        h2 = self.activation(z2)
        h3 = self.activation(z3)

        y = self.psi0 + self.psi1 * h1 + self.psi2 * h2 + self.psi3 * h3

        return y
    

class model_1_3_1_b(nn.Module):
    """
    1 → 3 → 1 shallow model with hardcoded thetas and (psi1..psi3).
    The ONLY configurable value is psi0 provided in __init__.
    """
    def __init__(self, psi0: float = 0.202, 
                 device: str = "cpu", 
                 apply_sigmoid: bool = False):
        """
        Init metoda koja gradi model.

        Args:
            * psi0, float, vrijednost biasa ispred ulaznog neurona
            * device, str, ime urđaja na kojem egzistira neuronska mreža
            * apply_sigmoid, bool, ako je apliciran, izlaz iz neuronske mreže prolazi
            kroz sigmoid
        """
        super().__init__()

        # ====== EDIT THESE CONSTANTS IF NEEDED ======
        # thetas: z_k = theta_k0 + theta_k1 * x  (k=1..3)
        theta10, theta11 = 4,  -9
        theta20, theta21 = -3.5, 7
        theta30, theta31 = -2.5, 3

        # fixed output weights (psi1..psi3)
        psi1, psi2, psi3 = -1, -0.1, 11
        # ===========================================

        # store as buffers (non-trainable, move with .to(device))
        self.register_buffer("theta10", torch.tensor(theta10, dtype=torch.float32))
        self.register_buffer("theta11", torch.tensor(theta11, dtype=torch.float32))
        self.register_buffer("theta20", torch.tensor(theta20, dtype=torch.float32))
        self.register_buffer("theta21", torch.tensor(theta21, dtype=torch.float32))
        self.register_buffer("theta30", torch.tensor(theta30, dtype=torch.float32))
        self.register_buffer("theta31", torch.tensor(theta31, dtype=torch.float32))

        self.register_buffer("psi1", torch.tensor(psi1, dtype=torch.float32))
        self.register_buffer("psi2", torch.tensor(psi2, dtype=torch.float32))
        self.register_buffer("psi3", torch.tensor(psi3, dtype=torch.float32))

        # only free knob
        self.register_buffer("psi0", torch.tensor(float(psi0), dtype=torch.float32))

        # activation
        self.activation = nn.ReLU()

        # Last activation
        if apply_sigmoid:
            self.final_activation = nn.Sigmoid()
        else:
            self.final_activation = nn.Identity()
        # device move
        self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            * x, torch.Tensor, ulaz u neuronsku mrežu

        Returns:
            * torch.Tensor, izlaz iz neuronske mreže
        """
        x = torch.as_tensor(x, dtype=torch.float32, device=self.psi0.device)

        z1 = self.theta10 + self.theta11 * x
        z2 = self.theta20 + self.theta21 * x
        z3 = self.theta30 + self.theta31 * x

        h1 = self.activation(z1)
        h2 = self.activation(z2)
        h3 = self.activation(z3)

        y = self.psi0 + self.psi1 * h1 + self.psi2 * h2 + self.psi3 * h3

        y = self.final_activation(y)

        return y