import torch
import torch.nn as nn

class MLP_1_3_3_1(nn.Module):
    """
    Simple MLP network:
        Input (1) → Hidden Layer 1 (3) → Hidden Layer 2 (3) → Output (1)
        Activation: ReLU between layers
    """
    def __init__(self):
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Linear(1, 3),     # Input → Hidden1
            nn.ReLU(),
            nn.Linear(3, 3),     # Hidden1 → Hidden2
            nn.ReLU(),
            nn.Linear(3, 1)      # Hidden2 → Output
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)