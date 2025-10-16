import torch
import torch.nn as nn

class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        # Single linear layer: y = ax + b
        self.linear = nn.Linear(in_features=1, out_features=1)

    def forward(self, x):
        return self.linear(x)

# Print model parameters (a and b)
for name, param in model.named_parameters():
    print(f"{name}: {param.data}")