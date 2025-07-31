"""
Policy network wrapping the ESN with softmax output.
"""

import torch.nn as nn
from esn import EchoStateNetwork

class PolicyNetwork(nn.Module):
    """
    Policy network with Echo State Network backbone.
    """
    def __init__(self, esn: EchoStateNetwork, action_dim: int):
        super().__init__()
        self.esn = esn
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.esn(x)
        return self.softmax(logits)
