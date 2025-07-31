"""
Echo State Network (ESN) implementation with Bayesian readout.
"""

import torch
import torch.nn as nn

class BayesianReadout(nn.Module):
    """
    Bayesian readout layer with dropout for uncertainty estimation.
    """
    def __init__(self, input_dim: int, output_dim: int, dropout_rate: float = 0.2):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc(x))

class EchoStateNetwork(nn.Module):
    """
    PyTorch module for an Echo State Network (ESN).
    """
    def __init__(self, input_dim: int, reservoir_size: int,
                 spectral_radius: float = 0.9, sparsity: float = 0.5,
                 leaky_rate: float = 0.2):
        super().__init__()
        self.reservoir_size = reservoir_size
        self.spectral_radius = spectral_radius
        self.leaky_rate = leaky_rate

        # Input-to-reservoir weights
        self.W_in = (torch.rand(reservoir_size, input_dim) - 0.5) * 2 / input_dim

        # Reservoir weights with sparsity
        W = torch.rand(reservoir_size, reservoir_size) - 0.5
        mask = torch.rand(reservoir_size, reservoir_size) > sparsity
        W[mask] = 0

        # Scale to desired spectral radius
        eigvec = torch.rand(reservoir_size, 1)
        for _ in range(50):
            eigvec = W @ eigvec
            eigvec /= eigvec.norm()
        max_eig = eigvec.norm()
        self.W = W * (spectral_radius / max_eig)

        # Internal state
        self.register_buffer('state', torch.zeros(reservoir_size))

        # Readout layer
        self.readout = BayesianReadout(reservoir_size, 2, dropout_rate=0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Update reservoir state and compute output logits.
        """
        x = x.view(-1)
        self.state = self.state.to(x.device)

        # Reservoir update
        pre_act = self.W_in @ x + self.W @ self.state
        self.state = (1 - self.leaky_rate) * self.state + self.leaky_rate * torch.tanh(pre_act)

        # Normalize
        norm = self.state.norm().clamp(min=1e-6)
        self.state = self.state / norm

        return self.readout(self.state)
