[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)]()
[![PyPI](https://img.shields.io/pypi/v/your-package-name.svg)]()

# CartPole-ESN: Bayesian Echo State Network Policy Gradient

A high-performance, PyTorch-based implementation of **Echo State Networks** (ESN) combined with **REINFORCE** policy gradient for solving OpenAI Gym’s CartPole-v1. Leveraging Bayesian model averaging for robust action selection and uncertainty quantification.

---

## 🚀 Features

- **Echo State Network backbone**  
  Fast, reservoir-based recurrent computation with spectral radius control and leaky integration.
- **Bayesian Readout**  
  Dropout-based Bayesian averaging (Monte Carlo sampling) for uncertainty-aware policy.
- **Plug-and-play**  
  Modular code split into `utils`, `esn`, `policy`, and `train`—easily extendable to other environments.
- **Reproducibility**  
  Deterministic seeding across NumPy, PyTorch, and CuDNN; compatible with old/new Gym APIs.

---

## 📦 Installation

```bash
git clone https://github.com/frezazadeh/rl-bayesian-ESN.git
cd rl-bayesian-ESN
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## 🏃 Usage

### Basic training
```bash
python train.py   --env CartPole-v1   --seed 42   --episodes 500   --reservoir_size 500   --num_samples 50   --lr 0.01   --gamma 0.99
```

### CLI arguments
| Argument        | Type    | Default     | Description                                         |
| --------------- | ------- | ----------- | --------------------------------------------------- |
| `--env`         | `str`   | `CartPole-v1` | Gym environment name                                |
| `--seed`        | `int`   | `1234`      | Random seed                                         |
| `--episodes`    | `int`   | `500`       | Number of training episodes                         |
| `--reservoir_size` | `int` | `500`     | Echo State reservoir dimension                      |
| `--num_samples` | `int`   | `50`        | Monte Carlo samples per action                      |
| `--lr`          | `float` | `1e-2`      | Learning rate                                       |
| `--gamma`       | `float` | `0.99`      | Discount factor                                      |

---

## 📁 Project Structure

```text
cartpole-esn/
├── esn.py            # Echo State Network + BayesianReadout
├── policy.py         # Softmax policy wrapper
├── utils.py          # Seed & reset helpers
├── train.py          # Training entrypoint & CLI
├── requirements.txt  # Pip dependencies
├── .gitignore        # Exclusions for Git
├── LICENSE           # MIT License
└── README.md         # This document
```

---

## 🔧 Extanding to New Environments

1. **Adjust action/output dimension**  
   Modify `BayesianReadout` output size in `esn.py` or pass `output_dim` via CLI.
2. **Custom observation pre-processing**  
   Insert feature transforms before `policy(state)` in `train.py`.
3. **Hyperparameter search**  
   Integrate with Optuna or Ray Tune in `train.py`’s training loop.

---

## 📜 License

This project is released under the **MIT License**.

