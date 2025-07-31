"""
Training script for ESN-based policy gradient on CartPole-v1.
"""

import gym
import torch
from torch.optim import Adam
from policy import PolicyNetwork
from esn import EchoStateNetwork
from utils import set_seed, reset_env

def train(env_name: str = 'CartPole-v1',
          seed: int = 1234,
          reservoir_size: int = 500,
          lr: float = 1e-2,
          gamma: float = 0.99,
          num_samples: int = 50,
          episodes: int = 500) -> None:
    """
    Train the policy network using REINFORCE with Bayesian model averaging.
    """
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create the environment
    try:
        env = gym.make(env_name, new_step_api=True)
    except TypeError:
        env = gym.make(env_name)

    obs = reset_env(env, seed)
    input_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    esn = EchoStateNetwork(input_dim, reservoir_size).to(device)
    policy = PolicyNetwork(esn, action_dim).to(device)
    optimizer = Adam(policy.parameters(), lr=lr)

    for episode in range(1, episodes + 1):
        state = torch.tensor(obs, dtype=torch.float32).to(device)
        rewards, log_probs = [], []
        done = False

        while not done:
            # Bayesian averaging
            samples = [policy(state) for _ in range(num_samples)]
            action_probs = torch.stack(samples).mean(0)
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
            log_probs.append(dist.log_prob(action))

            out = env.step(action.item())
            if len(out) == 5:
                obs, reward, terminated, truncated, _ = out
                done = terminated or truncated
            else:
                obs, reward, done, _ = out

            state = torch.tensor(obs, dtype=torch.float32).to(device)
            rewards.append(reward)

        # Compute discounted returns
        returns = []
        R = 0.0
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32).to(device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Policy gradient update
        loss = -torch.stack([lp * R for lp, R in zip(log_probs, returns)]).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if episode % 10 == 0:
            print(f'Episode {episode}, Total Reward: {sum(rewards)}')

    env.close()

if __name__ == '__main__':
    train()
