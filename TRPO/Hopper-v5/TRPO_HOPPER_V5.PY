import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
import matplotlib.pyplot as plt
import csv
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RunningMeanFilter:
    def __init__(self, shape, epsilon=1e-4):
        self.mean = np.zeros(shape, dtype=np.float32)
        self.var = np.ones(shape, dtype=np.float32)
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.mean, self.var, self.count = self._update_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count
        )

    def normalize(self, x):
        return (x - self.mean) / (np.sqrt(self.var) + 1e-8)

    @staticmethod
    def _update_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
        delta = batch_mean - mean
        new_count = count + batch_count
        new_mean = mean + delta * batch_count / new_count
        m_a = var * count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + delta**2 * count * batch_count / new_count
        new_var = m_2 / new_count
        return new_mean, new_var, new_count


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.policy_mean = nn.Sequential(
            nn.Linear(state_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, action_dim)
        )
        self.policy_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state):
        mean = self.policy_mean(state)
        std = torch.exp(self.policy_std)
        return mean, std


class ValueNetwork(nn.Module):
    def __init__(self, state_dim):
        super(ValueNetwork, self).__init__()
        self.value = nn.Sequential(
            nn.Linear(state_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, state):
        return self.value(state)


class TRPOAgent:
    def __init__(self, state_dim, action_dim, params):
        self.gamma = params["gamma"]
        self.gae_lambda = params["gae_lambda"]
        self.max_kl = params["max_kl"]
        self.cg_iters = params["cg_iters"]
        self.cg_damping = params["cg_damping"]
        self.vf_iters = params["vf_iters"]
        self.vf_batch_size = params["vf_batch_size"]
        self.vf_step_size = params["vf_step_size"]
        self.entropy_coeff = params["entropy_coeff"]

        self.policy_net = PolicyNetwork(state_dim, action_dim).to(device)
        self.value_net = ValueNetwork(state_dim).to(device)
        self.obs_filter = RunningMeanFilter(state_dim)
        self.buffer = []

    def store_transition(self, transition):
        self.buffer.append(transition)

    def compute_gae(self, rewards, values, dones):
        advantages, returns = [], []
        gae = 0
        next_value = 0
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * next_value * (1 - dones[step]) - values[step]
            gae = delta + self.gamma * self.gae_lambda * gae * (1 - dones[step])
            advantages.insert(0, gae)
            next_value = values[step]
            returns.insert(0, gae + values[step])
        return advantages, returns

    def train(self, states, actions, rewards, dones, old_log_probs, values):
        # Compute GAE and returns
        advantages, returns = self.compute_gae(rewards.cpu().numpy(), values.cpu().numpy(), dones.cpu().numpy())
        advantages = torch.FloatTensor(advantages).to(device).detach()
        returns = torch.FloatTensor(returns).to(device).detach()

        # Train value function
        optimizer = torch.optim.Adam(self.value_net.parameters(), lr=self.vf_step_size)
        for _ in range(self.vf_iters):
            perm = np.random.permutation(states.size(0))
            for i in range(0, states.size(0), self.vf_batch_size):
                idx = perm[i:i + self.vf_batch_size]
                batch_states = states[idx]
                batch_returns = returns[idx]
                value_loss = nn.MSELoss()(self.value_net(batch_states).squeeze(), batch_returns)
                optimizer.zero_grad()
                value_loss.backward()
                optimizer.step()


def train_trpo(env, agent, max_steps, log_interval, output_csv="trpo_results.csv"):
    total_steps = 0
    timesteps = []
    avg_returns = []
    std_errors = []
    episode_rewards = []

    # Create output directory if necessary
    if os.path.dirname(output_csv):
       os.makedirs(os.path.dirname(output_csv), exist_ok=True)


    # Initialize CSV file
    with open(output_csv, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Timesteps", "Avg Return", "Std Error"])

    while total_steps < max_steps:
        state, info = env.reset()
        state = agent.obs_filter.normalize(state)
        state = torch.FloatTensor(state).to(device)
        episode_reward = 0
        episode_transitions = []

        while True:
            with torch.no_grad():
                mean, std = agent.policy_net(state)
                dist = Normal(mean, std)
                action = dist.sample()
                action = torch.clamp(action, env.action_space.low[0], env.action_space.high[0])
                action_log_prob = dist.log_prob(action).sum(dim=-1)
            next_state, reward, done, truncated, info = env.step(action.cpu().numpy())
            next_state = agent.obs_filter.normalize(next_state)
            next_state = torch.FloatTensor(next_state).to(device)
            value = agent.value_net(state).item()

            # Store transition
            episode_transitions.append((state.cpu().numpy(), action.cpu().numpy(), reward, done, action_log_prob.cpu().item(), value))
            state = next_state
            episode_reward += reward
            total_steps += 1

            if done or truncated:
                episode_rewards.append(episode_reward)
                agent.store_transition(episode_transitions)
                break

        # Train the agent and log progress
        if total_steps % log_interval == 0:
            states, actions, rewards, dones, old_log_probs, values = zip(*[t for episode in agent.buffer for t in episode])
            states = torch.FloatTensor(np.array(states)).to(device)
            actions = torch.FloatTensor(np.array(actions)).to(device)
            rewards = torch.FloatTensor(np.array(rewards)).to(device)
            dones = torch.FloatTensor(np.array(dones)).to(device)
            old_log_probs = torch.FloatTensor(np.array(old_log_probs)).to(device)
            values = torch.FloatTensor(np.array(values)).to(device)

            agent.train(states, actions, rewards, dones, old_log_probs, values)
            agent.buffer = []

            avg_return = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
            std_error = np.std(episode_rewards[-100:]) / np.sqrt(len(episode_rewards[-100:])) if len(episode_rewards) >= 100 else 0
            timesteps.append(total_steps)
            avg_returns.append(avg_return)
            std_errors.append(std_error)

            print(f"Steps: {total_steps}, Avg Return: {avg_return:.2f}, Std Error: {std_error:.2f}")

            # Save results to CSV
            with open(output_csv, "a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([total_steps, avg_return, std_error])

    # Plot and save the results
    plt.errorbar(timesteps, avg_returns, yerr=std_errors, fmt="-o", ecolor="r", capsize=5, label="Avg Return")
    plt.xlabel("Timesteps")
    plt.ylabel("Average Return")
    plt.title("TRPO Training Performance")
    plt.legend()
    plt.savefig("trpo_training__Hopper_plot.png")
    plt.show()


if __name__ == "__main__":
    MAX_STEPS = 1_000_000
    LOG_INTERVAL = 1000
    PARAMS = {
        "gamma": 0.995,
        "gae_lambda": 0.9,
        "max_kl": 0.01,
        "cg_iters": 20,
        "cg_damping": 0.1,
        "vf_iters": 5,
        "vf_batch_size": 64,
        "vf_step_size": 1e-3,
        "entropy_coeff": 0.0,
    }

    env = gym.make("Hopper-v5")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent = TRPOAgent(state_dim, action_dim, PARAMS)
    train_trpo(env, agent, MAX_STEPS, LOG_INTERVAL, output_csv="trpo_Hopper_64_tanh_results.csv")
