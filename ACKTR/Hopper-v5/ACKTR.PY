import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
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
            nn.Linear(state_dim, 400), nn.Tanh(),
            nn.Linear(400, 300), nn.Tanh(),
            nn.Linear(300, action_dim)
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
            nn.Linear(state_dim, 400), nn.ELU(),
            nn.Linear(400, 300), nn.ELU(),
            nn.Linear(300, 1)
        )

    def forward(self, state):
        return self.value(state)

class ACKTRAgent:
    def __init__(self, state_dim, action_dim, lr, gamma, gae_lambda, kl_target):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.kl_target = kl_target
        self.policy_net = PolicyNetwork(state_dim, action_dim).to(device)
        self.value_net = ValueNetwork(state_dim).to(device)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=lr)
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

    def train(self, batch_size):
        states, actions, rewards, dones, old_log_probs, values = zip(*self.buffer)
        self.buffer = []
        states = torch.FloatTensor(np.array(states)).to(device)
        actions = torch.FloatTensor(np.array(actions)).to(device)
        rewards = torch.FloatTensor(np.array(rewards)).to(device)
        dones = torch.FloatTensor(np.array(dones)).to(device)
        old_log_probs = torch.FloatTensor(np.array(old_log_probs)).to(device)
        values = torch.FloatTensor(np.array(values)).to(device)
        advantages, returns = self.compute_gae(rewards.cpu().numpy(), values.cpu().numpy(), dones.cpu().numpy())
        advantages = torch.FloatTensor(advantages).to(device).detach()
        returns = torch.FloatTensor(returns).to(device).detach()

        for _ in range(batch_size):
            mean, std = self.policy_net(states)
            dist = Normal(mean, std)
            new_log_probs = dist.log_prob(actions).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)
            ratios = torch.exp(new_log_probs - old_log_probs)
            kl_divergence = (old_log_probs - new_log_probs).mean()
            if kl_divergence > self.kl_target:
                break
            policy_loss = -(ratios * advantages).mean()
            values_pred = self.value_net(states).squeeze()
            value_loss = nn.MSELoss()(values_pred, returns)
            self.policy_optimizer.zero_grad()
            policy_loss.backward(retain_graph=True)
            self.policy_optimizer.step()
            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()

def train_acktr(env, agent, max_steps, log_interval, output_csv="acktr_results.csv"):
    total_steps = 0
    timesteps = []
    avg_returns = []
    std_errors = []
    episode_rewards = []

    output_dir = os.path.dirname(output_csv)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(output_csv, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Timesteps", "Avg Return", "Std Error"])

    while total_steps < max_steps:
        state, info = env.reset()
        state = agent.obs_filter.normalize(state)
        state = torch.FloatTensor(state).to(device)
        episode_reward = 0
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
            agent.store_transition((state.cpu().numpy(),
                                    action.cpu().numpy(),
                                    reward, done,
                                    action_log_prob.detach().cpu().numpy(),
                                    agent.value_net(state).item()))
            state = next_state
            episode_reward += reward
            total_steps += 1
            if done or truncated:
                episode_rewards.append(episode_reward)
                break
        if total_steps % log_interval == 0:
            agent.train(batch_size=2500)
            avg_return = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
            std_error = np.std(episode_rewards[-100:]) / np.sqrt(len(episode_rewards[-100:])) if len(episode_rewards) >= 100 else 0
            timesteps.append(total_steps)
            avg_returns.append(avg_return)
            std_errors.append(std_error)
            print(f"Steps: {total_steps}, Avg Return: {avg_return:.2f}, Std Error: {std_error:.2f}")
            with open(output_csv, mode="a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([total_steps, avg_return, std_error])

    return timesteps, avg_returns, std_errors

if __name__ == "__main__":
    MAX_STEPS = 1_000_000
    LR = 5e-4
    GAMMA = 0.995
    GAE_LAMBDA = 0.97
    KL_TARGET = 0.002
    LOG_INTERVAL = 2000
    OUTPUT_CSV = "acktr_results_400_300.csv"

    env = gym.make("Hopper-v5")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    agent = ACKTRAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=LR,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        kl_target=KL_TARGET,
    )
    timesteps, avg_returns, std_errors = train_acktr(env, agent, MAX_STEPS, LOG_INTERVAL, OUTPUT_CSV)

    plt.errorbar(timesteps, avg_returns, yerr=std_errors, fmt="-o", ecolor="r", capsize=5, label="Avg Return")
    plt.xlabel("Timesteps")
    plt.ylabel("Average Return")
    plt.title("ACKTR Training Performance on Hopper-v5")
    plt.legend()
    plt.savefig("acktr_hopper_training_plot_400_300.png")
    plt.show()
