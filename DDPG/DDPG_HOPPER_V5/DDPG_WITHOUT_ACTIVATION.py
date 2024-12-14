import argparse
import os
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import matplotlib.pyplot as plt
from scipy.stats import sem
import csv
os.environ["OMP_NUM_THREADS"] = "1"

# Set the device to CPU or GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Replay buffer for experience replay
class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer = deque(maxlen=buffer_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def size(self):
        return len(self.buffer)

# Neural network for policy (actor) without activation functions
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_sizes):
        super(Actor, self).__init__()
        layers = []
        input_dim = state_dim
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_dim, hidden_size))
            input_dim = hidden_size
        layers.append(nn.Linear(input_dim, action_dim))
        layers.append(nn.Tanh())  # Ensure actions are within range [-1, 1]
        self.network = nn.Sequential(*layers)

    def forward(self, state):
        return self.network(state)

# Neural network for Q-function (critic) without activation functions
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_sizes):
        super(Critic, self).__init__()
        layers = []
        input_dim = state_dim + action_dim
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(input_dim, hidden_size))
            input_dim = hidden_size
        layers.append(nn.Linear(input_dim, 1))  # Single Q-value output
        self.network = nn.Sequential(*layers)

    def forward(self, state, action):
        return self.network(torch.cat([state, action], dim=-1))

# Ornstein-Uhlenbeck Noise for exploration
class OUNoise:
    def __init__(self, action_dim, mu=0.0, theta=0.15, sigma=0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def sample(self):
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.action_dim)
        self.state += dx
        return self.state

# DDPG Agent
class DDPGAgent:
    def __init__(self, state_dim, action_dim, actor_hidden_sizes, critic_hidden_sizes, 
                 actor_lr, critic_lr, gamma, tau, buffer_size, batch_size):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

        # Actor networks
        self.actor = Actor(state_dim, action_dim, actor_hidden_sizes).to(device)
        self.target_actor = Actor(state_dim, action_dim, actor_hidden_sizes).to(device)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)

        # Critic networks
        self.critic = Critic(state_dim, action_dim, critic_hidden_sizes).to(device)
        self.target_critic = Critic(state_dim, action_dim, critic_hidden_sizes).to(device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)

        # Noise for exploration
        self.noise = OUNoise(action_dim)

    def select_action(self, state, noise=True):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        action = self.actor(state).detach().cpu().numpy()[0]
        if noise:
            action += self.noise.sample()
        return np.clip(action, -1.0, 1.0)

    def train(self):
        if self.replay_buffer.size() < self.batch_size:
            return

        # Sample a batch from the replay buffer
        batch = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states).to(device)
        actions = torch.FloatTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).unsqueeze(-1).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).unsqueeze(-1).to(device)

        # Update Critic
        with torch.no_grad():
            target_actions = self.target_actor(next_states)
            target_q_values = self.target_critic(next_states, target_actions)
            q_targets = rewards + self.gamma * target_q_values * (1 - dones)
        q_values = self.critic(states, actions)
        critic_loss = nn.MSELoss()(q_values, q_targets)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update Actor
        actor_loss = -self.critic(states, self.actor(states)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update target networks
        self.soft_update(self.actor, self.target_actor)
        self.soft_update(self.critic, self.target_critic)

    def soft_update(self, source, target):
        for source_param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(self.tau * source_param.data + (1.0 - self.tau) * target_param.data)

# Function to plot average return vs time steps
def plot_average_return_vs_timesteps(timesteps, avg_returns, filename="return_vs_timesteps.png"):
    plt.figure(figsize=(10, 6))
    plt.plot(timesteps, avg_returns, label="Average Return")
    plt.xlabel("Time Steps")
    plt.ylabel("Average Return")
    plt.title("Average Return vs. Time Steps")
    plt.grid()
    plt.legend()
    plt.savefig(filename)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_steps", type=int, default=1000000, help="Maximum training steps")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--buffer_size", type=int, default=100000)
    parser.add_argument("--actor_lr", type=float, default=1e-4)
    parser.add_argument("--critic_lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--actor_hidden_sizes", type=int, nargs="+", default=[400, 300])
    parser.add_argument("--critic_hidden_sizes", type=int, nargs="+", default=[400, 300])
    parser.add_argument("--log_interval", type=int, default=1000)
    args = parser.parse_args()

    env = gym.make("HalfCheetah-v4")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = DDPGAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        actor_hidden_sizes=args.actor_hidden_sizes,
        critic_hidden_sizes=args.critic_hidden_sizes,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        gamma=args.gamma,
        tau=args.tau,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
    )

    total_steps = 0
    timesteps = []
    avg_returns = []
    std_errors = []
    episode_rewards = []

    results_file = "results.csv"
    with open(results_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Time Steps", "Average Return", "Standard Error"])

        while total_steps < args.max_steps:
            state, info = env.reset()
            agent.noise.reset()
            episode_reward = 0

            while True:
                action = agent.select_action(state)
                next_state, reward, done, truncated, info = env.step(action)
                agent.replay_buffer.add((state, action, reward, next_state, done))
                agent.train()

                state = next_state
                episode_reward += reward
                total_steps += 1

                if done or truncated:
                    episode_rewards.append(episode_reward)
                    break

            if total_steps % args.log_interval == 0:
                avg_return = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
                std_error = sem(episode_rewards[-100:]) if len(episode_rewards) >= 100 else sem(episode_rewards)
                timesteps.append(total_steps)
                avg_returns.append(avg_return)
                std_errors.append(std_error)
                writer.writerow([total_steps, avg_return, std_error])
                print(f"Steps: {total_steps}, Average Return: {avg_return:.2f}, Standard Error: {std_error:.2f}")

    plot_average_return_vs_timesteps(timesteps, avg_returns)
