import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import os
import pickle
import csv
import pandas as pd
import matplotlib.pyplot as plt
from ambiance import Atmosphere
import gymnasium as gym
from gymnasium import spaces
from rocket import Rocket

Path = 'train_data/'

class PPORocketAgent(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PPORocketAgent, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.Tanh(),
            nn.Linear(512, 512),
            nn.Tanh(),
            nn.Linear(512, action_dim),
            nn.Sigmoid()
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.Tanh(),
            nn.Linear(512, 512),
            nn.Tanh(),
            nn.Linear(512, 1)
        )
        self.log_std = nn.Parameter(torch.ones(action_dim) * 0.1)

    def forward(self, state):
        return self.actor(state), self.critic(state)

    def get_action(self, state):
        action_mean, _ = self.forward(state)
        action_log_std = self.log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)
        normal = Normal(action_mean, action_std)
        action = normal.sample()
        action_log_prob = normal.log_prob(action).sum(dim=-1)
        return action, action_log_prob

def scale_action(action, low, high):
    action = np.clip(action, 0, 1)
    scaled_action = low + (action) * (high - low)
    return scaled_action

def load_checkpoint(filepath, agent, actor_optimizer, critic_optimizer):
    if os.path.exists(filepath):
        checkpoint = torch.load(filepath)
        agent.load_state_dict(checkpoint['agent_state_dict'])
        actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        episode_rewards = checkpoint['episode_rewards']
        return episode_rewards
    return []

def test_agent(env, agent, num_episodes=10):
    episode_rewards = []
    action_low = np.array([-30] * 10 + [0.4 * env.max_thrust[0]] * 5 + [-30] * 6 + [0.4 * env.max_thrust[1]] * 3)
    action_high = np.array([30] * 10 + [env.max_thrust[0]] * 5 + [30] * 6 + [env.max_thrust[1]] * 3)

    for episode in range(num_episodes):
        state = env.reset()
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        episode_reward = 0
        done = False

        while not done:
            action, action_log_prob = agent.get_action(state)
            scaled_action = scale_action(action.detach().cpu().numpy()[0], action_low, action_high)
            next_state, reward, done, _ = env.step(scaled_action)
            next_state = torch.FloatTensor(next_state).unsqueeze(0).to(device)
            state = next_state


    return []

if __name__=='__main__':
    env = Rocket()  
    state_dim = env.state_dims
    action_dim = env.action_dims
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    agent = PPORocketAgent(state_dim, action_dim).to(device)

    actor_optimizer = optim.Adam([
        {'params': agent.actor.parameters()},
        {'params': agent.log_std}
    ], lr=1e-4)
    critic_optimizer = optim.Adam(agent.critic.parameters(), lr=1e-4)

    checkpoint_path = os.path.join(Path, 'checkpoint.pth')
    episode_rewards = load_checkpoint(checkpoint_path, agent, actor_optimizer, critic_optimizer)

    test_rewards = test_agent(env, agent, num_episodes=10)
    env.show_path_from_state_buffer()
    env.animate_trajectory_noearth(skip_steps=1)