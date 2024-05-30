import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import os
import pickle
import csv
from collections import deque
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
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
            nn.LeakyReLU(),
            nn.Linear(512, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 512),
            nn.LeakyReLU(),
            nn.Linear(512, action_dim)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 512),
            nn.LeakyReLU(),
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
    # Ensure action is within [-1, 1]
    action = np.clip(action, -1, 1)
    scaled_action = low + (0.5 * (action + 1.0) * (high - low))
    return scaled_action

def normalize_state(state):
    return (state - state.mean()) / (state.std() + 1e-8)

def save_data_to_csv(data, filename):
    header = [
        'epi_timestep', 'action0_5', 'action5_10', 'action10_15',
        'state0_3', 'state3_6', 'state6_9', 'state9_12', 'state12', 'state13'
    ]
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        if file.tell() == 0:  # Check if file is empty to write header
            writer.writerow(header)
        for row in data:
            row = [item.tolist() if torch.is_tensor(item) else item for item in row]
            writer.writerow(row)

def save_pickle(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def load_pickle(filename):
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
    return None

def load_data_from_csv(filename):
    data = []
    if os.path.exists(filename):
        with open(filename, mode='r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip header
            data = [row for row in reader]
    return data
def train_ppo_agent(env, agent, num_episodes=1000, actor_learning_rate=1e-4, critic_learning_rate=1e-4, gamma=0.99, clip_epsilon=0.09, update_timestep=2000, k_epochs=4, save_path='train_data/', early_stopping_threshold=295):
    actor_optimizer = optim.Adam([
        {'params': agent.actor.parameters()},
        {'params': agent.log_std, 'lr': actor_learning_rate}
    ], lr=actor_learning_rate)
    critic_optimizer = optim.Adam(agent.critic.parameters(), lr=critic_learning_rate)
    mse_loss = nn.MSELoss()
    episode_rewards = load_pickle(os.path.join(save_path, 'episode_rewards.pkl')) or []
    data_to_save = load_data_from_csv(os.path.join(save_path, 'training_data.csv')) or []  # 데이터를 저장할 리스트
    losses = []

    action_low = np.array([-30] * 10 + [0.4 * env.max_thrust[0]] * 5 + [-30] * 6 + [0.4 * env.max_thrust[1]] * 3)
    action_high = np.array([30] * 10 + [env.max_thrust[0]] * 5 + [30] * 6 + [env.max_thrust[1]] * 3)

    model_path = os.path.join(save_path, 'ppo_rocket_model.pth')
    csv_path = os.path.join(save_path, 'training_data.csv')

    timestep = 0
    # 모델이 존재하면 로드
    if os.path.exists(model_path):
        agent.load_state_dict(torch.load(model_path))
        
    for episode in range(num_episodes):
        state = env.reset()
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        episode_reward = 0
        episode_data = []
        epi_timestep = 0

        for t in range(env.max_step):
            timestep += 1
            epi_timestep += 1
            temp_state = state
            state = normalize_state(state)
            action, action_log_prob = agent.get_action(state)
            scaled_action = scale_action(action.detach().cpu().numpy()[0], action_low, action_high)

            next_state, reward, done, _ = env.step(scaled_action)
            next_state = torch.FloatTensor(next_state).unsqueeze(0).to(device)
            episode_reward = reward + gamma*episode_reward
            advantage = reward + gamma * agent(next_state)[1].item() - agent(state)[1].item()
            advantage = torch.tensor(advantage).view(-1, 1).detach().to(device)
            # 데이터 저장
            data_to_save.append([
                epi_timestep,
                scaled_action[0:5].tolist(),
                scaled_action[5:10].tolist(),
                scaled_action[10:15].tolist(),
                temp_state[0][0:3].tolist(),
                temp_state[0][3:6].tolist(),
                temp_state[0][6:9].tolist(),
                temp_state[0][9:12].tolist(),
                temp_state[0][12].item() if torch.is_tensor(next_state[0][12]) else next_state[0][12],
                temp_state[0][13].item() if torch.is_tensor(next_state[0][13]) else next_state[0][13]
            ])
            episode_data.append((state, action, action_log_prob, advantage, reward, next_state))

            if timestep % update_timestep == 0 or done:
                for _ in range(k_epochs):
                    for state, action, action_log_prob, advantage, reward, next_state in episode_data:
                        state = normalize_state(state)
                        action_mean, state_value = agent(state)
                        action_log_std = agent.log_std.expand_as(action_mean)
                        action_std = torch.exp(action_log_std)
                        normal = Normal(action_mean, action_std)
                        log_prob = normal.log_prob(action).sum(dim=-1)
                        ratio = torch.exp(log_prob - action_log_prob.detach())
                        surr1 = ratio * advantage
                        surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantage
                        actor_loss = -torch.min(surr1, surr2).mean()
                        critic_loss = mse_loss(state_value, torch.tensor([reward + gamma * agent(next_state)[1].item()]).view(-1, 1).float().to(device).detach())

                        # Actor 네트워크 업데이트
                        actor_optimizer.zero_grad()
                        actor_loss.backward(retain_graph=True)
                        actor_optimizer.step()

                        # Critic 네트워크 업데이트
                        critic_optimizer.zero_grad()
                        critic_loss.backward()
                        critic_optimizer.step()

                        losses.append((actor_loss.detach().cpu().numpy(), critic_loss.detach().cpu().numpy()))

                episode_data = []

            state = next_state

            if done:
                
                break

        episode_rewards.append(episode_reward)

        # 에피소드 보상 출력
        print(f"Episode {len(episode_rewards)}, Reward: {episode_reward}")
        print("Updated log_std:", agent.log_std)

        # 모델 저장 및 데이터 저장
        if (episode + 1) % 25 == 0:
            torch.save(agent.state_dict(), model_path)
            save_data_to_csv(data_to_save, csv_path)
            save_pickle(episode_rewards, os.path.join(save_path, 'episode_rewards.pkl'))
            data_to_save = []  # 저장 후 리스트 초기화

        # 평균 보상 계산 및 종료 조건 확인
        if len(episode_rewards) >= 100:
            average_reward = np.mean(episode_rewards[-100:])
            print(f"Episode {episode + 1}, Average Reward (last 100 episodes): {average_reward}")
            if average_reward >= early_stopping_threshold:
                print(f"Early stopping at episode {episode + 1} with average reward {average_reward}")
                break
    
    
    return episode_rewards

if __name__=='__main__':
    env = Rocket()  # Rocket 환경 인스턴스
    state_dim = env.state_dims
    action_dim = env.action_dims
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    agent = PPORocketAgent(state_dim, action_dim).to(device)

    # Train the agent
    episode_rewards = []
    episode_rewards = train_ppo_agent(env, agent, num_episodes=5000, save_path = Path, early_stopping_threshold=295)
    plt.plot(episode_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training Progress')
    plt.show()
