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
    # Ensure action is within [-1, 1]
    action = np.clip(action, 0, 1)
    scaled_action = low +  (action) * (high - low)
    return scaled_action

def normalize_state(state):
    return (state - state.mean()) / (state.std() + 1e-8)

def save_data_to_csv(data, filename):
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epi','Timestep', 'A_0', 'A_1','A_2','A_3','A_4','A_5','A_6','A_7','A_8', 'A_9','A_10','A_11','A_12','A_13','A_14',
                         'S_0', 'S_1', 'S_2', 'S_3', 'S_4', 'S_5', 'S_6', 'S_7', 'S_8', 'S_9', 'S_10', 'S_11', 'S_12', 'S_13', 'S_13' ])
        for row in data:
            writer.writerow([int(row[0])] +
                            [int(row[1])] + 
                            [float(x) for x in row[2]] + 
                            [float(x) for x in row[3]] + 
                            [float(x) for x in row[4]] + 
                            [float(x) for x in row[5]] + 
                            [float(x) for x in row[6]] + 
                            [float(x) for x in row[7]] + 
                            [float(x) for x in row[8]] + 
                            [float(row[9])] + 
                            [float(row[10])])

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

def save_checkpoint(agent, actor_optimizer, critic_optimizer, episode_rewards, filepath):
    checkpoint = {
        'agent_state_dict': agent.state_dict(),
        'actor_optimizer_state_dict': actor_optimizer.state_dict(),
        'critic_optimizer_state_dict': critic_optimizer.state_dict(),
        'episode_rewards': episode_rewards
    }
    torch.save(checkpoint, filepath)

def load_checkpoint(filepath, agent, actor_optimizer, critic_optimizer):
    if os.path.exists(filepath):
        checkpoint = torch.load(filepath)
        agent.load_state_dict(checkpoint['agent_state_dict'])
        actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        episode_rewards = checkpoint['episode_rewards']
        return episode_rewards
    return []

def train_ppo_agent(env, agent, num_episodes=1000, actor_learning_rate=1e-4, critic_learning_rate=1e-4, gamma=0.99, clip_epsilon=0.1, gae_lambda = 0.95,update_timestep=700, k_epochs=4, save_path='train_data/', early_stopping_threshold=350):
    actor_optimizer = optim.Adam([
        {'params': agent.actor.parameters()},
        {'params': agent.log_std, 'lr': actor_learning_rate}
    ], lr=actor_learning_rate)
    critic_optimizer = optim.Adam(agent.critic.parameters(), lr=critic_learning_rate)
    mse_loss = nn.MSELoss()

    model_path = os.path.join(save_path, 'ppo_rocket_model.pth')
    checkpoint_path = os.path.join(save_path, 'checkpoint.pth')
    csv_path = os.path.join(save_path, 'training_data.csv')

    # 모델과 옵티마이저 상태 불러오기
    episode_rewards = load_checkpoint(checkpoint_path, agent, actor_optimizer, critic_optimizer)
    data_to_save =  []

    action_low = np.array([-30] * 10 + [0.4 * env.max_thrust[0]] * 5 + [-30] * 6 + [0.4 * env.max_thrust[1]] * 3)
    action_high = np.array([30] * 10 + [env.max_thrust[0]] * 5 + [30] * 6 + [env.max_thrust[1]] * 3)

    timestep = 0

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
            episode_reward = reward + gamma * episode_reward
            data_to_save.append([
                len(episode_rewards),
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
            episode_data.append((state.clone().detach(), action.clone().detach(), action_log_prob.clone().detach(), reward, next_state.clone().detach()))

            if timestep % update_timestep == 0 or done:
                # GAE를 사용하여 advantage를 계산
                rewards = [data[3] for data in episode_data]
                values = [agent(data[0])[1].item() for data in episode_data]
                next_values = values[1:] + [agent(episode_data[-1][4])[1].item()]
                deltas = [r + gamma * nv - v for r, nv, v in zip(rewards, next_values, values)]

                gae = 0
                advantages = []
                for delta in reversed(deltas):
                    gae = delta + gamma * gae_lambda * gae
                    advantages.insert(0, gae)

                returns = [adv + val for adv, val in zip(advantages, values)]

                for _ in range(k_epochs):
                    for i in range(0, len(episode_data), update_timestep):
                        batch_data = episode_data[i:i + update_timestep]
                        batch_advantages = advantages[i:i + update_timestep]
                        batch_returns = returns[i:i + update_timestep]

                        states = torch.cat([data[0] for data in batch_data]).to(device)
                        actions = torch.cat([data[1] for data in batch_data]).to(device)
                        old_log_probs = torch.cat([data[2] for data in batch_data]).to(device)
                        batch_advantages = torch.tensor(batch_advantages).view(-1, 1).to(device).clone().detach()
                        batch_returns = torch.tensor(batch_returns).view(-1, 1).to(device).clone().detach()

                        action_mean, state_values = agent(states)
                        action_log_std = agent.log_std.expand_as(action_mean)
                        action_std = torch.exp(action_log_std)
                        normal = Normal(action_mean, action_std)
                        log_probs = normal.log_prob(actions).sum(dim=-1)
                        ratios = torch.exp(log_probs - old_log_probs)
                        surr1 = ratios * batch_advantages
                        surr2 = torch.clamp(ratios, 1 - clip_epsilon, 1 + clip_epsilon) * batch_advantages
                        actor_loss = -torch.min(surr1, surr2).mean()
                        critic_loss = mse_loss(state_values, batch_returns)

                        actor_optimizer.zero_grad()
                        actor_loss.backward(retain_graph=True)
                        actor_optimizer.step()

                        critic_optimizer.zero_grad()
                        critic_loss.backward(retain_graph=True)
                        critic_optimizer.step()


                episode_data = []

            state = next_state

            if done:
                break
        episode_rewards.append(episode_reward)

        # 에피소드 보상 출력
        print(f"Episode {len(episode_rewards)}, Reward: {episode_reward}")

        # 모델 저장 및 데이터 저장
        if len(episode_rewards) % 50 == 0 :
            save_checkpoint(agent, actor_optimizer, critic_optimizer, episode_rewards, checkpoint_path)
            save_data_to_csv(data_to_save, csv_path)
            save_pickle(episode_rewards, os.path.join(save_path, 'episode_rewards.pkl'))
            data_to_save = []  # 저장 후 리스트 초기화
            #if(episode +1)%75 == 0:
                #env.show_path_from_state_buffer()
                #env.animate_trajectory(skip_steps=100)

        # 평균 보상 계산 및 종료 조건 확인
        if len(episode_rewards) >= 15:
            average_reward = np.mean(episode_rewards[-15:])
            print(f"Episode {episode + 1}, Average Reward (last 15 episodes): {average_reward}")
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
    episode_rewards = train_ppo_agent(env, agent, num_episodes=10000, save_path = Path, early_stopping_threshold=550)
    plt.plot(episode_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training Progress')
    plt.show()
