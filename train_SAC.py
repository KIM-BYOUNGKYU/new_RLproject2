import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import matplotlib.pyplot as plt
import csv
import os
import pickle
from rocket import Rocket  # rocket.py 파일에서 Rocket 클래스 임포트

# 환경 초기화
env = Rocket()
state_dim = 32
action_dim = 24
action_low = np.array([-30] * 10 + [0.4 * env.max_thrust[0]] * 5 + [-30] * 6 + [0.4 * env.max_thrust[1]] * 3)
action_high = np.array([30] * 10 + [env.max_thrust[0]] * 5 + [30] * 6 + [env.max_thrust[1]] * 3)

# SAC Actor
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, action_dim)
        self.fc_log_std = nn.Linear(256, action_dim)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_log_std(x)
        log_std = torch.clamp(log_std, -20, 2)  # For numerical stability
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        z = normal.rsample()
        action = torch.tanh(z)
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        action = action * torch.FloatTensor((action_high - action_low) / 2) + torch.FloatTensor((action_high + action_low) / 2)
        return action, log_prob.sum(dim=1, keepdim=True)

# SAC Critic
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_q = nn.Linear(256, 1)

    def forward(self, state, action):
        x = torch.relu(self.fc1(torch.cat([state, action], 1)))
        x = torch.relu(self.fc2(x))
        q = self.fc_q(x)
        return q

# 경험 리플레이 버퍼
class ReplayBuffer:
    def __init__(self, max_size=1000000):
        self.buffer = []
        self.max_size = max_size
        self.ptr = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.max_size:
            self.buffer.append(None)
        self.buffer[self.ptr] = (state, action, reward, next_state, done)
        self.ptr = (self.ptr + 1) % self.max_size

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.buffer), size=batch_size)
        states, actions, rewards, next_states, dones = zip(*[self.buffer[i] for i in ind])
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)

    def size(self):
        return len(self.buffer)


# 학습에 필요한 하이퍼파라미터
batch_size = 256
learning_rate = 3e-4
gamma = 0.99
tau = 0.005
alpha = 0.2  # Entropy coefficient

# 모델 초기화
actor = Actor(state_dim, action_dim)
critic1 = Critic(state_dim, action_dim)
critic2 = Critic(state_dim, action_dim)
critic1_target = Critic(state_dim, action_dim)
critic2_target = Critic(state_dim, action_dim)
critic1_target.load_state_dict(critic1.state_dict())
critic2_target.load_state_dict(critic2.state_dict())

# 옵티마이저
actor_optimizer = optim.Adam(actor.parameters(), lr=learning_rate)
critic1_optimizer = optim.Adam(critic1.parameters(), lr=learning_rate)
critic2_optimizer = optim.Adam(critic2.parameters(), lr=learning_rate)

# 리플레이 버퍼 초기화
replay_buffer = ReplayBuffer()

def save_model():
    torch.save({
        'actor_state_dict': actor.state_dict(),
        'critic1_state_dict': critic1.state_dict(),
        'critic2_state_dict': critic2.state_dict(),
        'actor_optimizer_state_dict': actor_optimizer.state_dict(),
        'critic1_optimizer_state_dict': critic1_optimizer.state_dict(),
        'critic2_optimizer_state_dict': critic2_optimizer.state_dict()
    }, 'checkpoint.pth')
    print("Models and optimizers saved.")

def load_model():
    if os.path.exists('checkpoint.pth'):
        checkpoint = torch.load('checkpoint.pth')
        actor.load_state_dict(checkpoint['actor_state_dict'])
        critic1.load_state_dict(checkpoint['critic1_state_dict'])
        critic2.load_state_dict(checkpoint['critic2_state_dict'])
        actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        critic1_optimizer.load_state_dict(checkpoint['critic1_optimizer_state_dict'])
        critic2_optimizer.load_state_dict(checkpoint['critic2_optimizer_state_dict'])
        print("Models and optimizers loaded.")

def save_rewards(episode_rewards):
    with open('episode_rewards.pkl', 'wb') as f:
        pickle.dump(episode_rewards, f)

def load_rewards():
    if os.path.exists('episode_rewards.pkl'):
        with open('episode_rewards.pkl', 'rb') as f:
            episode_rewards = pickle.load(f)
        return episode_rewards
    return []

def train_sac(num_episodes):
    episode_rewards = load_rewards()
    pending_writes = []

    # 진행 상태 저장 파일 초기화
    if not os.path.exists('progress.csv'):
        with open('progress.csv', 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Episode', 'Timestep','Reward', 'X', 'Y', 'Z'])
    

    load_model()

    for episode in range(num_episodes):
        state = env.reset()
        
        episode_reward = 0
        for t in range(env.max_step):  # Assuming max steps per episode is 1000

            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action, _ = actor.sample(state_tensor)
            action = action.detach().cpu().numpy()[0]
            next_state, reward, done , _= env.step(action)  # Assuming env.step() returns next_state, reward, done

            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            episode_reward = reward + gamma * episode_reward

            # 상태의 x, y, z 값을 임시로 저장
            pending_writes.append([episode, t, reward, state[0], state[1], state[2]])

            if replay_buffer.size() > batch_size:
                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

                states = torch.FloatTensor(states)
                actions = torch.FloatTensor(actions)
                rewards = torch.FloatTensor(rewards).unsqueeze(1)
                next_states = torch.FloatTensor(next_states)
                dones = torch.FloatTensor(dones).unsqueeze(1)

                with torch.no_grad():
                    next_actions, next_log_probs = actor.sample(next_states)
                    target_q1 = critic1_target(next_states, next_actions)
                    target_q2 = critic2_target(next_states, next_actions)
                    target_q = torch.min(target_q1, target_q2) - alpha * next_log_probs
                    target_q = rewards + (1 - dones) * gamma * target_q

                current_q1 = critic1(states, actions)
                current_q2 = critic2(states, actions)
                critic1_loss = torch.mean((current_q1 - target_q).pow(2))
                critic2_loss = torch.mean((current_q2 - target_q).pow(2))

                critic1_optimizer.zero_grad()
                critic2_optimizer.zero_grad()
                critic1_loss.backward()
                critic2_loss.backward()
                critic1_optimizer.step()
                critic2_optimizer.step()

                new_actions, log_probs = actor.sample(states)
                q1 = critic1(states, new_actions)
                q2 = critic2(states, new_actions)
                actor_loss = (alpha * log_probs - torch.min(q1, q2)).mean()

                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                for param, target_param in zip(critic1.parameters(), critic1_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

                for param, target_param in zip(critic2.parameters(), critic2_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            if done:
                break

        episode_rewards.append(episode_reward)

        # 진행 상황 출력
        if episode % 10 == 0:
            print(f'Episode {len(episode_rewards)}, Reward: {episode_reward}')

        # 모델 저장
        if episode % 100 == 0:
            save_model()

        # 진행 상태를 CSV 파일에 저장
        if len(pending_writes) >= 50:
            with open('progress.csv', 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(pending_writes)
            pending_writes = []
            save_rewards(episode_rewards)

        # 조기 종료 조건 체크
        if len(episode_rewards) > 100 and np.mean(episode_rewards[-100:]) > 200:
            print(f'Training finished at episode {len(episode_rewards)}')
            save_model()
            break

    # 남아있는 진행 상태를 CSV 파일에 저장
    if pending_writes:
        with open('progress.csv', 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(pending_writes)
        save_rewards(episode_rewards)

    # 리워드 플롯
    plt.plot(episode_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.show()

# 학습 시작
train_sac(num_episodes=1000)