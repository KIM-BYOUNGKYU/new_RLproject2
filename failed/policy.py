import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def calculate_returns(next_value, rewards, masks, gamma=0.99):
    # 에피소드 끝: mask = 0 / 에피소드 진행중: mask = 1
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.append(R)
    returns.reverse() # reverse를 뒤에서 하도록 수정
    return returns


class PositionalMapping(nn.Module):
    """
    Positional mapping Layer.
    This layer map continuous input coordinates into a higher dimensional space
    and enable the prediction to more easily approximate a higher frequency function.
    See NERF paper for more details (https://arxiv.org/pdf/2003.08934.pdf)
    """

    def __init__(self, input_dim, L=5, scale=1.0):
        super(PositionalMapping, self).__init__()
        self.L = L
        self.output_dim = input_dim * (L*2 + 1)
        self.scale = scale

    def forward(self, x):

        x = x * self.scale

        if self.L == 0:
            return x

        h = [x]
        PI = 3.141592653589793
        for i in range(self.L):
            x_sin = torch.sin(2**i * PI * x)
            x_cos = torch.cos(2**i * PI * x)
            h.append(x_sin)
            h.append(x_cos)

        return torch.cat(h, dim=-1) / self.scale


class MLP(nn.Module):
    """
    Multilayer perception with an embedded positional mapping
    """

    def __init__(self, input_dim, output_dim, h_dim=128, L=7, scale=1.0):
        super().__init__()

        self.mapping = PositionalMapping(input_dim=input_dim, L=L, scale=scale)

        self.linear1 = nn.Linear(in_features=self.mapping.output_dim, out_features=h_dim, bias=True)
        self.linear2 = nn.Linear(in_features=h_dim, out_features=h_dim, bias=True)
        self.linear3 = nn.Linear(in_features=h_dim, out_features=h_dim, bias=True)
        self.linear4 = nn.Linear(in_features=h_dim, out_features=output_dim, bias=True)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        # shape x: 1 x m_token x m_state
        x = x.view([1, -1])
        x = self.mapping(x)
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.relu(self.linear3(x))
        x = self.linear4(x)
        return x


class ActorCritic(nn.Module):
    """
    RL policy and update rules
    """

    def __init__(self, state_size, action_size, h_dim=128, L=7, scale=1.0, lr=5e-5):
        # input_dim: state size / output_dim: action size
        super().__init__()

        self.action_size = action_size
        self.actor = MLP(input_dim=state_size, output_dim=action_size, h_dim=h_dim, L=L, scale=scale)
        self.critic = MLP(input_dim=state_size, output_dim=1, h_dim=h_dim, L=L, scale=scale)
        self.softmax = nn.Softmax(dim=-1)

        self.optimizer = optim.RMSprop(self.parameters(), lr=lr)

    def forward(self, x):
        # shape x: batch_size x m_token x m_state
        y = self.actor(x)
        probs = self.softmax(y) # 각 행동에 대한 확률 분포
        value = self.critic(x)

        return probs, value

    def get_action(self, state, deterministic=False, exploration=0.01):

        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device) # state를 PyTorch 텐서로 변환 후 device로 이동
        probs, value = self.forward(state)
        # 첫번째 배치 차원 제거, 1차원 텐서로 변경
        probs = probs[0, :] 
        value = value[0]

        if deterministic:
            action_id = np.argmax(np.squeeze(probs.detach().cpu().numpy()))
        else:
            if random.random() < exploration:  # exploration
                action_id = random.randint(0, self.action_size - 1)
            else: # 확률 분포에 따라 선택
                action_id = np.random.choice(self.action_size, p=np.squeeze(probs.detach().cpu().numpy()))

        log_prob = torch.log(probs[action_id] + 1e-9)

        return action_id, log_prob, value

    @staticmethod
    def update_ac(network, rewards, log_probs, values, masks, Qval, gamma=0.99):

        # compute Q values
        Qvals = calculate_returns(Qval.detach(), rewards, masks, gamma=gamma)
        Qvals = torch.tensor(Qvals, dtype=torch.float32).to(device).detach()

        log_probs = torch.stack(log_probs)
        values = torch.stack(values)

        advantage = Qvals - values
        actor_loss = (-log_probs * advantage.detach()).mean()
        critic_loss = 0.5 * advantage.pow(2).mean()
        ac_loss = actor_loss + critic_loss

        network.optimizer.zero_grad()
        ac_loss.backward() # backpropagation
        network.optimizer.step()