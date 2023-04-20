import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np


class QNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(8, 10)
        self.fc2 = nn.Linear(10, 10)
        # self.fc3 = nn.Linear(15, 20)
        # self.fc4 = nn.Linear(20, 15)
        # self.fc5 = nn.Linear(15, 10)
        self.fc6 = nn.Linear(10, 4)
        self.optimizer = optim.Adam(self.parameters(), lr=5e-4)
        self.q_values = torch.rand((4))
        self.gamma = 0.1
        self.epsilon = 0.7

    def forward(self, x):
        x = x.to(torch.float32)
        x = F.tanh(self.fc1(x))
        # print(1,x)
        x = F.tanh(self.fc2(x))
        # x = F.tanh(self.fc3(x))
        # x = F.tanh(self.fc4(x))
        # x = F.tanh(self.fc5(x))
        # x = F.relu(self.fc2(x))
        # print(2,x)
        x = F.tanh(self.fc6(x))
        # print(3,x)
        return x

    def train(self, n_eps):
        env = gym.make("LunarLander-v2", render_mode="none")
        observation, _ = env.reset()
        last_state = torch.tensor(observation)
        action = torch.randint(4, ())
        losses = []
        # discount = 0.99
        for i in range(n_eps):
            # if i % 500 == 0:
            # print(i)
            observation, reward, terminated, truncated, ___ = env.step(action.item())
            if terminated or truncated:
                losses.append(loss.detach())
                # discount = 0.99
                env.reset()
                continue
            last_action = action
            state = torch.tensor(last_state, dtype=torch.float32)
            next_state = torch.tensor(observation, dtype=torch.float32)
            if np.random.rand(0, 1) < self.epsilon:
                action = np.random.randint(0, 4)
            else:
                action = torch.argmax(self.forward(next_state))
            # action = torch.tensor(last_action)
            reward = torch.tensor(reward, dtype=torch.float32)
            with torch.no_grad():
                next_q_values = self.forward(next_state)
                max_next_q_value = torch.max(next_q_values).item()
                q_target = reward + self.gamma * max_next_q_value
                # discount = discount * discount
            q_values = self(state)
            # print(q_values, q_target)
            self.optimizer.zero_grad()
            q_value = q_values[action]
            # for name, param in self.named_parameters():
            #     if param.requires_grad:
            #         print(name, param.data)
            loss = F.mse_loss(q_value, q_target)
            loss.backward()
            self.optimizer.step()
        plt.plot(losses)
        plt.show()
