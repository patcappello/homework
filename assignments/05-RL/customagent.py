import gymnasium as gym
import numpy as np
from net import QNet
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class Agent:
    def __init__(
        self, action_space: gym.spaces.Discrete, observation_space: gym.spaces.Box
    ):
        self.action_space = action_space
        self.observation_space = observation_space
        self.epsilon = 0.1
        self.gamma = 0.6
        self.last_action = None
        self.last_state = None
        self.qnet = QNet()
        self.optimizer = optim.Adam(self.qnet.parameters(), lr=0.005)

    def act(self, observation: gym.spaces.Box) -> gym.spaces.Discrete:
        self.last_state = observation
        state = torch.tensor(observation, dtype=torch.float)
        with torch.no_grad():
            q_values = self.qnet(state)
            action = torch.argmax(q_values).item()
        self.last_action = action
        return action

    def learn(
        self,
        observation: gym.spaces.Box,
        reward: float,
        terminated: bool,
        truncated: bool,
    ) -> None:
        state = torch.tensor(self.last_state, dtype=torch.float32)
        next_state = torch.tensor(observation, dtype=torch.float32)
        action = torch.tensor(self.last_action)
        reward = torch.tensor(reward, dtype=torch.float32)
        with torch.no_grad():
            next_q_values = self.qnet(next_state)
            max_next_q_value = torch.max(next_q_values).item()
            q_target = reward + self.gamma * max_next_q_value
        q_values = self.qnet(state)
        q_value = q_values[action]
        q_target.to(torch.float32)
        q_value.to(torch.float32)
        loss = F.mse_loss(q_value, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
