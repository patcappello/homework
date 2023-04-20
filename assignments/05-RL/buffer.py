import torch


class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.ptr = 0
        self.n_samples = 0
        self.state = torch.zeros(buffer_size, 8)
        self.action = torch.zeros(buffer_size, 4)
        self.reward = torch.zeros(buffer_size, 1)
        self.discount = torch.zeros(buffer_size, 1)
        self.next_state = torch.zeros(buffer_size, 8)

    def add(self, state, action, reward, discount, next_state):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.discount[self.ptr] = discount
        self.next_state[self.ptr] = next_state

        if self.n_samples < self.buffer_size:
            self.n_samples += 1
        self.ptr = (self.ptr + 1) % self.buffer_size

    def sample(self, batch_size):
        idx = self.rand_generator.choice(self.n_samples, size=batch_size)
        state = self.state[idx]
        action = self.action[idx]
        reward = self.reward[idx]
        discount = self.discount[idx]
        next_state = self.next_state[idx]
        return (state, action, reward, discount, next_state)
