from torch import nn
import torch
import torch.nn.functional as F
import numpy as np

GAMMA = 0.99

class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(9, 32)
        self.fc2 = nn.Linear(32, 32)
        self.out = nn.Linear(32, 1)
        self.optim = torch.optim.Adam(self.parameters(), lr=0.001)

    def forward(self, x):
        x = torch.Tensor(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x

    def td_loss(self, s, r, s_, eval=True):
        v = self.forward(s)
        v_ = self.forward(s_).detach()
        td_loss = GAMMA * v_ + r - v
        if eval:
            return td_loss.data.numpy()
        else:
            return td_loss

    def learn(self, s, r, s_):
        td_loss = self.td_loss(s, r, s_, eval=False)
        loss = td_loss.square()
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()