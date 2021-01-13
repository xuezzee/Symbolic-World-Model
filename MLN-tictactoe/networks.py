import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical
import copy
GAMMA = 1

class Q_net(nn.Module):
    def __init__(self, dim_s, dim_a, dim_h, device="cpu"):
        super(Q_net, self).__init__()
        self.dim_s = dim_s
        self.dim_a = dim_a
        self.dim_h = dim_h
        self.device = device
        self.fc1 = nn.Linear(dim_s, dim_h)
        self.fc2 = nn.Linear(dim_h, dim_h)
        self.fc3 = nn.Linear(dim_h, 1)
        self.optim = torch.optim.Adam(self.parameters(), lr=0.0001)
        self.scheduler_lr = torch.optim.lr_scheduler.StepLR(self.optim, step_size=2000, gamma=0.9, last_epoch=-1)

    def forward(self, input):
        x = self.convert_type(input)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.tanh(self.fc3(x))
        return x

    def cal_td_loss(self, s, r, _s):
        v1 = self.forward(s)
        v2 = self.forward(_s).detach()
        td_loss = GAMMA * v2 + r - v1
        return td_loss

    def learn(self, s, r, _s):
        td_loss = self.cal_td_loss(s, r, _s)
        loss = td_loss.square()
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        self.scheduler_lr.step()

    def convert_type(self, x):
        if not isinstance(x, torch.Tensor):
            return torch.Tensor(x).to(self.device)
        elif x.device() != torch.device(self.device):
            return x.to(self.device)
        else:
            return x

    def save(self, path='./model/'):
        state = {}
        state['net'] = self.state_dict()
        state['optim'] = self.optim.state_dict()
        torch.save(state, path+'modelCritic.pth')
        print('model saved')

    def load(self, path='./model/', device='cpu'):
        state = torch.load(path+"modelCritic.pth", map_location=device)
        self.load_state_dict(state['net'])
        self.optim.load_state_dict(state['optim'])


class Actor(nn.Module):
    def __init__(self, dim_s, dim_a, dim_h, device="cpu"):
        super(Actor, self).__init__()
        self.dim_s = dim_s
        self.dim_a = dim_a
        self.dim_h = dim_h
        self.device = device
        self.fc1 = nn.Linear(dim_s, dim_h)
        self.fc2 = nn.Linear(dim_h, dim_h)
        self.fc3 = nn.Linear(dim_h, dim_a)
        self.optim = torch.optim.Adam(self.parameters(), lr=0.001)
        self.scheduler_lr = torch.optim.lr_scheduler.StepLR(self.optim, step_size=2000, gamma=0.9, last_epoch=-1)

    def forward(self, s):
        s = self.convert_type(s)
        x = F.relu(self.fc1(s))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def choose_action(self, s):
        s = self.convert_type(s)
        self.logist = self.forward(s)
        self.prob = F.softmax(self.logist)
        m = Categorical(self.prob)
        a = m.sample()
        return int(a.data.cpu().numpy())

    def learn(self, td, a):
        log_prob = torch.log(self.prob[a])
        loss = -log_prob * td.detach()
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        self.scheduler_lr.step()

    def convert_type(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.Tensor(x)
        x = x.to(self.device)
        return x

    def save(self, path='./model/'):
        state = {}
        state['net'] = self.state_dict()
        state['optim'] = self.optim.state_dict()
        torch.save(state, path+'modelActor.pth')
        print('model saved')

    def load(self, path='./model/', device='cpu'):
        state = torch.load(path+"modelActor.pth", map_location=device)
        self.load_state_dict(state['net'])
        self.optim.load_state_dict(state['optim'])



