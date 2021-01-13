import sys
import os
base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.append(base_dir)
sys.path.append(os.path.join(base_dir, 'MLN-tictactoe'))
from networks import Q_net as Critic
from networks import Actor

class Agent():
    def __init__(self, args):
        self.args = args
        self.critic = Critic(args.dim_s, args.dim_a, args.dim_h, args.device)
        self.actor = Actor(args.dim_s, args.dim_a, args.dim_h, args.device)


    def choose_action(self, s):
        # print("agent state:", s)
        return self.actor.choose_action(s)

    def learn(self, trans):
        td = self.critic.cal_td_loss(trans['s'],
                                     trans['r'],
                                     trans['s_'])
        self.critic.learn(trans['s'],
                          trans['r'],
                          trans['s_'])
        self.actor.learn(td, trans['a'])

    def save(self, path):
        self.critic.save(path)
        self.actor.save(path)

    def load(self, path):
        self.critic.load(path)
        self.actor.load(path)


