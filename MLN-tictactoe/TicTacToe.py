from __future__ import print_function, division, absolute_import
import sys, os
base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.append(base_dir)
from core.clause import *
from core.ilp import LanguageFrame
import copy
from random import choice, random
import torchsnooper


class SymbolicEnvironment(object):
    def __init__(self, background, initial_state, actions):
        '''
        :param language_frame
        :param background: list of atoms, the background knowledge
        :param positive: list of atoms, positive instances
        :param negative: list of atoms, negative instances
        '''
        self.background = background
        self._state = copy.deepcopy(initial_state)
        self.initial_state = copy.deepcopy(initial_state)
        self.actions = actions
        self.acc_reward = 0
        self.step = 0

    def reset(self):
        self.acc_reward = 0
        self.step = 0
        self._state = copy.deepcopy(self.initial_state)


UP = Predicate("up",0)
DOWN = Predicate("down",0)
LEFT = Predicate("left",0)
RIGHT = Predicate("right",0)
LESS = Predicate("less",2)
ZERO = Predicate("zero",1)
LAST = Predicate("last",1)
CLIFF = Predicate("cliff",2)
SUCC = Predicate("succ",2)
GOAL = Predicate("goal",2)
CURRENT = Predicate("current", 2)






ON = Predicate("on", 2)
TOP = Predicate("top", 1)
MOVE = Predicate("move", 2)
INI_STATE = [["a", "b", "c", "d"]]
INI_STATE2 = [["a"], ["b"], ["c"], ["d"]]
FLOOR = Predicate("floor", 1)
BLOCK = Predicate("block", 1)
CLEAR = Predicate("clear", 1)
MAX_WIDTH = 7

import string


"""
def random_initial_state():
    result = [[] for _ in range(self._block_n)]
    all_entities = ["a", "b", "c", "d", "e", "f", "g"][:BLOCK_N]
    swap(all_entities)
    for entity in all_entities:
        stack_id = np.random.randint(0, BLOCK_N)
        result[stack_id].append(entity)
    return result
"""

PLACE = Predicate("place", 2)
MINE = Predicate("mine", 2)
EMPTY = Predicate("empty", 2)
OPPONENT = Predicate("opponent", 2)
class TicTacTeo(SymbolicEnvironment):
    all_variations = ("")
    def __init__(self, width=3, know_valid_pos=True):
        actions = [PLACE]
        self.language = LanguageFrame(actions, extensional=[ZERO, MINE, EMPTY, OPPONENT, SUCC],
                                      constants=[str(i) for i in range(width)])
        background = []
        #background.extend([Atom(LESS, [str(i), str(j)]) for i in range(0, WIDTH)
        #                   for j in range(0, WIDTH) if i < j])
        background.extend([Atom(SUCC, [str(i), str(i + 1)]) for i in range(width - 1)])
        background.append(Atom(ZERO, ["0"]))
        self.max_step = 50
        initial_state = np.zeros([3,3])
        super(TicTacTeo, self).__init__(background, initial_state, actions)
        self.width = width
        self.all_positions = [(i, j) for i in range(width) for j in range(width)]
        self.know_valid_pos = know_valid_pos
        self.action_n = len(self.all_positions)
        self.state_dim = width**2

    def next_step(self, action):
        def tuple2int(t):
            return (int(t[0]), int(t[1]))
        self.step += 1
        valids = self.get_valid()
        if tuple2int(action) in valids:
            self._state[tuple2int(action)] = 1
        else:
            reward, finished = self.get_reward()
            return reward, finished, False
        self.random_move(self.know_valid_pos)
        reward, finished = self.get_reward()
        return reward, finished, True

    def next_step2(self, action):
        def tuple2int(t):
            return (int(t[0]), int(t[1]))
        self.step += 1
        reward, finished = self.get_reward()
        if finished:
            return reward, finished
        valids = self.get_valid()
        if tuple2int(action) in valids:
            self._state[tuple2int(action)] = 1
        self.random_move(self.know_valid_pos)
        reward, finished = self.get_reward()
        return reward, finished

    def next_step3(self, action):
        placed = False
        def tuple2int(t):
            return (int(t[0]), int(t[1]))
        self.step += 1
        # reward, finished = self.get_reward()
        # if finished:
        #     return reward, finished
        valids = self.get_valid()
        if tuple2int(action) in valids:
            self._state[tuple2int(action)] = 1
            placed = True
        if self.know_valid_pos:
            self.random_move(self.know_valid_pos)
        reward, finished = self.get_reward()
        return reward, finished, placed

    def get_valid(self):
        return [(x,y) for x,y in self.all_positions if self._state[x,y]==0]

    @property
    def all_actions(self):
        return [Atom(PLACE, [str(position[0]), str(position[1])]) for position in self.all_positions]

    def state2vector(self, state):
        return state.flatten()

    def state2atoms(self, state):
        atoms = set()
        def tuple2strings(t):
            return str(t[0]), str(t[1])
        for position in self.all_positions:
            if state[position] == 0:
                atoms.add(Atom(EMPTY, tuple2strings(position)))
            elif state[position] == -1:
                atoms.add(Atom(OPPONENT, tuple2strings(position)))
            elif state[position] == 1:
                atoms.add(Atom(MINE, tuple2strings(position)))
        return atoms

    @property
    def state(self):
        return copy.deepcopy(self._state)

    def random_move(self, know_valid):
        valid_position = self.get_valid()
        if not valid_position:
            return
        if know_valid:
            position = choice(valid_position)
            self._state[position] = -1
        else:
            position = choice(self.all_positions)
            if position in valid_position:
                self._state[position] = -1

    def get_reward(self):
        if np.any(np.sum(self._state, axis=0)==3) or np.any(np.sum(self._state, axis=1)==3):
            return 1, True
        for i in range(-self.width, self.width):
            if np.trace(self._state, i)==3 or np.trace(np.flip(self._state, 0),i)==3:
                return 1, True
        if np.any(np.sum(self._state, axis=0)==-3) or np.any(np.sum(self._state, axis=1)==-3):
            return -1, True
        for i in range(-self.width, self.width):
            if np.trace(self._state, i)==-3 or np.trace(np.flip(self._state, 0),i)==-3:
                return -1, True
        if not self.get_valid():
            return 0, True
        return 0, False


if __name__ == '__main__':
    board = {"win":0, "lose":0, "draw":0}
    import random
    env = TicTacTeo()
    for ep in range(100):
        env.reset()
        done = False
        while not done:
            valid = env.get_valid()
            act = valid[random.randint(0, len(valid)-1)]
            reward, done, _ = env.next_step(act)
            if done:
                if reward == 1:
                    board["win"] = board["win"] + 1
                elif reward == -1:
                    board["lose"] = board["lose"] + 1
                else:
                    board["draw"] = board["draw"] + 1