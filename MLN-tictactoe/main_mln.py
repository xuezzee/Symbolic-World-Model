import pracmln
'''
if __name__ == "__main__":

    # init env

    # random policy to collect a set of transition

    # suprevised learning in mln
'''

#from __future__ import print_function, division, absolute_import
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


import os
import sys
from pracmln.utils.project import PRACMLNConfig
from pracmln.utils import config, locs
from pracmln.utils.config import global_config_filename
import os
from pracmln.mlnlearn import MLNLearn
from pracmln import MLN, Database, query


class social_modelling():

    def read_data(paths, predicate):  # 读txt数据用
        content = []
        base_path = os.getcwd()
        file = open(base_path + '/' + paths, 'r', encoding='utf8')
        pre_content = file.read()
        pre_content = pre_content.split('###')
        pre_content = [x for x in pre_content if x != '']
        for i in pre_content:
            element = i.split('\n')
            element = [x.replace(':', '_') for x in element if x != '']
            for j in element[1::]:
                splited = j.split('(')
                content.append((element[0], splited[0] + '(' + splited[1].upper()))
                # pracmln 要求 证据数据库db中的格式為 Predicate(CONTANT_1), 即谓语首字及谓语内
                # 变量contant首字為大写, 还有不可以有空格. 这里单纯方便而把变量大写
                # 另外暂不支持中文输入.
        return content

    def read_formula(paths, predicate):
        predicate_list = [x.split('(')[0] for x in predicate]
        predicate_list = predicate_list + ['!' + x for x in predicate_list]
        predicate_list = [' ' + x for x in predicate_list]
        predicate_list = [(x.lower(), x) for x in predicate_list]
        formula = []
        base_path = os.getcwd()
        file = open(base_path + '/' + paths, 'r', encoding='utf8')
        formula = file.read()
        formula = formula.split('\n')
        formula = [x for x in formula if x != '']
        formula = [' ' + x.replace(' or ', ' v ').replace(' and ', ' ^ ').replace(':', '') for x in formula]
        # exist_list = [x for x in formula if 'Exists ' in x]
        formula = ['0 ' + x for x in formula if 'Exists ' not in x]
        # 笔者仍在探索量词逻辑的使用
        # 加0 的作用是表示formula的权重, 这里先一律定义為0
        return formula

    def read_predicate(paths):
        predicate = []
        base_path = os.getcwd()
        file = open(base_path + '/' + paths, 'r', encoding='utf8')
        predicate = file.read()
        predicate = predicate.split('\n')
        predicate_list = [x.split('(')[0] for x in predicate]
        predicate_list2 = [x.split('(')[1].replace(' ', '').lower() for x in predicate]
        predicate = []
        for i in zip(predicate_list, predicate_list2):
            predicate.append(i[0] + '(' + i[1])
        return predicate

    def model_config(predicate, formula, database, mln_path, db_path):  # mln_path,db_path 為string
        base_path = os.getcwd()
        mln = MLN(grammar='StandardGrammar', logic='FirstOrderLogic')
        for i in predicate:
            mln << i
            print('input predicate successful:' + i)
        for i in formula:
            mln << i
            print('input formula successful :' + i)
        mln.write()
        mln.tofile(base_path + '/' + mln_path)  # 把谓语数据储存成 mln_path.mln 档案
        db = Database(mln)
        '''
        try:
            for i in enumerate(database):
                db << i[1][1]
                print('input database successful : ' + i[1][0] + ' : ' + i[1][1])
        except:
            for j in database[i[0]::]:
                db << j[1]

        db.write()
        db.tofile(base_path + '/' + db_path)  # 把证据数据储存成 db_path.db 档案
        '''
        return (db, mln)


    def activate_model(database, mln):

        DEFAULT_CONFIG = os.path.join(locs.user_data, global_config_filename)
        conf = PRACMLNConfig(DEFAULT_CONFIG)

        config = {}
        config['verbose'] = True
        config['discr_preds'] = 0
        config['db'] = database
        config['mln'] = mln
        config['ignore_zero_weight_formulas'] = 1  # 0
        config['ignore_unknown_preds'] = True  # 0
        config['incremental'] = 1  # 0
        config['grammar'] = 'StandardGrammar'
        config['logic'] = 'FirstOrderLogic'
        config['method'] = 'BPLL'  # BPLL
        config['multicore'] = False
        config['profile'] = 0
        config['shuffle'] = 0
        config['prior_mean'] = 0
        config['prior_stdev'] = 10  # 5
        config['save'] = True
        config['use_initial_weights'] = 0
        config['use_prior'] = 0
        # config['output_filename'] = 'learnt.dbpll_cg.student-new-train-student-new-2.mln'
        # 亲测无效, 此句没法储存.mln 档案
        config['infoInterval'] = 500
        config['resultsInterval'] = 1000
        conf.update(config)

        print('training...')
        learn = MLNLearn(conf, mln=mln, db=database)
        # learn.output_filename(r'C:\Users\anaconda3 4.2.0\test.mln')
        # 亲测无效, 此句没法储存.mln 档案
        result = learn.run()
        print('finished...')
        return result

    def inference(path, result, data, mln):  # 推理查询未知的命题
        query_list = []
        base_path = os.getcwd()
        file = open(base_path + '/' + path, 'r', encoding='utf8')
        query_list = file.read()
        query_list = query_list.split('\n')
        query_list = [x for x in query_list if x != '']
        for i in query_list:
            print(query(queries=i, method='MC-SAT', mln=mln, db=data, verbose=False, multicore=True).run().results)
        # #Other Methods: EnumerationAsk, MC-SAT, WCSPInference, GibbsSampler

    def inference_str(string, data, mln):
        result = query(queries=string, method='WCSPInference', mln=mln, db=data, verbose=True, multicore=False, save=True,
                    output_filename=r'learnt.dbpll_cg.student-new-train-student-new-2.mln').run().results
        print(result)
        # save = True, output_filename=r'learnt.dbpll_cg.student-new-train-student-new-2.mln' 无效
        return result

def stateToPredicate(state, next = False):
    result = []

    suffix = "_" if next else ""
    """
    for i in range(3):
        for j in range(3):
            if state[i][j] == "E":
                result.append("E{s}({i},{j})".format(i=i, j=j, s=suffix))
            if state[i][j] == "O":
                result.append("O{s}({i},{j})".format(i=i, j=j, s=suffix))
            if state[i][j] == "X":
                result.append("X{s}({i},{j})".format(i=i, j=j, s=suffix))
    """
    for symbol in ["E","O","X"]:
        for i in range(3):
            for j in range(3):
                if state[i][j] == symbol:
                    result.append("{sym}{s}({i},{j})".format(sym=symbol, i=i, j=j, s=suffix))
                else:
                    result.append("!{sym}{s}({i},{j})".format(sym=symbol, i=i, j=j, s=suffix))
    return result

def actionToPredicate(role,position):
    result = []
    for r in ["X","O"]:
        for i in range(3):
            for j in range(3):
                if r == role and position[0] == i and position[1] == j:
                    result.append("Place{r}({i},{j})".format(r=r, i=i, j=j))
                else:
                    result.append("!Place{r}({i},{j})".format(r=r, i=i, j=j))
    return result


def template(predicate):
    formula = []
    # generate one to one
    for i in range(len(predicate)):
        for j in range(len(predicate)):
            if i != j:
                formula.append("0 "+predicate[i]+" => "+predicate[j])

    return formula


def train_mln(mln, state, state_next, action):
    # input the training world
    world = []
    world.append(stateToPredicate(state))
    world.append(actionToPredicate(action[0], action[1]))
    world.append(stateToPredicate(state_next, True))

    # construct the database
    data = Database(mln)
    world = sum(world, [])  # flatten the 2-D list into a 1-D
    for item in world:
        data << item
    data.write()

    # learn the mln weight
    output = social_modelling.activate_model(data, mln)

    return  output

def generate_data():
    import numpy as np

    # 0-enputy, 1-X, 2-O
    char_map = {0:"E",1:"X",2:"O"}

    # 8 numbers in [0,3)
    data = np.random.randint(0, 3, 8)
    data = np.append(data, 0)

    # disorder the zero in the array
    np.random.shuffle(data)

    # find where to set new step
    emputy_idx = list(np.where(data==0)[0])
    #position = choice(emputy_idx)
    position = np.random.randint(0,9)
    role = 'X' if np.random.randint(2) == 0 else 'O'

    # state
    state = [["E", "E", "E"],
             ["E", "E", "E"],
             ["E", "E", "E"]]
    for i in range(3):
        for j in range(3):
            state[i][j] = char_map[data[3 * i + j]]

    # state_next
    state_next = [["E", "E", "E"],
                  ["E", "E", "E"],
                  ["E", "E", "E"]]
    for i in range(3):
        for j in range(3):
            state_next[i][j] = char_map[data[3 * i + j]]
    if data[position] == 0:
        state_next[int(position/3)][position % 3] = role

    return state, [role, [int(position/3), position % 3]], state_next

def print_state(state):
    for i in state:
        print(i)
    print("------------------")

if __name__ == '__main__':
    # init formula and predicate
    predicate = ["X(x,y)", "O(x,y)", "E(x,y)", "PlaceX(x,y)", "PlaceO(x,y)", "X_(x,y)", "O_(x,y)", "E_(x,y)"]
    formula = ["0 E(x,y) ^ PlaceX(x,y) => X_(x,y)",
               "0 E(x,y) ^ PlaceX(x,y) => !E_(x,y)",
               "0 E(x,y) ^ PlaceO(x,y) => O_(x,y)",
               "0 E(x,y) ^ PlaceO(x,y) => !E_(x,y)",
               #"0 E(x,y) => E_(x,y)",
               #"0 O(x,y) => O_(x,y)",
               #"0 X(x,y) => X_(x,y)"
               ]
    formula.extend(template(predicate))

    # init mln model
    data, mln = social_modelling.model_config(predicate, formula, None, 'tictacteo.mln', 'tictacteo.db')

    # input the training world
    for iter in range(50):
        # prepare train data
        '''
        state = [["X","E","E"],
                 ["X","O","E"],
                 ["E","E","E"]]

        state_next = [["X", "E", "E"],
                      ["X", "O", "E"],
                      ["O", "E", "E"]]

        aciton = ['O', [2,0]]
        '''
        state, action, state_next = generate_data()

        # train mln
        mln = train_mln(mln, state, state_next, action)

    output = mln

    state_eval = [["X", "E", "E"],
                  ["X", "O", "E"],
                  ["O", "E", "E"]]
    # input the evaluation world
    data = Database(output)
    world = []
    world.append(actionToPredicate('X', [1, 1]))
    world.append(stateToPredicate(state_eval))

    world = sum(world, [])
    for item in world:
        data << item
    data.write()

    # evaluate the transition
    res = []
    res.append(social_modelling.inference_str('E_(x,y)', data, output))
    res.append(social_modelling.inference_str('X_(x,y)', data, output))
    res.append(social_modelling.inference_str('O_(x,y)', data, output))


    print_state(state)
    print_state(state_next)
    print_state(state_eval)
    out = [[[], [], []],
           [[], [], []],
           [[], [], []]]
    i = 0
    for symbol in ["E","X","O"]:
        for p in list(res[i]):
            if res[i][p] == 1:
                x = int(p[3])
                y = int(p[5])
                out[x][y].append(symbol)
        i += 1

    for item in out:
        print(item)




















    '''
    #input the data
    board = {"win": 0, "lose": 0, "draw": 0}
    import random

    env = TicTacTeo()
    for ep in range(100):
        env.reset()
        done = False
        while not done:
            valid = env.get_valid()
            act = valid[random.randint(0, len(valid) - 1)]
            reward, done, _ = env.next_step(act)
            if done:
                if reward == 1:
                    board["win"] = board["win"] + 1
                elif reward == -1:
                    board["lose"] = board["lose"] + 1
                else:
                    board["draw"] = board["draw"] + 1

       

    output = social_modelling.activate_model(data, mln)
    output.tofile(os.getcwd() + '/' + 'learnt_mln.mln')
    social_modelling.inference_str('Smokes(person)', output, data, output)
    '''

    #print(board)