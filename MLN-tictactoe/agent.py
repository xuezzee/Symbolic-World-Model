import torch
import numpy as np
from networks import Q_net as Critic
from pracmln.utils.project import PRACMLNConfig
from pracmln.utils import config, locs
from pracmln.utils.config import global_config_filename
from Critic import Critic
import os
import random
import torch.nn.functional as F
from core import *
from pracmln.mlnlearn import MLNLearn
from pracmln import MLN, Database, query
from Query.QueryMLN import QueryMLN

# def get_predicate():
#     predicate = []
#     predicate.append("Empty(coordx,coordy)")
#     predicate.append("Opponent(coordx,coordy)")
#     predicate.append("Mine(coordx,coordy)")
#     predicate.append("Place(coordx,coordy)")
#     return predicate

def get_predicate():
    predicate = []
    predicate.append("Empty(coordx,coordy)")
    predicate.append("Opponent(coordx,coordy)")
    predicate.append("Mine(coordx,coordy)")
    predicate.append("Place(coordx,coordy)")
    return predicate

def get_body_predicate():
    predicate = []
    predicate.append("Empty(coordx,coordx)")
    predicate.append("Opponent(coordx,coordx)")
    predicate.append("Mine(coordx,coordx)")
    # predicate.append("Place(coordx,coordx)")
    return predicate

def get_formula():
    formula = []
    # formula.append("!Empty(x,y) <=> !Place(x,y)")
    # formula.append("!Empty(x,y) <=> Place(x,y)")
    # formula.append("Opponent(x,y) <=> !Place(x,y)")
    # formula.append("Mine(x,y) <=> !Place(x,y)")
    # formula.append("Mine(x,y) ^ Mine(z,y) ^ Empty(k,y) <=> Place(k,y)")
    # formula.append("Opponent(x,y) ^ Opponent(z,y) ^ Empty(k,y) <=> Place(k,y)")
    # formula.append("Mine(x,y) ^ Mine(x,z) ^ Empty(x,k) <=> Place(x,k)")
    # formula.append("Opponent(x,y) ^ Opponent(x,z) ^ Empty(x,k) <=> Place(x,k)")
    # formula.append("Mine(x,y) <=> Place(x,y)")
    # formula.append("Opponent(x,y) ^ Empty(x,y) <=> Place(x,y)")
    # formula.append("Opponent(x,y) ^ Mine(z,y) ^ Empty(k,y) <=> Place(k,y)")
    formula.append("Empty(x,y) ^ Place(x,y)")
    formula.append("!Empty(x,y) ^ Place(x,y)")
    formula.append("Opponent(x,y) ^ Place(x,y)")
    formula.append("Mine(x,y) ^ Place(x,y)")
    formula.append("Mine(x,y) ^ Mine(z,y) ^ Empty(k,y) ^ Place(k,y)")
    formula.append("Opponent(x,y) ^ Opponent(z,y) ^ Empty(k,y) ^ Place(k,y)")
    formula.append("Mine(x,y) ^ Mine(x,z) ^ Empty(x,k) ^ Place(x,k)")
    formula.append("Opponent(x,y) ^ Opponent(x,z) ^ Empty(x,k) ^ Place(x,k)")
    formula.append("Mine(x,y) ^ Place(x,y)")
    formula.append("Opponent(x,y) ^ Empty(x,y) ^ Place(x,y)")
    formula.append("Opponent(x,y) ^ Mine(z,y) ^ Empty(k,y) ^ Place(k,y)")
    return formula

def get_data():
    raise NotImplemented

def add_w_to_formula(formula, weights):
    temp = []
    print(formula, weights)
    for i in range(len(formula)):
        temp.append(str(weights[i])+" "+formula[i])
    return temp

def template_2(predicate):
    formula = []
    # generate two to one
    for i in range(len(predicate)):
        for j in range(len(predicate)):
            for k in range(len(predicate)):
                if i > j > k:
                    formula.append("0 " + predicate[i] + "^" + predicate[j] + " ^ " + predicate[k])

    return formula

def template_3(predicate_truth, predicate_target):
    # generate 2 atom and 1 intertional variable in body
    formula = []

    # prepare the ary-pair
    pair = []
    for first in [['x', 'y'], ['x', 'z'], ['y', 'x'], ['y', 'z'], ['z', 'x'], ['z', 'y']]:
        for second in [['x', 'y'], ['x', 'z'], ['y', 'x'], ['y', 'z'], ['z', 'x'], ['z', 'y']]:
            temp = first.copy()
            temp.extend(second)
            if 'x' in temp and 'y' in temp:
                pair.append(temp)

    # generate the formula
    for target in predicate_target:
        t = target.split("(")[0]
        for first_item in predicate_truth:
            f = first_item.split("(")[0]
            for second_item in predicate_truth:
                s = second_item.split("(")[0]
                for p in pair:
                    single_formula = "{f}({p[0]},{p[1]})^{s}({p[2]},{p[3]}) ^ {t}(x,y)".format(f=f,s=s,t=t,p=p)
                    formula.append(single_formula)

    return formula


def inference(mln, evidence):
    evidence = mln.evidence

class Agent():

    def __init__(self, args):
        self.args = args
        # self.critic = Critic(args.dim_s, args.dim_a, args.dim_h, args.device)
        # self.optim = torch.optim.Adam(self.critic.parameters(), lr=0.001)
        # self.scheduler_lr = torch.optim.lr_scheduler.StepLR(self.optim, step_size=1000, gamma=0.9, last_epoch=-1)
        self.critic = Critic()
        self.predicate = get_predicate()
        f = get_formula()
        # f = template_3(get_body_predicate(), ["Place(coordx,coordx)"]) + get_formula()
        self.formula = add_w_to_formula(f, [0 for i in f])
        self.database = []
        self.data, self.mln = self.model_config(self.predicate, self.formula, self.database, 'TicTacToe.mln', 'TicTacToe.db')
        self.state_list = []
        self.step = 0
        self.action_list={0:{0:'Place(0,0)', 1:'Place(0,1)', 2:'Place(0,2)'},
                          1:{0:'Place(1,0)', 1:'Place(1,1)', 2:'Place(1,2)'},
                          2:{0:'Place(2,0)', 1:'Place(2,1)', 2:'Place(2,2)'}}
        self.EPSILON = 0.1

    def model_config(self, predicate, formula, database, mln_path, db_path, arg_mln=None):  # mln_path,db_path 為string
        base_path = os.getcwd()
        if arg_mln == None:
            mln = MLN(grammar='StandardGrammar', logic='FirstOrderLogic')
            for i in predicate:
                mln << i
                # print('input predicate successful:' + i)
            for i in formula:
                mln << i
                # print('input formula successful :' + i)
            # mln.write()
        else:
            mln = arg_mln
        mln.tofile(base_path + '\\' + mln_path)  # 把谓语数据储存成 mln_path.mln 档案
        db = Database(mln)
        try:
            for i in enumerate(database):
                db << i[1]
                # print('input database successful : ' + i[1][0] + ' : ' + i[1][1])
        except:
            for j in database[i[0]::]:
                db << j[1]

        # db.write()
        db.tofile(base_path + '\\' + db_path)  # 把证据数据储存成 db_path.db 档案
        for w in range(len(mln.weights)):
            mln.weights[w] = random.random()
        return (db, mln)

    def activate_model(self, database, mln, learning_rate):

        DEFAULT_CONFIG = os.path.join(locs.user_data, global_config_filename)
        conf = PRACMLNConfig(DEFAULT_CONFIG)

        config = {}
        config['verbose'] = False
        config['discr_preds'] = 0
        config['db'] = database
        config['mln'] = mln
        config['closedworld'] = True
        config['ignore_zero_weight_formulas'] = 0  # 0
        config['ignore_unknown_preds'] = True  # 0
        config['incremental'] = 1  # 0
        config['grammar'] = 'StandardGrammar'
        config['logic'] = 'FirstOrderLogic'
        config['method'] = 'CLL'  # BPLL
        config['multicore'] = True
        config['profile'] = 0
        config['shuffle'] = 0
        config['prior_mean'] = 0
        config['prior_stdev'] = 10  # 5
        config['save'] = True
        config['use_init_weights'] = True
        config['use_prior'] = 0
        config['learning_rate'] = learning_rate
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

    def choose_action(self, state, valid_action):
        state_list = []
        for item in state:
            if item.predicate.name == "empty":
                state_list.append("Empty({0},{1})".format(item.terms[0], item.terms[1]))
                state_list.append("!Mine({0},{1})".format(item.terms[0], item.terms[1]))
                state_list.append("!Opponent({0},{1})".format(item.terms[0], item.terms[1]))
            elif item.predicate.name == "mine":
                state_list.append("Mine({0},{1})".format(item.terms[0], item.terms[1]))
                state_list.append("!Empty({0},{1})".format(item.terms[0], item.terms[1]))
                state_list.append("!Opponent({0},{1})".format(item.terms[0], item.terms[1]))
            elif item.predicate.name == "opponent":
                state_list.append("Opponent({0},{1})".format(item.terms[0], item.terms[1]))
                state_list.append("!Mine({0},{1})".format(item.terms[0], item.terms[1]))
                state_list.append("!Empty({0},{1})".format(item.terms[0], item.terms[1]))

        self.state_list = state_list
        # if random.random() < self.EPSILON:
        #     act = "Place{0}".format(valid_action[random.randint(0, len(valid_action) - 1)])
        #     # act = self.action_list[random.randint(0,  2)][random.randint(0,  2)]
        #     print("random choice")
        # else:
        # state_list = [state_list[:20], state_list[20:]]
        # data, self.mln = self.model_config(self.predicate, self.formula, state_list, 'TicTacToe.mln', 'TicTacToe.db')
        data, self.mln = self.model_config(self.predicate, None, state_list, 'TicTacToe.mln', 'TicTacToe.db', self.mln)
        # querymln = QueryMLN(self.mln, data)
        # worlds, actions = self.worlds()
        # results = []
        # for w, a in zip(worlds, actions):
        # result = inference(self.mln, state_list)
        results = (query(queries="Place(x,y)", method='MC-SAT', mln=self.mln, db=data, verbose=False, multicore=True).run().results)
        probs = F.softmax(torch.Tensor(list(results.values())))
        print("probability:", probs)
        act = torch.distributions.Categorical(probs).sample()
        act = int(act.data.numpy())
        act = list(results.keys())[act]
            # original_act = act
        # act = act.replace(" ", "")
        # if (int(act[-4]), int(act[-2])) not in valid_action:
        #     act = "Place{0}".format(valid_action[random.randint(0, len(valid_action) - 1)])
        # print("valid action:", valid_action)
        print("act:", act)
        return (int(act[-4]), int(act[-2])), str(act)

    # def MLNlearn(self, a):
    #     self.state_list.append(a)
    #     for i in range(3):
    #         for j in range(3):
    #             if self.action_list[i][j] == a or '!'+self.action_list[i][j] == a:
    #                 pass
    #             else:
    #                 self.state_list.append('!' + self.action_list[i][j])
    #     data, mln = self.model_config(self.predicate, self.formula, self.state_list, 'TicTacToe.mln', 'TicTacToe.db')
    #     self.mln = self.activate_model(data, mln)
    #     self.formula = add_w_to_formula(get_formula(), self.mln.weights)
    #     print("formulas:", self.formula)

    def worlds(self):
        state_list = self.state_list
        actions = ['!Place(0,0)', '!Place(0,1)', '!Place(0,2)',
                   '!Place(1,0)', '!Place(1,1)', '!Place(1,2)',
                   '!Place(2,0)', '!Place(2,1)', '!Place(2,2)']
        actions_tobe_chosen = ['Place(0,0)', 'Place(0,1)', 'Place(0,2)',
                               'Place(1,0)', 'Place(1,1)', 'Place(1,2)',
                               'Place(2,0)', 'Place(2,1)', 'Place(2,2)']
        data = [actions[:i] + actions[i+1:] for i in range(len(actions))]
        data = [self.model_config(self.predicate, None, state_list+data[i], 'TicTacToe.mln', 'TicTacToe.db', self.mln)[0] for i in range(9)]
        return data, actions_tobe_chosen

    def MLNlearn(self, data, td):
        for i in range(len(data)):
            self.mln = self.activate_model(data, self.mln, td)
        # self.formula = add_w_to_formula(get_formula(), self.mln.weights)
        # self.formula = add_w_to_formula(template_3(get_predicate(), ["Place(coordx,coordx)"]), self.mln.weights)
        # print("formulas:", self.formula)

    def get_world(self, a):
        self.state_list.append(a)
        for i in range(3):
            for j in range(3):
                if self.action_list[i][j] == a or '!'+self.action_list[i][j] == a:
                    pass
                else:
                    self.state_list.append('!' + self.action_list[i][j])

        # return self.model_config(self.predicate, self.formula, self.state_list, 'TicTacToe.mln', 'TicTacToe.db', self.mln)[0]
        return \
        self.model_config(self.predicate, None, self.state_list, 'TicTacToe.mln', 'TicTacToe.db', self.mln)[0]

    # def learn(self, s, r, _s, atom_act, info=None):
    #     self.td = self.critic.cal_td_loss(s, r, _s).detach().cpu().numpy()
    #     # self.td = self.critic(s).detach().cpu().numpy()
    #     print("td_loss:", self.td)
    #     if self.td >= -0.1:
    #         # if info != None:
    #         #     self.state_list.append(info)
    #         self.MLNlearn(atom_act)
    #     # self.critic.learn(s, r, _s)

    def learn(self, data, transition, rate=None):
        td_loss = self.critic.td_loss(*transition)
        print("td loss:", td_loss)
        self.critic.td_loss(*transition)
        self.MLNlearn(data, td_loss)
        # self.EPSILON = self.EPSILON * 0.8
        # for i in range(len(self.mln.weights)):
        #     if self.mln.weights[i] < 0:
        #         self.mln.weights[i] = 0
        # a = self.mln.weight
        # print("epsilon: ", self.EPSILON)

    def load(self, path, device='cpu'):
        state = torch.load(path+"modelCritic.pth", map_location=device)
        self.critic.load_state_dict(state['net'])
        self.critic.optim.load_state_dict(state['optim'])


# if __name__ == '__main__':
#     print(template_3(get_predicate(), get_predicate()))

