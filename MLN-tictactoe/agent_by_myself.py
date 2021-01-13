import numpy as np
import os
from pracmln import MLN, Database, query
import random
from Query.QueryMLN import QueryMLN, learnMLN
from Critic import Critic

def model_config(predicate, formula, database, mln_path, db_path, arg_mln=None):
    base_path = os.getcwd()
    if arg_mln == None:
        mln = MLN(grammar='StandardGrammar', logic='FirstOrderLogic')
        for i in predicate:
            mln << i
        for i in formula:
            mln << i
    else:
        mln = arg_mln
    mln.tofile(base_path + '\\' + mln_path)  # 把谓语数据储存成 mln_path.mln 档案
    db = Database(mln)
    try:
        for i in enumerate(database):
            db << i[1]
    except:
        for j in database[i[0]::]:
            db << j[1]
    db.tofile(base_path + '\\' + db_path)
    for w in range(len(mln.weights)):
        mln.weights[w] = random.random()
    return (db, mln)

def config_database(database, mln, db_path):
    base_path = os.getcwd()
    db = Database(mln)
    try:
        for i in enumerate(database):
            db << i[1]
    except:
        for j in database[i[0]::]:
            db << j[1]
    db.tofile(base_path + '\\' + db_path)
    return db

def get_formula():
    formula = []
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

def get_predicate():
    predicate = []
    predicate.append("Empty(coordx,coordy)")
    predicate.append("Opponent(coordx,coordy)")
    predicate.append("Mine(coordx,coordy)")
    predicate.append("Place(coordx,coordy)")
    return predicate

def get_data():
    data = []
    const = []
    for i in range(3):
        for j in range(3):
            const.append("({0},{1})".format(i,j))
    for i in range(9):
        data.append("Empty" + const[i])
    for i in range(9):
        data.append("Mine" + const[i])
    for i in range(9):
        data.append("Opponent" + const[i])

    # for i in range(random.randint(0, 10)):
    #     idx = [random.randint(0, len(data)-1)]
    idx = np.random.choice(len(data), random.randint(10, 20), replace=False)
    for i in idx:
        data[i] = '!'+data[i]

    return data

def add_w_to_formula(formula, weights):
    temp = []
    # print(formula, weights)
    for i in range(len(formula)):
        temp.append(str(weights[i])+" "+formula[i])
    return temp

def learnMLN(query_mln, n_f, act, probs, lr):
    def grad(ns, act, probs):
        ns = np.array(ns)
        n = ns[act]
        En = np.multiply(probs, ns).sum()
        grad = n - En
        return grad

    n_w = []
    for i in range(n_f):
        n_w.append(query_mln.get_n(i))

    weight = []
    for w in n_w:
        weight.append(w[-1])

    ns = []
    for n in n_w:
        ns.append(n[0])

    grads = []
    for i, f in enumerate(ns):
        grads.append(grad(f, act, probs))
        weight[i] = weight[i] + lr * grads[i]

    return weight

class MyMLNAgent():
    def __init__(self):
        formulas = get_formula()
        self.formulas = add_w_to_formula(formulas, [0 for _ in formulas])
        predicats = get_predicate()
        data = get_data()
        data, mln = model_config(predicats, self.formulas, data, 'TicTacToe.mln', 'TicTacToe.db')
        self.query = QueryMLN(mln)
        self.critic = Critic
        self.state_list = []
        self.prob = []
        self.atom_act = ["Place(0,0)"]

    def choose_action(self, state):
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
        mln = self.query.mln
        data = config_database(state_list, mln, 'TicTacToe.db')
        # act, self.prob = self.query.choose_action(data)
        idx_act, prob = self.query.choose_action(data)
        atom_act = self.query.variables[-9:][idx_act]
        exe_act = (int(atom_act[-4]), int(atom_act[-2]))
        self.prob.append(prob)
        return exe_act, atom_act, idx_act

    def learn(self, act, lr):
        # td_loss = self.critic()
        new_weight = learnMLN(self.query, len(self.formulas), act, self.prob, lr)
        self.query.update_weight(new_weight)

    def MCMC_learn(self, worlds):
        grads = [0 for _ in worlds]
        for i, w in enumerate(worlds):
            act = w[-1]
            state = w[0]
            mln = self.query.mln
            data, _ = config_database(state, mln, 'TicTacToe.db')
            new_weight, grad = learnMLN(self.query, act, self.formulas, self.prob[i])
            for i, g in enumerate(grad):
                grads[i] += g

        self.query.update(new_weight)

    def get_world(self, a):
        self.state_list.append(a)
        for i in range(3):
            for j in range(3):
                if self.action_list[i][j] == a or '!'+self.action_list[i][j] == a:
                    pass
                else:
                    self.state_list.append('!' + self.action_list[i][j])

        return \
        config_database(self.state_list, 'TicTacToe.db', self.query.mln)[0]
