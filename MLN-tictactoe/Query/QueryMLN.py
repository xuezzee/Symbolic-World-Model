from pracmln.mln.mrf import MRF
import numpy as np
import os
from pracmln import MLN, Database, query
import random
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

class QueryMLN():
    def __init__(self, mln):
        self.mln = mln
        self.formula = self.mln.formulas
        self.ground_formulas = [Formula(f) for f in self.formula]

    def most_prob_world(self, evidence):
        self.data = evidence
        self.mrf = self.mln.ground(self.data)
        self.variables = list(self.mrf._gndatoms.keys())
        score = []
        for world in self.worlds():
            score.append(self.world_score(world))
        score = torch.Tensor(score)
        prob = F.softmax(score)
        return prob

    def choose_action(self, evidence):
        prob = self.most_prob_world(evidence)
        cat = Categorical(prob)
        act = cat.sample().data.numpy()
        return int(act), prob.data.numpy()

    def worlds(self):
        evidence = []
        unknown = 0
        for e in self.mrf.evidence:
            if e != None:
                evidence.append(e)
            else:
                unknown += 1

        for i in range(unknown):
            a = np.zeros((unknown,)).astype(np.int)
            a[i] = 1
            world = evidence + list(a)
            yield world

    def world_score(self ,world):
        evidence = dict(map(lambda x,y:[x,y], self.variables, world))
        score = 0
        for formula in self.ground_formulas:
            for gf in formula.ground_formula:
                gf = list(reversed(gf))
                holds = True
                for atom in gf:
                    if atom[0] != '!':
                        if evidence[atom]:
                            continue
                        else:
                            holds = False
                            break
                    else:
                        if not evidence[atom[1:]]:
                            continue
                        else:
                            holds = False
                            break
                if holds:
                    score += formula.weight
        return score

    def get_n(self, f_idx):
        formula = self.ground_formulas[f_idx]
        weight = formula.weight
        count = []
        for world in self.worlds():
            evidence = dict(map(lambda x, y: [x, y], self.variables, world))
            count.append(0)
            for gf in formula.ground_formula:
                gf = list(reversed(gf))
                holds = True
                for atom in gf:
                    if atom[0] != '!':
                        if evidence[atom]:
                            continue
                        else:
                            holds = False
                            break
                    else:
                        if not evidence[atom[1:]]:
                            continue
                        else:
                            holds = False
                            break
                if holds:
                    count[-1] += 1
        return count, weight

    def update_weight(self, new_weight):
        for gf in range(len(self.ground_formulas)):
            self.ground_formulas[gf].weight = new_weight[gf]
        self.mln.weights = new_weight


class Formula():
    def __init__(self, formula):
        self.args = {}
        literal = formula.children
        num_lit = len(literal)
        self.weight = formula.weight
        self.lits = [Predicate(literal[i].predname, literal[i].negated,
                               args=literal[i].args) for i in range(num_lit)]
        for i in range(num_lit):
            for a in literal[i].args:
                self.args[a] = None
        self.ground_formula = self.ground()

    def ground(self):
        ground_formula = []
        all_subs = []
        num_combination = 3 ** len(self.args)
        for i in range(num_combination):
            sub = []
            res = i
            for i in range(len(self.args)):
                sub.append(res % 3)
                res = res // 3
            all_subs.append(sub)
        for sub in all_subs:
            for const, idx in zip(sub, list(self.args.keys())):
                self.args[idx] = const
            ground_formula.append([lit.ground(self.args) for lit in self.lits])

        return ground_formula


class Predicate():
    def __init__(self, name, negated, args):
        arity = len(args)
        if negated:
            self.name = '!' + name
        else:
            self.name = name
        self.negated = negated
        if arity == 1:
            self.Lift_Predicate = self.name + "(arg{0})".format(args[0])
        elif arity == 2:
            self.Lift_Predicate = self.name + "(arg{0},arg{1})".format(args[0], args[1])
        self.args = args

    def ground(self, substitution):
        ground_predicate = self.Lift_Predicate
        for idx, value in substitution.items():
            if idx in self.args:
                ground_predicate = ground_predicate.replace("arg{0}".format(idx), str(substitution[idx]))

        return ground_predicate


def learnMLN(query_mln, n_f, act, probs):
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

    for i, f in enumerate(ns):
        weight[i] = weight[i] + 1 * grad(f, act, probs)

    return weight





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


if __name__ == '__main__':
    formulas = get_formula()
    print(formulas)
    formulas = add_w_to_formula(formulas, [0 for _ in formulas])
    predicats = get_predicate()
    data = get_data()
    print(data)
    data, mln = model_config(predicats, formulas, data, 'TicTacToe.mln', 'TicTacToe.db')
    query = QueryMLN(mln)
    data = get_data()
    data, _ = model_config(predicats, formulas, data, 'TicTacToe.mln', 'TicTacToe.db', mln)
    for i in range(1000):
        act, prob = query.choose_action(evidence=data)
        print("prob:", prob)
        new_weight = learnMLN(query, len(formulas), act, prob)
        query.update_weight(new_weight)

