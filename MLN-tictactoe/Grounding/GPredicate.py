import numpy as np


class GPredicate():
    def __init__(self, predicate_name, domain):
        assert len(domain) == 2, "the domain is not complete"
        if not isinstance(predicate_name, str):
            raise Exception("predicate is not type str")

        self.predicate = predicate_name
        self.grounding(domain)

    def grounding(self, domain):
        grounding_candidates = []
        var1 = domain[0][0]
        var2 = domain[1][0]
        for i in range(len(domain[0][1])):
            for j in range(len(domain[1][1])):
                grounding_candidates.append([domain[0][1][i], domain[1][1][j]])
        self.grounding_predicate = []
        for g1, g2 in grounding_candidates:
            pred = self.predicate.replace(var1, g1)
            pred = pred.replace(var2, g2)
            self.grounding_predicate.append(pred)



if __name__ == '__main__':
    p = GPredicate('Place(x,y)', [['x', ['0', '1', '2']], ['y', ['0', '1', '2']]])
    print(p.grounding_predicate)