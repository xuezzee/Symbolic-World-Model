import numpy as np
import random

class GFormula():
    def __init__(self, formula_name, domain):
        self.formula_name = formula_name
        self.domain = domain

    def grounding(self):
        grounding_predicate = []
        vars = [self.d[0] for d in self.domain]




if __name__ == '__main__':
    s = 'hello world'
    print(len(s))
    for i in s:
        print(i + 1)