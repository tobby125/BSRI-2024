import numpy as np
from itertools import product, permutations
import random
from scipy.optimize import linprog


class Tensor:
    def __init__(self, m, n):
        self.m = m
        self.n = n
        self.vals = np.zeros(n*(m,))
        self.indices = list(product(range(m), repeat=n))
        self.hyperplanes = []
        self.vecs = [np.array(i) for i in self.indices]
        self.zero = self.vecs[0]
        self.diags = list(self.diags_())

    def __getitem__(self, item):
        item = tuple(item)
        return self.vals[item]

    def __setitem__(self, key, value):
        key = tuple(key)
        self.vals[key] = value

    def random(self):
        for i in self.indices:
            self[i] = random.random()

    def span(self, elems):
        elems = [np.array(v) for v in elems]
        k = len(elems)
        s = {tuple(sum([c[i] * elems[i] for i in range(k)]) % self.m) for c in product(range(self.m), repeat=k)}
        return [np.array(v) for v in s]

    def perp(self, v):
        v = np.array(v)
        return [w for w in self.vecs if w @ v % self.m == 0]

    def cosets(self, h):
        for v in self.vecs:
            if list(v) not in [list(w) for w in h]:
                u = v
                break
        return [[(w + i*u) % self.m for w in h] for i in range(self.m)]

    def h_sum(self, planes=None):
        if planes is None:
            planes = self.hyperplanes
        return [round(sum(self[v] for v in h), 5) for h in planes]

    def diags_(self):
        for p in product(permutations(range(self.m)), repeat=self.n-1):
            # [i, p[0][i], p[1][i], ..., p[m-2][i]]
            yield [np.array([i] + [p[j][i] for j in range(self.n-1)]) for i in range(self.m)]

    def is_pos(self, diag):
        return all(self[i] > 0 for i in diag)

    def pos_diag(self):
        for diag in self.diags:
            if all(self[i] > 0 for i in diag):
                return True, diag
        return False

    def set_hyperplanes(self, perps):
        for p in perps:
            self.hyperplanes += self.cosets(self.perp(p))

    def h_matrix(self):
        return np.array([[1 if tuple(v) in map(tuple, h) else 0 for v in self.vecs] for h in self.hyperplanes])


    def __repr__(self):
        r = ''
        for i in product(range(self.m), repeat=self.n-3):
            for j in range(self.m):
                for k in self[i]:
                    r += ' '.join(str(round(l, 3)) for l in k[j]) + '    '
                r += '\n'
            r += '\n'
        return r

m = 4
n = 3
M = Tensor(m, n)
perps = [tuple(1 if j == i else 0 for j in range(n)) for i in range(n)] + [n*(1,)]
print(perps)
M.set_hyperplanes(perps)

def find_solution():
    A_eq = M.h_matrix()
    b_eq = np.ones(len(M.hyperplanes))

    A_ineq = -1 * np.eye(m**n)
    b_ineq = np.zeros(m**n)

    c = np.array([random.choice([-1]) * random.random() for _ in range(m**n)])
    c = np.zeros(m**n)
    bounds = [(0, random.random()) for _ in range(m**n)]
    result = linprog(c=c, A_eq=A_eq, b_eq=b_eq, bounds=bounds)

    # Extract the solution
    if result.success:
        solution = result.x
        M.vals = np.reshape(solution, n*(m,))
        return 'success'

for i in range(10000):
    if find_solution() == 'success':
        #if 1 or not M.pos_diag() and set(M.vals.flatten()) == {0, 1/2}:
        if not M.pos_diag():
            print(M)


'''from termcolor import colored
import os
os.system('color')

print(colored('hello', 'red'), colored('world', 'green'))'''