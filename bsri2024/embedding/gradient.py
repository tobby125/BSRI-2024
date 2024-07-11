import random
import math
import numpy as np
from itertools import combinations
from scipy.optimize import basinhopping
from graph_embedding import Simplicial



def optimize(f, d, bounds=(0, 1)):
    a, b = bounds

    p = np.array([(b - a) * random.random() + a for _ in range(d)])
    units = np.eye(d)

    step = 1
    while True:
        val = f(p)
        print(p)
        print(val, f'step={step}')
        if val <= 0:
            return p

        grad = np.array([(f(p + step * units[i]) - f(p)) / step for i in range(d)])
        grad_ = np.array([(f(p - step * units[i]) - f(p)) / step for i in range(d)])

        if all(grad >= 0) and all(grad_ >= 0):
            step *= 1.1
            continue

        norm = np.linalg.norm(grad)
        if norm == 0:
            step *= 1.1
            continue
        new_p = p - step / norm * grad
        while f(new_p) > val:
            step *= 0.9
            new_p = p - step / norm * grad

        p -= step / norm * grad


i = 0
def num_intersections(coords, S):
    global i
    i += 1
    if i % 100 == 0:
        print(i)
    S.coords = coords
    for f in S.faces:
        if not S.is_simplex(f):
            return math.comb(len(S.faces), 2) + 1

    s = 0
    for f1, f2 in combinations(S.faces, 2):
        inter, point = S.intersect(f1, f2)
        if inter:
            s += 1
    return s

S = Simplicial(0)
if __name__ == '__main__':
    v = 8
    F = [(0, 1, 2), (3, 4, 5), (1, 6, 7), (0, 3, 5), (2, 4, 6), (0, 3, 7), (1, 4, 5), (2, 3, 6), (4, 5, 7)]
    d = v - 4

    S.verts = v

    for i, f in enumerate(F):
        for j, f_ in enumerate(F):
            if j in [(i - 1) % len(F), (i + 1) % len(F)]:
                assert not set(f) & set(f_)
            else:
                assert set(f) & set(f_)

    S.non_faces(F)
    S.reduce_faces()
    S.embed(d)
    print(num_intersections(S.coords))


    f = lambda p: num_intersections(np.reshape(p, (v, d)))

    #print(optimize(f, v * d))

    guess = np.array([random.random() for _ in range(v * d)])
    b = basinhopping(f, guess)
    print(b)
    S.coords = np.reshape(b.x, (v, d))
    print()
    print(S.coords)
    assert S.check_embedding()