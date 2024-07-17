import numpy as np
from graph_embedding import Simplicial
from sympy import Matrix, symbols
# from sympy.solvers.solveset import linsolve
from pypoman import compute_polytope_vertices
import time
from scipy.spatial import ConvexHull, _qhull
from scipy.optimize import basinhopping, minimize
from itertools import combinations
import random
import math

np.set_printoptions(precision=2)

def num_intersections(S):
    for f in S.faces:
        if not S.is_simplex(f):
            return math.comb(len(S.faces), 2) + 1

    s = 0
    for f1, f2 in combinations(S.faces, 2):
        inter, point = S.intersect(f1, f2)
        if inter:
            s += 1
    return s

def intersection_volume(S, f1, f2):
    d = len(S.coords[0])
    if len(f1) < d + 1 or len(f2) < d + 1:
        return 0

    A_eq = np.array([np.concatenate((S.coords[i], (1, 0))) for i in f1] + [np.concatenate((-np.array(S.coords[i]), [0, 1])) for i in f2]).T
    b_eq = np.concatenate((np.zeros(len(S.coords[0])), [1, 1]))

    if np.linalg.matrix_rank(A_eq) != np.linalg.matrix_rank(np.column_stack((A_eq, b_eq))):
        return 0

    def give_ineq(A_eq, b_eq):
        A = np.column_stack((A_eq, b_eq))

        M, pivots = Matrix(A).rref()
        M = np.array(M, dtype=float)


        new = np.zeros((M.shape[1] - 1, M.shape[1]))
        for i, p in enumerate(pivots):
            new[p] = M[i]

        for i in range(new.shape[0]):
            if i in pivots:
                new[i] = M[pivots.index(i)]
            else:
                new[i][i] = -1

        neww = np.array([[i[j] for j in range(new.shape[1]) if j not in pivots] for i in new])

        return neww[:, :-1], neww[:, -1]

    A, b = give_ineq(A_eq, b_eq)


    def vertices(A, b):
        vs = compute_polytope_vertices(A, b)
        return [-A @ v + b for v in vs]

    try:
        vs = vertices(A, b)
    except RuntimeError:
        return 1000

    verts = []
    for vert in vs:
        lin_comb = sum(c * np.array(v) for c, v in zip(vert[:len(f1)], [S.coords[i] for i in f1]))
        verts.append(lin_comb)

    if not verts:
        return 0
    verts = np.array(verts)
    dim = np.linalg.matrix_rank(verts - verts[0])
    if dim < len(verts[0]):
        return 0

    if len(verts) == 2:
        return math.dist(*verts)

    polytope = ConvexHull(verts)
    return polytope.volume

def total_intersection_volume(S):
    for face in S.faces:
        if not S.is_simplex(face):
            return 1000

    s = 0
    for f1, f2 in combinations(S.faces, 2):
        try:
            v = intersection_volume(S, f1, f2)
        except _qhull.QhullError:
            v = 1000

        s += v
    return s
def total_intersection_volume_approx(S):
    for face in S.faces:
        if not S.is_simplex(face):
            return 1000

    s = 0
    for f1, f2 in subset:
        try:
            v = intersection_volume(S, f1, f2)
        except _qhull.QhullError:
            v = 1000

        s += v

    if s == 0 and not S.check_embedding():
        return 1
    return s * math.comb(len(S.faces), 2) / len(subset)

def intersection_volume_proportion(S, approx=False):
    total = 0
    for face in S.faces:
        d = len(S.coords[0])
        if len(face) < d + 1:
            continue
        total += abs(np.linalg.det([S.coords[i] - S.coords[face[0]] for i in face[1:]])) / math.factorial(d)
    return (total_intersection_volume_approx() if approx else total_intersection_volume(S)) / total


def find_embedding(S, approx=False):
    global subset

    sphere_count = 0

    def f(p):
        S.coords = np.reshape(p, (v, d))
        r = intersection_volume_proportion(S, approx=approx)
        r = round(r, 12)
        '''if r == 0:
            if S.check_embedding() and d <= 3:
                S.plot(d)'''
        print(100*'\b', end='')
        print(r, end='')
        return r

    results = []
    while True:
        subset = random.sample(list(combinations(S.faces, 2)), 50)
        guess = np.array([random.random() for _ in range(v * d)])

        optimizer = minimize

        b = optimizer(f, guess)
        print(100*'\b', end='')
        print(b.fun)
        if b.fun == 0.5:
            print('probably a sphere')
            print(list(list(float(j) for j in i) for i in S.coords))
            S.plot(d)
            if sphere_count == 1:
                return 0.5
            sphere_count += 1
            continue
        if b.fun != 0:
            sphere_count = 0
            continue
        if S.check_embedding():
            print('success')
            return 0
        else:
            print('WEIRD CASE: EMBEDDING DIDN\'T WORK ')

        def g(p):
            S.coords = np.reshape(p, (v, d))
            pro = intersection_volume_proportion(S)
            if pro > 0:
                r = max(100, pro * 10000)
                print(100 * '\b', end='')
                print(r, end='')
                return r
            r = num_intersections()
            print(100 * '\b', end='')
            print(r, end='')
            return r

        c = basinhopping(g, b.x)
        print(c)
        results.append(c.fun)
        print(results)
        if c.fun == 0:
            if S.check_embedding():
                print('success')
            return 0


def better_find_embedding(S):
    sphere_count = 0

    def f(p):
        basis = S.faces[0]
        non_basis = [i for i in range(d + 4) if i not in basis]

        S.coords[basis[0]] = np.zeros(d)
        for i, x in enumerate(basis[1:]):
            S.coords[x] = np.eye(d)[i]
        vars = np.reshape(p, (v - d - 1, d))

        for i, x in enumerate(non_basis):
            S.coords[x] = vars[i]

        #S.coords = np.reshape(p, (v, d))

        r = intersection_volume_proportion(S)
        r = round(r, 12)
        print(100 * '\b', end='')
        print(r, end='')
        return r

    while True:
        guess = np.array([random.random() for _ in range((v - d - 1) * d)])
        #guess = np.array([random.random() for _ in range(v * d)])

        optimizer = minimize

        b = optimizer(f, guess)
        print(100 * '\b', end='')
        print(b.fun)
        if b.fun == 0.5:
            print('probably a sphere')
            print(list(list(float(j) for j in i) for i in S.coords))
            S.plot(d)
            if sphere_count == 1:
                return 0.5
            sphere_count += 1
            continue
        if b.fun != 0:
            sphere_count = 0
            continue
        if S.check_embedding():
            print('success')
            print(S.coords)
            S.plot(d)
            return 0
        else:
            print('WEIRD CASE: EMBEDDING DIDN\'T WORK ')


F = [(0, 1, 2), (3, 4, 5), (1, 2, 6), (0, 3, 5), (1, 2, 4), (0, 3, 6), (1, 4, 5), (0, 2, 3), (4, 5, 6)]

t = time.time()

v = max(max(i) for i in F) + 1
d = v - 4
S = Simplicial(v)

print(f'v = {v}, d = {d}')
print(f'F = {F}')

S.non_faces(F)
S.reduce_faces()

print(f'{len(S.faces)} faces: {S.faces}')
print(f'{len(S.faces)} choose 2 = {math.comb(len(S.faces), 2)}')

def multi():
    em = better_find_embedding
    print(f'#############  {em(S)}  #############')
    print(f'runtime = {time.time() - t}')
    print()


if __name__ == '__main__':

    from multiprocessing import Process

    p = []
    for _ in range(20):
        p1 = Process(target=multi)
        p1.start()
        p.append(p1)
    for p1 in p:
        p1.join()

    quit()

    Fs = [
        [(0, 1, 2), (3, 4, 5), (1, 2, 6), (0, 3, 5), (1, 2, 4), (0, 3, 6), (1, 4, 5), (0, 2, 3), (4, 5, 6)],
        [(0, 5, 3), (1, 6, 4), (2, 0, 5), (3, 1, 6), (4, 2, 0), (5, 3, 1), (6, 4, 2)],
        [(0, 5, 6, 3), (1, 7, 4), (2, 0, 5, 6), (3, 1, 7), (4, 2)],
        [(0, 3, 5, 6), (1, 4), (2, 0, 5), (3, 1, 6), (4, 2)],
        [(0, 3), (1, 4), (2, 0), (3, 1), (4, 2)],
        [(0, 3, 5), (1, 4), (2, 0, 5), (3, 1), (4, 2)],
        [(0, 3, 5, 6), (1, 4), (2, 0, 5, 6), (3, 1), (4, 2)],
        [(0, 5, 3), (1, 6, 4), (2, 0, 5), (3, 1, 6), (4, 2, 0), (5, 3, 1), (6, 4, 2)],
        [(0, 5, 3, 7), (1, 6, 4), (2, 0, 5, 7), (3, 1, 6), (4, 2, 0, 7), (5, 3, 1), (6, 4, 2)],
        [(0, 3, 5, 7), (1, 4, 6, 8), (2, 0, 5, 7), (3, 1, 6, 8), (4, 0, 2, 7), (5, 1, 3, 8), (6, 0, 2, 4), (7, 1, 3, 5), (8, 2, 4, 6)],
        [(0, 3, 5, 7, 9), (1, 4, 6, 8), (2, 0, 5, 7), (3, 1, 6, 8, 9), (4, 0, 2, 7), (5, 1, 3, 8), (6, 0, 2, 4, 9), (7, 1, 3, 5), (8, 2, 4, 6)]
    ]

    subset = None

    F = Fs[0]
    t = time.time()

    v = max(max(i) for i in F) + 1
    d = v - 4
    S = Simplicial(v)

    print(f'v = {v}, d = {d}')
    print(f'F = {F}')

    for i, f in enumerate(F):
        for j, f_ in enumerate(F):
            if j in [(i - 1) % len(F), (i + 1) % len(F)]:
                assert not set(f) & set(f_)
            else:
                assert set(f) & set(f_)

    S.non_faces(F)
    S.reduce_faces()

    print(f'{len(S.faces)} faces: {S.faces}')
    print(f'{len(S.faces)} choose 2 = {math.comb(len(S.faces), 2)}')
