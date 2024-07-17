# importing mplot3d toolkits, numpy and matplotlib
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations, chain
import random

from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.optimize import linprog
import time
import math
from sympy import Matrix
from pypoman import compute_polytope_vertices
from scipy.spatial import ConvexHull, _qhull
from scipy.optimize import basinhopping, minimize

def t(f):
    def r(*args, **kwargs):
        start = time.time()
        result = f(*args, **kwargs)
        print(time.time() - start)
        return result
    return r


class Graph:
    def __init__(self, v, e):
        self.v = v
        self.e = e
        self.coords = v*[(0, 0, 0)]

    def embed3(self):
        self.coords = [(t, t**2, t**3) for t in range(self.v)]
        self.coords = [(random.random(), random.random(), random.random()) for _ in range(self.v)]

    def plot(self, d):
        fig = plt.figure()

        # syntax for 3-D projection
        ax = plt.axes(projection='3d')

        # defining all 3 axis
        t = np.linspace(0, 1, 100)
        for v1, v2 in self.e:
            x, y, z = np.outer(self.coords[v1], t) + np.outer(self.coords[v2], 1-t)

            # plotting
            ax.plot3D(x, y, z, 'green')
        ax.scatter(*np.array(self.coords).T)
        ax.set_title('simplicial complexes')

        '''ax.set_xlim3d(0, 100)
        ax.set_ylim3d(0, 100)
        ax.set_zlim3d(0, 100)'''
        plt.show()


def powerset(x):
    return list(chain.from_iterable(combinations(x, i) for i in range(len(x) + 1)))

class Simplicial:
    def __init__(self, vertices, faces=None):
        self.verts = range(vertices)
        if faces is None:
            self.faces = [(v,) for v in range(vertices)]
        else:
            self.faces = faces
        self.coords = vertices*[(0, 0, 0)]
        self.dim = max(map(len, self.faces)) + 1

    def non_faces(self, subsets):
        self.faces = []
        for s in powerset(self.verts):
            if not any(set(s_) <= set(s) for s_ in subsets):
                self.faces.append(s)

    def reduce_faces(self):
        old_faces = self.faces
        self.faces = []
        for s in old_faces:
            if not any(set(s) < set(s_) for s_ in old_faces):
                self.faces.append(s)

    def embed(self, d):
        #space = list(product(np.linspace(0, 9, 10), repeat=d))
        #self.coords = random.sample(space, len(self.verts))
        self.coords = [tuple(random.random() for _ in range(d)) for _ in range(len(self.verts))]

    def is_simplex(self, f):
        if len(f) <= 1:
            return True
        v1 = np.array(self.coords[f[0]])
        M = np.array([self.coords[i] - v1 for i in f[1:]])
        return np.linalg.matrix_rank(M) == len(f) - 1

    def intersect(self, f1, f2):
        if not (f1 and f2):
            return False, []
        M = np.array([np.concatenate((self.coords[i], (1, 0))) for i in f1] + [np.concatenate((-np.array(self.coords[i]), [0, 1])) for i in f2]).T
        c = np.array([-1 if i not in f2 else 0 for i in f1] + [-1 if i not in f1 else 0 for i in f2])
        b_eq = np.concatenate((np.zeros(len(self.coords[0])), [1, 1]))
        result = linprog(c=c, A_eq=M, b_eq=b_eq, bounds=[0, 1])

        if result.success:
            result = [round(i, 5) for i in result.x]
            for i, j in zip(c, result):
                if i and j:
                    return True, result
            return False, result
        return False, []

    def check_embedding(self, p=False):
        for f in self.faces:
            if not self.is_simplex(f):
                if p:
                    print(f'Face {f} with coords {[self.coords[i] for i in f]} not a simplex')
                return False

        for f1, f2 in combinations(self.faces, 2):
            inter, point = self.intersect(f1, f2)
            if inter:
                if p:
                    print(f'Faces {f1} and {f2} intersect with coords {[self.coords[i] for i in f1]} and {[self.coords[i] for i in f2]} at point {point}')
                return False

        return True

    def plot(self, d, double=None):
        fig = plt.figure(figsize=plt.figaspect(0.5))

        if d <= 2:
            ax = plt.axes()
        elif d == 3:
            ax = fig.add_subplot(1, 2, 1, projection='3d')
        else:
            return

        for face in self.faces:
            for pair in combinations(face, 2):
                points = [self.coords[i] for i in pair]
                points.append(points[0])
                ax.plot(*zip(*points))

        '''poly = [[self.coords[i] for i in tri] for face in self.faces for tri in combinations(face, 3)]
        ax.add_collection3d(Poly3DCollection(poly, edgecolors='black'))'''

        for v in self.verts:
            ax.text(*self.coords[v], v + 1, fontsize=20)

        if double is not None:

            self.faces = double
            ax2 = fig.add_subplot(1, 2, 2, projection='3d')
            for face in self.faces:
                for pair in combinations(face, 2):
                    points = [self.coords[i] for i in pair]
                    points.append(points[0])
                    ax2.plot(*zip(*points))
            for v in self.verts:
                ax2.text(*self.coords[v], v + 1, fontsize=20)

        lim = [-1, 1]
        '''ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_zlim(lim)'''
        plt.show()

    def num_intersections(self):
        for f in self.faces:
            if not self.is_simplex(f):
                return math.comb(len(self.faces), 2) + 1

        s = 0
        for f1, f2 in combinations(self.faces, 2):
            inter, point = self.intersect(f1, f2)
            if inter:
                s += 1
        return s

    def intersection_volume(self, f1, f2):
        d = len(self.coords[0])
        if len(f1) < d + 1 or len(f2) < d + 1:
            return 0

        A_eq = np.array(
            [np.concatenate((self.coords[i], (1, 0))) for i in f1] + [np.concatenate((-np.array(self.coords[i]), [0, 1])) for
                                                                   i in f2]).T
        b_eq = np.concatenate((np.zeros(len(self.coords[0])), [1, 1]))

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
            lin_comb = sum(c * np.array(v) for c, v in zip(vert[:len(f1)], [self.coords[i] for i in f1]))
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

    def total_intersection_volume(self):
        for face in self.faces:
            if not self.is_simplex(face):
                return 1000

        s = 0
        for f1, f2 in combinations(self.faces, 2):
            try:
                v = self.intersection_volume(f1, f2)
            except _qhull.QhullError:
                v = 1000

            s += v
        return s

    def total_intersection_volume_approx(self):
        for face in self.faces:
            if not self.is_simplex(face):
                return 1000

        s = 0
        for f1, f2 in subset:
            try:
                v = self.intersection_volume(f1, f2)
            except _qhull.QhullError:
                v = 1000

            s += v

        if s == 0 and not self.check_embedding():
            return 1
        return s * math.comb(len(S.faces), 2) / len(subset)

    def intersection_volume_proportion(self, approx=False):
        total = 0
        for face in self.faces:
            d = len(self.coords[0])
            if len(face) < d + 1:
                continue
            total += abs(np.linalg.det([self.coords[i] - self.coords[face[0]] for i in face[1:]])) / math.factorial(d)
        return (self.total_intersection_volume_approx() if approx else self.total_intersection_volume()) / total

    def find_embedding(self, approx=False):
        global subset

        sphere_count = 0

        def f(p):
            S.coords = np.reshape(p, (v, d))
            r = intersection_volume_proportion(S, approx=approx)
            r = round(r, 12)
            '''if r == 0:
                if S.check_embedding() and d <= 3:
                    S.plot(d)'''
            print(100 * '\b', end='')
            print(r, end='')
            return r

        results = []
        while True:
            subset = random.sample(list(combinations(S.faces, 2)), 50)
            guess = np.array([random.random() for _ in range(v * d)])

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

    def better_find_embedding(self, d):
        v = len(self.verts)

        sphere_count = 0

        def f(p):
            self.coords[0] = np.zeros(d)
            self.coords[1:d + 1] = np.eye(d)
            self.coords[d + 1:] = np.reshape(p, (v - d - 1, d))
            r = self.intersection_volume_proportion()
            r = round(r, 12)
            print(100 * '\b', end='')
            print(r, end='')
            return r

        while True:
            guess = np.array([random.random() for _ in range((v - d - 1) * d)])

            optimizer = minimize

            b = optimizer(f, guess)
            print(100 * '\b', end='')
            print(b.fun)
            if b.fun == 0.5:
                print('probably a sphere')
                print(list(list(float(j) for j in i) for i in self.coords))
                self.plot(d)
                if sphere_count == 1:
                    print(self.coords)
                    return 0.5
                sphere_count += 1
                continue
            if b.fun != 0:
                sphere_count = 0
                continue
            if self.check_embedding():
                print('success')
                print(self.coords)
                return 0
            else:
                print('WEIRD CASE: EMBEDDING DIDN\'T WORK ')

