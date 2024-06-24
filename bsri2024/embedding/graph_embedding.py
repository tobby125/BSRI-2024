# importing mplot3d toolkits, numpy and matplotlib
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations, chain, product
import random
from scipy.optimize import linprog
import time

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
        '''space = list(product(np.linspace(0, 99, 100), repeat=d))
        self.coords = random.sample(space, len(self.verts))'''
        self.coords = [tuple(random.random() for _ in range(d)) for _ in range(len(self.verts))]
        #self.coords = [(0, 0), (1, 1), (1, 0), (0, 1)]

    def is_simplex(self, f):
        if len(f) <= 1:
            return True
        v1 = np.array(self.coords[f[0]])
        M = np.array([self.coords[i] - v1 for i in f[1:]])
        return np.linalg.matrix_rank(M) == len(f) - 1

    def intersect(self, f1, f2):
        if not (f1 and f2):
            return False, []
        M = np.array([self.coords[i] + (1, 0) for i in f1] + [np.concatenate((-np.array(self.coords[i]), [0, 1])) for i in f2]).T
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
        r = True
        for f in self.faces:
            if not self.is_simplex(f):
                if p:
                    print(f'Face {f} with coords {[S.coords[i] for i in f]} not a simplex')
                r = False
                return False

        for f1, f2 in combinations(self.faces, 2):
            inter, point = self.intersect(f1, f2)
            if inter:
                if p:
                    print(f'Faces {f1} and {f2} intersect with coords {[S.coords[i] for i in f1]} and {[S.coords[i] for i in f2]} at point {point}')
                r = False
                return False

        return r

    def plot(self):
        plt.figure()

        for face in self.faces:
            points = [self.coords[i] for i in face]
            points += [points[0]]
            plt.plot(*np.array(points).T)

        for v in self.verts:
            plt.annotate(v, self.coords[v])

        plt.show()

def powerset(x):
    return list(chain.from_iterable(combinations(x, i) for i in range(len(x) + 1)))


def run(v, F, d):
    S = Simplicial(v)
    S.non_faces(F)
    print(S.faces)
    S.reduce_faces()
    print(S.faces)
    print()

    for i in range(100000):
        if i % 100 == 0:
            print(i)
        S.embed(d)
        if S.check_embedding():
            print()
            print('success!')
            print(S.coords)
            S.plot()
            break

v = 6
F = [(0, 1), (2, 3), (0, 4), (1, 2), (3, 4)]
d = 2

run(v, F, d)