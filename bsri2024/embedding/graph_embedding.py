# importing mplot3d toolkits, numpy and matplotlib
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations, chain, product
import random

from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.optimize import linprog
import time
from math import sqrt

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
        r = True
        for f in self.faces:
            if not self.is_simplex(f):
                if p:
                    print(f'Face {f} with coords {[self.coords[i] for i in f]} not a simplex')
                r = False
                return False

        for f1, f2 in combinations(self.faces, 2):
            inter, point = self.intersect(f1, f2)
            if inter:
                if p:
                    print(f'Faces {f1} and {f2} intersect with coords {[self.coords[i] for i in f1]} and {[self.coords[i] for i in f2]} at point {point}')
                r = False
                return False

        return r

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

def powerset(x):
    return list(chain.from_iterable(combinations(x, i) for i in range(len(x) + 1)))


def run(v, F, d):
    print(F)
    S = Simplicial(v)
    S.non_faces(F)
    print(S.faces)
    S.reduce_faces()
    print(S.faces)
    print(len(S.faces))
    print()

    for i in range(100000):
        if i % 100 == 0:
            print(i)
        S.embed(d)
        if S.check_embedding():
            print(i)
            print()
            print('success!')
            print(S.coords)
            S.plot(d)
            break

def add1(f):
    return [tuple(j + 1 for j in i) for i in f]
def sub1(f):
    return [tuple(j - 1 for j in i) for i in f]
def non_faces_from_maximals(maximals, k):
    if k == 5:
        return [maximals[i] + maximals[(i - 2) % 5] for i in range(5)]

def simplex_coords(d):
    coords = np.append([np.zeros(d)], np.eye(d), axis=0)
    avg = sum(coords) / len(coords)
    coords -= avg
    return coords

def boundary(face):
    return list(combinations(face, len(face) - 1))
def join(sets):
    return [sum(x, start=()) for x in product(*sets)]

def boundary_join(maximals, boundary_indices):
    j = [boundary(maximals[i]) for i in boundary_indices] + [[x] for i, x in enumerate(maximals) if i not in boundary_indices]
    j = join(j)
    for i, x in enumerate(j):
        j[i] = tuple(sorted(x))
    return j

def facet_partition(maximals, k):
    if k == 5:
        return [boundary_join(maximals, np.array([i, i + 1, i + 2]) % 5) for i in range(5)]
    if k == 7:
        return [boundary_join(maximals, np.array([i, i + 1, i + 2]) % 7) for i in range(7)] + \
                [boundary_join(maximals, np.array([i, i + 1, i + 4]) % 7) for i in range(7)]

def layer_general(verts, start_coord, total_dim, z_coords):
    return [np.concatenate((np.zeros(start_coord), x, np.zeros(total_dim - start_coord - verts - len(z_coords) + 1), z_coords))
              for x in simplex_coords(verts - 1)]

seven_coords = np.array([
     [0.78699, 0.24726, -1],
     [0.4, .6, -1],
     [0, 1.35653, 0.30976],
     [0.73746, 1.23658, 1],
     [0.24891, 0.6016, 2],
     [-0.26699, 0.35056, -0.5],
     [0.8, 0.3, 0.5],
     [0.78699, 0.24726, -1],
     [0.4, .6, -1],
     [0.4, .6, -1]
])

def layer_coords_general(maximals, k):
    v = max(sum(maximals, start=())) + 1
    d = v - 4
    x = v * [0]
    start = 0

    for i, s in enumerate(maximals):
        lay = layer_general(len(s), start, d, seven_coords[i])
        for j, la in zip(s, lay):
            x[j] = la
        start += len(s) - 1
    return x

def embed_layers_general(maximals, k):
    x = layer_coords_general(maximals, k)
    part = facet_partition(maximals, k)
    faces = sum(part, start=[])

    faces1 = part[1] + part[5] + part[8] + part[9] + part[12] + part[13]
    embed_maximal_general(maximals, x, faces, faces1)

def embed_maximal_general(maximals, x, faces, faces1):
    v = max(max(i) for i in maximals)
    d = v - 4
    S = Simplicial(v)
    S.faces = faces
    print(f'faces = {faces}')
    faces2 = list(set(faces) - set(faces1))
    print(f'faces1 = {faces1}')
    print(f'faces2 = {faces2}')

    S.coords = np.array(x)
    for f in faces1, faces2:
        S.faces = f
        print(S.check_embedding())

    from testing import intersection_volume_proportion
    S.faces = faces
    print('intersection volume:', intersection_volume_proportion(S))
    S.faces = faces1
    #S.plot(d, double=faces2)

if __name__ == '__main__':
    maximals = (0, 7, 8), (1, 12), (2,), (3, 11), (4, 9, 10, 13), (5,), (6,)
    embed_layers_general(maximals, 7)