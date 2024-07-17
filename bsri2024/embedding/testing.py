from itertools import combinations, product
import math
import numpy as np
from graph_embedding import Simplicial


def add1(f):
    return [tuple(j + 1 for j in i) for i in f]
def sub1(f):
    return [tuple(j - 1 for j in i) for i in f]

def non_faces_from_maximals(maximals, k):
    F = k * [()]
    for j in range(k):
        for i in range((k - 1) // 2):
            F[(j + 2 * i) % k] += tuple(maximals[j])

def simplex_coords(d):
    coords = np.append([np.zeros(d)], np.eye(d), axis=0)
    avg = sum(coords) / len(coords)
    coords -= avg
    return coords

def reg_simplex_coords(n):
    if n == 0:
        return np.array([[]])
    if n == 1:
        return np.array([[1], [-1]])
    base_coords = reg_simplex_coords(n - 1)
    coords = np.concatenate((np.zeros((n, 1)), base_coords), axis=1)

    dist = math.dist(coords[0], coords[1])
    x = math.sqrt(dist ** 2 - sum(i ** 2 for i in coords[0]))
    new = np.zeros((1, n))
    new[0, 0] = x

    coords = np.concatenate((new, coords), axis=0)
    avg = sum(coords) / (n + 1)
    coords -= avg
    coords /= np.linalg.norm(coords[0])
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
    part = []

    for i in range(k):
        for j in range(i + 1, k, 2):
            for l in range(j + 1, k, 2):
                part.append(boundary_join(maximals, np.array([i, j, l]) % k))
    return part
def layer_general(verts, start_coord, total_dim, z_coords):
    return [np.concatenate((np.zeros(start_coord), x, np.zeros(total_dim - start_coord - verts - len(z_coords) + 1), z_coords))
              for x in reg_simplex_coords(verts - 1)]

def base_coords(k):
    d = k - 4

    x = np.zeros((k + 1, d))

    simplex = reg_simplex_coords((d - 1) // 2)

    for i in range((d - 1) // 2 + 1):
        x[2 * i + 1] = np.concatenate((simplex[i], np.zeros((d - 1) // 2), [-1]))
        x[2 * i + 2] = np.concatenate((np.zeros((d - 1) // 2), simplex[i], [1]))

    x[d + 4] = np.concatenate((np.zeros(d - 1), [-0.99]))
    x[d + 2] = np.concatenate((np.zeros(d - 1), [0.99]))

    x[d + 3] = np.concatenate(((d - 1) // 2 * [-0.001], (d - 1) // 2 * [0.001], [0]))

    return x[1:]

def layer_coords_general(maximals, k):
    v = max(sum(maximals, start=())) + 1
    d = v - 4
    x = v * [0]
    start = 0

    base = base_coords(k)
    print(base)

    for i, s in enumerate(maximals):
        lay = layer_general(len(s), start, d, base[i])
        for j, la in zip(s, lay):
            x[j] = la
        start += len(s) - 1
    return x

def embed_layers_general(maximals, k):
    x = layer_coords_general(maximals, k)
    part = facet_partition(maximals, k)
    faces = sum(part, start=[])

    faces1 = part[-1]
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

    S.faces = faces
    print('intersection volume:', S.intersection_volume_proportion())
    S.faces = faces1
    #S.plot(d, double=faces2)

if __name__ == '__main__':
    maximals = (0,), (1,), (2, 16, 17), (3, 15), (4,), (5, 18), (6, 7), (8,), (11,), (9,), (10,), (12, 13), (14,)
    embed_layers_general(maximals, 13)
