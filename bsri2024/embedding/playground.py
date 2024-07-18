import numpy as np
from graph_embedding import Simplicial
from testing import sub1, add1
from math import sqrt
import random
from itertools import product

maximals = (3, 6, 9), (2, 5, 8), (1, 4, 7), (1, 3, 5, 7), (2, 4, 6, 8), (3, 5, 7, 9), (4, 6, 8, 1), (5, 7, 9, 2), (6, 8, 1, 3), (7, 9, 2, 4), (8, 1, 3, 5), (9, 2, 4, 6)

v = len(maximals)
S = Simplicial(v)

maximals = sub1(maximals)

F = 9*[()]
for i, x in enumerate(maximals):
    for j in x:
        F[j] += (i,)

S.non_faces(F)
S.reduce_faces()

print(add1(F))
print(add1(S.faces))

x = np.zeros((9, 4))

x[1] = 0, 0, 0, -1
x[8] = 0, 0, 0, 1

x[5] = 1, 0, 0, 0
x[6] = -1, 0, 0, 0

x[2] = 0, 1, 0, 0
x[7] = 0, -1, 0, 0

x[3] = 0, 0, 1, 0
x[4] = 0, 0, -1, 0


S.coords = x[1:]

faces1 = (1, 2, 3, 5, 8), (1, 2, 3, 6, 8), (1, 2, 4, 5, 8), (1, 2, 4, 6, 8), (1, 3, 5, 7, 8), (1, 3, 6, 7, 8), (1, 4, 5, 7, 8), (1, 4, 6, 7, 8)
faces2 = (2, 3, 4, 5, 8), (2, 3, 6, 7, 8), (4, 5, 6, 7, 8)

S.faces = sub1(faces1)
print(S.check_embedding())

S.faces = sub1(faces2)
print(S.check_embedding(True))
print(S.intersection_volume_proportion())

S.plot(4)