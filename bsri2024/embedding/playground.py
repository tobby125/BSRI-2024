import numpy as np
from graph_embedding import Simplicial
from testing import sub1, add1
from math import sqrt
import random
from itertools import product

S = Simplicial(7)

maximals = (3, 6, 9), (1, 3, 5, 7), (2, 4, 6, 8), (4, 6, 8, 1), (5, 7, 9, 2), (7, 9, 2, 4), (8, 1, 3, 5)

maximals = sub1(maximals)

F = 9*[()]
for i in range(7):
    for j in maximals[i]:
        F[j] += (i,)

S.non_faces(F)
S.reduce_faces()

print(add1(S.faces))

'''d = [0, 1, 2, 4, 5, 7, 8]
for i in S.faces:
    print(tuple(d[j] for j in i))'''

x = np.zeros((8, 3))

x[1] = 0, 0, 0

x[2] = 1, 0, -1
x[3] = -1, 0, -1
x[4] = 0, 1, 1
x[7] = 0, -1, 1

x[6] = 0, -1/4, 0
x[5] = 0, 1/4, 0


S.coords = x[1:]

S.plot(3)