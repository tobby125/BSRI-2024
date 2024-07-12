import numpy as np
from graph_embedding import Simplicial, sub1, add1

'''x = np.array([
     [0.78699, 0.24726, -1],
     [0.4, .6, -1],
     [0, 1.35653, 0.30976],
     [0.73746, 1.23658, 1],
     [0.24891, 0.6016, 2],
     [-0.26699, 0.35056, -0.5],
     [0.8, 0.3, 0.5]
])'''

x = np.array([
     [0, 0, 0],
     [1, 0, 0],
     [0, 1, 0],
     [0, 0, 1],
     [1, 1, 1],
     [1, 1, -0.2],
     [0.5, 0.5, -1]
])

from testing import intersection_volume_proportion

S = Simplicial(7)

S.non_faces([(0, 5, 3), (1, 6, 4), (2, 0, 5), (3, 1, 6), (4, 2, 0), (5, 3, 1), (6, 4, 2)])
S.reduce_faces()
S.coords = x
print(intersection_volume_proportion(S))

'''y = [(0, 1, 2, 3), (0, 1, 2, 6), (0, 1, 3, 4), (0, 1, 4, 5), (0, 1, 5, 6), (0, 2, 3, 6), (0, 3, 4, 6), (0, 4, 5, 6), (1, 2, 3, 4), (1, 2, 4, 5), (1, 2, 5, 6), (2, 3, 4, 5), (2, 3, 5, 6), (3, 4, 5, 6)]

S.faces = [y[i] for i in (0, 1, 3, 5, 7, 9)]'''

faces1 = (1, 2, 3, 4), (2, 3, 4, 5), (2, 3, 5, 6), (2, 3, 6, 7), (2, 3, 7, 1)
faces2 = (3, 4, 5, 6), (3, 4, 1, 7), (3, 4, 6, 7), (1, 2, 4, 5), (1, 2, 5, 6), (1, 2, 6, 7)

S.faces = sub1(faces1)
print(intersection_volume_proportion(S))
S.faces = sub1(faces2)
print(intersection_volume_proportion(S))

'''for face in S.faces:
    assert S.is_simplex(face)'''

S.faces = sub1(faces1)
S.plot(3, double=sub1(faces2))

#print([[round(j, 5) for j in i] for i in x])