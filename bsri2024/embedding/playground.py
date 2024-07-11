import numpy as np

x = np.array([
     [0.78699, 0.24726, -0.15856],
     [0.4337, 0.68243, -.5],
     [0, 1.35653, 0.30976],
     [0.73746, 1.23658, 1.09644],
     [0.24891, 0.6016, 2],
     [-0.26699, 0.35056, 0.57124],
     [1.03099, -0.04149, 0.72262]
])

from testing import S, intersection_volume_proportion

S.verts = range(7)
S.non_faces([(0, 5, 3), (1, 6, 4), (2, 0, 5), (3, 1, 6), (4, 2, 0), (5, 3, 1), (6, 4, 2)])
S.reduce_faces()

y = [(0, 1, 2, 3), (0, 1, 2, 6), (0, 1, 3, 4), (0, 1, 4, 5), (0, 1, 5, 6), (0, 2, 3, 6), (0, 3, 4, 6), (0, 4, 5, 6), (1, 2, 3, 4), (1, 2, 4, 5), (1, 2, 5, 6), (2, 3, 4, 5), (2, 3, 5, 6), (3, 4, 5, 6)]

S.faces = [y[i] for i in (0, 1, 3, 5, 7, 9)]
S.coords = x

print(intersection_volume_proportion())
for face in S.faces:
    assert S.is_simplex(face)
S.plot(3)

#print([[round(j, 5) for j in i] for i in x])