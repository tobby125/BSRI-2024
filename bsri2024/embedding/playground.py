import numpy as np
from graph_embedding import Simplicial, sub1, add1

F = (0, 3, 5), (1, 4, 6, 7), (0, 2, 5), (1, 3, 6), (0, 2, 4, 7), (1, 3, 5), (0, 2, 4, 6), (1, 3, 5, 7), (2, 4, 6)
F = (0, 1, 2), (3, 4, 5), (1, 2, 6), (0, 3, 5), (1, 2, 4), (0, 3, 6), (1, 4, 5), (0, 2, 3), (4, 5, 6)

m = [[]] + [np.array([i, i+2, i+4, i+6]) % 9 for i in range(9)] + [np.array([i, i+3, i+6]) % 9 for i in range(3)]
for i in range(len(m)):
     m[i] = sorted(m[i])

for i in range(9):
     mm = [j for j in range(9) if i in F[j]]
    # print(m.index(mm))

for i, f in enumerate(F):
     for j, f_ in enumerate(F):
          if j in [(i - 1) % len(F), (i + 1) % len(F)]:
               assert not set(f) & set(f_)
          else:
               assert set(f) & set(f_)

S = Simplicial(7)

faces = (1, 2, 4, 5), (1, 2, 5, 7), (1, 2, 6, 7), (1, 3, 5, 6), (1, 3, 5, 7), (1, 3, 6, 7), (2, 3, 4, 6), (2, 4, 5, 7), (2, 4, 6, 7), (3, 4, 5, 7), (3, 4, 6, 7)

S.faces = sub1(faces)
print(add1(S.faces))

x = 8*[(0, 0, 0)]

x[1] = 0, 0, 0
x[2] = 1, 0, 0
x[3] = 1.2, -0.14, 1.12
x[4] = 0, 1, 0
x[5] = 0, 0, 1
x[6] = 1.16, -0.18, 0.92
x[7] = 1.02, -0.09, 0.83

S.coords = np.array(x[1:])

from testing import intersection_volume_proportion
print(intersection_volume_proportion(S))
print(S.check_embedding())
S.plot(3)
quit()




y = np.array([
     [0.78699, 0.24726, -1, 1, 0, 0],
     [0.4, .6, -1, 0, 1, 0],
     [0, 1.35653, 0.30976, 0, 0, 0],
     [0.73746, 1.23658, 1, 0, 0, 0],
     [0.24891, 0.6016, 2, 0, 0, 0],
     [-0.26699, 0.35056, -0.5, 0, 0, 0],
     [0.8, 0.3, 0.5, 0, 0, 0],
     [0.78699, 0.24726, -1, -1, 0, 0],
     [0.4, .6, -1, 0, -1, 1],
     [0.4, .6, -1, 0, -1, -1]
])

y = np.array([
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

maximals = (0,), (1,), (2,), (3,), (4,), (5,), (6,)
from graph_embedding import facet_partition
S.faces = sum(facet_partition(maximals, 7), start=[])

S.coords = y
print(intersection_volume_proportion(S))
print(len(S.faces), S.faces)

faces1 = (1, 2, 6, 7), (4, 5, 6, 7), (2, 3, 6, 7), (3, 4, 6, 7), (3, 4, 5, 6), (1, 2, 3, 4), (3, 4, 7, 1), (2, 3, 7, 1)
faces2 = (1, 2, 4, 5), (1, 2, 5, 6), (2, 3, 4, 5), (2, 3, 5, 6), (4, 5, 7, 1), (5, 6, 7, 1)

S.faces = sub1(faces1)
print(intersection_volume_proportion(S))
S.faces = sub1(faces2)
print(S.faces)
print(intersection_volume_proportion(S))

for face in S.faces:
    assert S.is_simplex(face)

S.faces = sub1(faces1)
S.plot(3, double=sub1(faces2))

#print([[round(j, 5) for j in i] for i in x])