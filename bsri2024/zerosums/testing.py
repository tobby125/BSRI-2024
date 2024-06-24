from tensors import *

m = 3
n = 3
M = Tensor(m, n)

M.vals = np.array(
    [
        [[1/3, 1/3, 0],
         [0, 0, 0],
         [0, 0, 1/2]],

        [[0, 1/3, 0],
         [0, 0, 0],
         [0, 0, 1/3]],

        [[0, 0, 0],
         [0, 0, 0],
         [0, 1/3, 0]]
    ]
)

def rand_half():
    vals = []
    for i in range(3):
        r, s = random.choice(range(9)), random.choice(range(8))
        mm = [3*[0] for _ in range(3)]
        mm[r // 3][r % 3] = 1/2
        mm[(r + s + 1) % 9 // 3][(r + s + 1) % 9 % 3] = 1/2
        vals.append(mm)
    M.vals = np.array(vals)

perps = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 2)]
for p in perps:
    M.hyperplanes += M.cosets(M.perp(p))

print(M.hyperplanes)
print(M.h_matrix())

a, b, c, d, e, f, g, h = 0,0,0,.1,.2,0,.1,.3

i = -a - b - c - d - e - f - g - h + 1

j, k, l, m, n, o, p, q = 0,.2,0,0,.4,0,0,.4

r = -j - k - l - m - n - o - p - q + 1

s, t = .1,0

u = -a - b - c - j - k - l - s - t + 1
v = b + c + h - m - o - p + t
w = -a - 2*b - c - d - e - f - g - h - j - k - l - m - n - q - s - t + 2
x = a + b + g + j + k + l + m + p + q + s - 1
y = -a - b - c - d - g - h - j + o - s - t + 1
z = a + b + c + d + f + g + j + l + m + s - 1
aa = a + b + c + d + e + h + j + k + n + t - 1


M.vals = np.array(
    [
        [[a, b, c],
         [d, e, f],
         [g, h, i]],

        [[j, k, l],
         [m, n, o],
         [p, q, r]],

        [[s, t, u],
         [v, w, x],
         [y, z, aa]]
    ]
)

print(M)
for i in M.hyperplanes[-3]:
    print(i)

print(M.h_sum())