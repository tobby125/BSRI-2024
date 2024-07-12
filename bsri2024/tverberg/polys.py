import math
from itertools import product

def numberToBase(n, b):
    if n == 0:
        return [0]
    digits = []
    while n:
        digits.append(int(n % b))
        n //= b
    return digits[::-1]
class Poly:
    def __init__(self, mod, order, values=None):
        self.mod = mod
        self.order = order
        if values is None:
            self.values = {}
        else:
            self.values = values

    def __add__(self, other):
        assert (self.mod, self.order) == (other.mod, other.order)
        p = Poly(self.mod, self.order)
        p.values = self.values.copy()
        for i, j in other.values:
            if (i, j) in p.values:
                p.values[i, j] = (p.values[i, j] + other.values[i, j]) % mod
                if p.values[i, j] == 0:
                    p.values.pop((i, j))
            else:
                p.values[i, j] = other.values[i, j]
        return p

    def __mul__(self, other):
        if not isinstance(other, Poly):
            p = Poly(self.mod, self.order)
            p.values = self.values.copy()
            for i, j in self.values:
                p.values[i, j] = (other * self.values[i, j]) % mod
                if p.values[i, j] == 0:
                    p.values.pop((i, j))
            return p

        assert (self.mod, self.order) == (other.mod, other.order)
        p = Poly(self.mod, self.order)
        for i, j in self.values:
            for i_, j_ in other.values:
                if i + i_ < order and j + j_ < order:
                    if (i + i_, j + j_) in p.values:
                        p.values[i + i_, j + j_] += self.values[i, j] * other.values[i_, j_]
                    else:
                        p.values[i + i_, j + j_] = self.values[i, j] * other.values[i_, j_]
        for i, j in list(p.values):
            p.values[i, j] %= mod
            if p.values[i, j] == 0:
                p.values.pop((i, j))
        return p

    def __rmul__(self, other):
        return self * other

    def __pow__(self, power):
        if power == 0:
            p = Poly(mod, order)
            p.values[0, 0] = 1
            return p
        if power >= self.order:
            return Poly(mod, order)
        if power == 1:
            return self
        new_power = power // 2
        p = self**new_power
        p = p * p
        if power % 2 == 1:
            p *= self
        return p

    def __str__(self):
        return str(self.values)

prints = 0
for i in range(1000):
    q = 11
    d = 1
    m = i
    n = ((q - 1)*(d + m + 1) + m*(q**2-1)//2) // 2
    n1 = (q - 1)*(d + m + 1) // 2

    A = list(product(range(1, (q+1)//2), [0])) + list(product(range(q), range(1, (q+1)//2)))

    mod = q
    order = n + 1
    x = Poly(mod, order)
    x.values[1, 0] = 1
    y = Poly(mod, order)
    y.values[0, 1] = 1

    poly = x**n1 * y**n1 * math.prod(a[0]*x + a[1]*y for a in A)**m * (math.factorial(q-1)**n1)

    if poly.values:
        prints += 1

        print(f'm = {m}, n = {n}')
        print(f'\t\t{poly}')