from sympy import *
import itertools

for i in range(100):
    q = 5
    d = 1
    m = i
    n = ((q - 1)*(d + m + 1) + m*(q**2-1)//2) // 2
    n1 = (q - 1)*(d + m + 1) // 2

    A = list(itertools.product(range(1, (q+1)//2), [0])) + list(itertools.product(range(q), range(1, (q+1)//2)))

    x, y = symbols('x,y')

    poly = Poly(x**n1 * y**n1 * prod(a[0]*x + a[1]*y for a in A)**m)

    poly = poly.as_dict()
    for key in poly:
        poly[key] %= q
    poly = {i: poly[i] for i in poly if poly[i] and i[0] <= n and i[1] <= n}

    if poly:
        print(f'm = {m}', poly)