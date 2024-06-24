from sympy import *
from sympy.solvers.solveset import linsolve
from itertools import product

a = ','.join([f'a_{i}{j}{k}' for i, j, k in product('01', repeat=3)])
a = 'a,b,c,d,e,f,g,h'[::-1]

M = Matrix(
    [
        [1,1,1,1,0,0,0,0],
        [0,0,0,0,1,1,1,1],
        [1,1,0,0,1,1,0,0],
        [0,0,1,1,0,0,1,1],
        [1,0,1,0,1,0,1,0],
        [0,1,0,1,0,1,0,1],
        [1,0,0,1,0,1,1,0],
        [0,1,1,0,1,0,0,1]
        ])

b = 8*[1]

s = linsolve((M, b), symbols(a))
s = s.args[0][::-1]

for i in range(2):
    for j in range(2):
        print(s[4*i + 2*j: 4*i + 2*(j+1)])
    print()