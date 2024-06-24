import numpy as np
from scipy.optimize import linprog
import random

M = np.array([
        [1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1],
        [1,1,1,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0],
        [0,0,0,1,1,1,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,1,1,1,0,0,0],
        [0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,1,1,1],
        [1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0],
        [0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0],
        [0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1],
        [1,0,0,0,1,0,0,0,1,0,1,0,0,0,1,1,0,0,0,0,1,1,0,0,0,1,0],
        [0,0,1,1,0,0,0,1,0,1,0,0,0,1,0,0,0,1,0,1,0,0,0,1,1,0,0],
        [0,1,0,0,0,1,1,0,0,0,0,1,1,0,0,0,1,0,1,0,0,0,1,0,0,0,1]
        ])

# Coefficients of the equations Ax = b
A_eq = np.array(M)  # Coefficient matrix for equations
b_eq = np.array(12*[1])            # Constants for equations

# Coefficients of the inequalities Ax <= b
A_ineq = np.eye(A_eq.shape[1]) * -1  # Identity matrix multiplied by -1
b_ineq = [random.choice([-round(.3*random.random(), 2), 0, 0, 0]) for _ in range(27)]
print(b_ineq)

# Solve the linear programming problem
result = linprog(c=np.array(27*[7]), A_eq=A_eq, b_eq=b_eq, A_ub=A_ineq, b_ub=b_ineq)

# Extract the solution
if result.success:
    solution = result.x
    for i in range(9):
        print(solution[3*i:3*i+3])
        if i%3==2:
            print()
    print(solution)
    print(M @ solution)
else:
    print("No solution found.")
