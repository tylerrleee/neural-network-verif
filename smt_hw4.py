from z3 import *
import numpy as np
from scipy.optimize import milp, Bounds, LinearConstraint
import time

# code structure from smt_fnn.py from Files

# 1. Declare real variables

x, y, z11, z12, z13, z21, z22, z3 = Reals('x, y, z11, z12, z13, z21, z22, z3')

# 2. Create a solver instance

s = Solver()

# 3. Add linear constraints

# constraints for z11:
z11_active = x - y

c1 = Or(z11 == 0, z11 == z11_active)
c2 = Or(z11 == z11_active, z11_active <= 0)
c3 = Or(z11 == 0, z11_active > 0)
c4 = Or(z11_active <= 0, z11_active > 0)


# constraints for z12:
# -x + y
z12_active = - x + y

c5 = Or(z12 == 0, z12 == z12_active)
c6 = Or(z12 == z12_active, z12_active <= 0)
c7 = Or(z12 == 0, z12_active > 0)
c8 = Or(z12_active <= 0, z12_active > 0)

# constraints for z13:
z13_active = x - 2*y

c9 = Or(z13 == 0, z13 == z13_active)
c10 = Or(z13 == z13_active, z13_active <= 0)
c11 = Or(z13 == 0, z13_active > 0)
c12 = Or(z13_active <= 0, z13_active > 0)

# constraints for z21:
z21_active = 2*z11 + 2*z12 + -3*z13

c13 = Or(z21 == 0, z21 == z21_active)
c14 = Or(z21 == z21_active, z21_active <=0)
c15 = Or(z21 == 0, z21_active > 0)
c16 = Or(z21_active <= 0, z21_active > 0)

# constraints for z22:
z22_active = 0.5*z11 + -z12 + -2*z13

c17 = Or(z22 == 0, z22 == z22_active)
c18 = Or(z22 == z22_active, z22_active <=0)
c19 = Or(z22 == 0, z22_active > 0)
c20 = Or(z22_active <= 0, z22_active > 0)

# constraints for z3:
# z3 = -z21 + 2*z22

# constraints for inputs:
# x = [0, 1]
# y = [-1, 1]

in1 = And(x <= 1, x >= 0)
in2 = And(y <= 1, y >= -1)


# combine all together ...

# 4. Check for satisfiability and get the model
properties = [z3 <= 8 ,     
              And(z3 <= 5, z3 >= 2), 
              z3 >= 10
              ]


# 4. Check for satisfiability and get the model
for i, prop in enumerate(properties):
    s.reset()
    s.add(c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, c16, c17, c18, c19, c20)   # ReLU node constraints
    s.add(z3 == -z21 + 2*z22)   # output
    s.add(in1, in2) # input constraints
    s.add(prop)  # verification property
    if s.check() == sat:
        model = s.model()
        print(f"Satisfiable. Model: {str(prop)}")
        print(f"x={model[x]}")
        print(f"y={model[y]}")
    else:
        print(f"Model: {str(prop)} -  Unsatisfiable.")

    print('-' * 10) # for readability


"""
Initial hypothesis:
From Homework 3, we know the results using MILP approach is ABOUT [-2, 0]. Hence:
- property 1 would be Satisfiable
- property 2 would be unsatisable 
- property 2 would be unsatisable 

Satisfiable. Model: z3 <= 8
x=1/2 ; yes, within [0,1]
y=5/22 ; yes, within [-1, 1]
----------
Model: And(z3 <= 5, z3 >= 2) -  Unsatisfiable.
----------
Model: z3 >= 10 -  Unsatisfiable.
----------

Our hypotheis was right, only property 1 satisfies. 
"""