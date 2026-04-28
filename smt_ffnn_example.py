import z3
import numpy as np
from scipy.optimize import milp, Bounds, LinearConstraint
import time

# 1. Declare real variables
#    z11, z12, z21, z22, z3, x, y

x, y, z11, z12, z21, z22, z3 = Reals('x, y, z11, z12, z21, z22, z3')

# 2. Create a solver instance

s = Solver()

# 3. Add linear constraints

# constraints for z11:
# (z11 == 0 Or z11 == x-y) And (z11 == x-y Or x-y <= 0) And (z11 == 0 Or x-y > 0) And (x-y <= 0 Or x-y>0)

c1 = Or(z11 == 0, z11 == x-y)
c2 = Or(z11 == x-y, x-y <= 0)
c3 = Or(z11 == 0, x-y > 0)
c4 = Or(x-y <= 0, x-y > 0)


# constraints for z12:
# (z12 == 0 Or z12 == -x + 2y) And (z12 == -x + 2y Or -x + 2y <= 0) And (z12 == 0 Or -x +2y > 0) And (-x + 2y <= 0 Or -x + 2y > 0)

c5 = Or(z12 == 0, z12 == -x + 2*y)
c6 = Or(z12 == -x + 2*y, -x + 2*y <= 0)
c7 = Or(z12 == 0, -x + 2*y > 0)
c8 = Or(-x + 2*y <= 0, -x + 2*y > 0)

# constraints for z21:
# (z21 == 0 Or z21 == z11 + z12) And (z21 == z11 + z12 Or z11 + z12 <= 0) And (z21 == 0 Or z11 + z12 > 0) And (z11 + z12 <= 0 Or z11 + z12 > 0)

c9 = Or(z21 == 0, z21 == z11 + z12)
c10 = Or(z21 == z11 + z12, z11 + z12 <= 0)
c11 = Or(z21 == 0, z11 + z12 > 0)
c12 = Or(z11 + z12 <= 0, z11 + z12 > 0)

# constraints for z22:
# (z22 == 0 Or z22 = -3z11 + 2z12) And (z22 = -3z11 + 2z12 Or -3z11 + 2z12 <= 0) And (z22 == 0 Or -3z11 + 2z12 > 0) And (-3z11 + 2z12 <= 0 Or -3z11 + 2z12 > 0)

c13 = Or(z22 == 0, z22 == -3*z11 + 2*z12)
c14 = Or(z22 == -3*z11 + 2*z12, -3*z11 + 2*z12 <=0)
c15 = Or(z22 == 0, -3*z11 + 2*z12 > 0)
c16 = Or(-3*z11 + 2*z12 <= 0, -3*z11 + 2*z12 > 0)

# constraints for z3:
# z3 = -z21 + z22

# constraints for inputs:
# x <=2 And x >= -1, y <= 2 And y >= 1
in1 = And(x <= 2, x >= -1)
in2 = And(y <= 2, y >= 1)

# properties: z3 >= 3

# combine all together ...

properties = [z3 <= 8 ,     
              z3 <= 5 and z3 >= 2, 
              z3 >= 10
              ]


# 4. Check for satisfiability and get the model
for i, prop in enumerate(properties):
    s.add(c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, c16)   # ReLU node constraints
    s.add(z3 == -z21 + 2*z22)   # output
    s.add(in1, in2) # input constraints
    s.add(prop)  # verification property
    print(prop)
    if s.check() == sat:
        model = s.model()
        print(f"Satisfiable. Model {i}: {str(prop)}")
        print(f"x={model[x]}")
        print(f"y={model[y]}")
    else:
        print(f"Model {i}: {str(prop)} -  Unsatisfiable.")



