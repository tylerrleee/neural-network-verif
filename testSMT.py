from z3 import *

x, y = Reals('x y')
z11, z12, z13 = Reals('z11 z12 z13')
z21, z22 = Reals('z21 z22')
z3 = Real('z3')

s = Solver()

# Helper function for ReLU encoding
def relu_constraints(z, z_active):
    return [
        Or(z == 0, z == z_active),
        Or(z == z_active, z_active <= 0),
        Or(z == 0, z_active > 0),
    ]

# Layer 1 (input -> hidden1)
z11_active = x - y
z12_active = -x + y
z13_active = x - 2*y

# Layer 2 (hidden1 -> hidden2) - UPDATE THESE BASED ON ACTUAL WEIGHTS
z21_active = 2*z11 + 2*z12 + -3*z13
z22_active = 0.5*z11 + -z12 + -2*z13
  # function of z11, z12, z13

# Collect all ReLU constraints
constraints = []
constraints += relu_constraints(z11, z11_active)
constraints += relu_constraints(z12, z12_active)
constraints += relu_constraints(z13, z13_active)
constraints += relu_constraints(z21, z21_active)
constraints += relu_constraints(z22, z22_active)

# Input bounds
input_constraints = [x >= 0, x <= 1, y >= -1, y <= 1]

# Check each property separately
properties = [z3 <= 8, And(z3 <= 5, z3 >= 2), z3 >= 10]

for i, prop in enumerate(properties):
    s.reset()  # ← Important!
    s.add(constraints + input_constraints)
    s.add(z3 == -z21 + 2*z22)  # output equation
    s.add(prop)
    if s.check() == sat:
        model = s.model()
        print(f"Satisfiable. Model: {str(prop)}")
        print(f"x={model[x]}")
        print(f"y={model[y]}")
    else:
        print(f"Model: {str(prop)} -  Unsatisfiable.")