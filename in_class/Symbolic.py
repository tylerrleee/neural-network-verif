import numpy as np
import copy 

class SymbolicBound():
    def __init__(self, W_upper, b_upper, W_lower, b_lower, lower, upper ):
        self.sym_upper  = (W_upper, b_upper)
        self.sym_lower  = (W_lower, b_lower)
        self.bounds     = (lower, upper)
    
    def getSymbolicBounds(self, A, b):
        """
        Returns:
            SymbolicBounds
        """

        # -==== Some assertions

        if A is None:
            new_W_upper = copy.deepcopy(self.sym_upper[0])
            new_b_upper = copy.deepcopy(self.sym_upper[1])
            new_W_lower = copy.deepcopy(self.sym_lower[0])
            new_b_lower = copy.deepcopy(self.sym_lower[1])
            if b is not None:
                new_b_upper += b
                new_b_lower += b
        else:
            A_pos = copy.deepcopy(A)
            A_pos[A_pos < 0] = 0 # positive part
            A_neg = copy.deepcopy(A)
            A_neg[A_neg > 0] = 0 # negative part

            new_W_upper = np.matmul(A_pos, self.sym_upper[0]) + np.matmul(A_neg, self.sym_lower[0])
            new_b_upper = np.matmul(A_pos, self.sym_upper[1]) + np.matmul(A_neg, self.sym_lower[1])

            new_W_lower = np.matmul(A_pos, self.sym_lower[0]) + np.matmul(A_neg, self.sym_upper[0])
            new_b_lower = np.matmul(A_pos, self.sym_lower[1]) + np.matmul(A_neg, self.sym_upper[1])
            if b is not None:
                new_b_upper += b
                new_b_lower += b
        
        return SymbolicBound(new_W_upper, new_b_upper, new_W_lower, new_b_lower, self.bounds[0], self.bounds[1])
    
    def getConcreteBounds(self):
            
            b_upper = copy.deepcopy(self.sym_upper[1])
            b_lower = copy.deepcopy(self.sym_lower[1])

            W_upper_pos = copy.deepcopy(self.sym_upper[0])
            W_upper_pos[W_upper_pos < 0] = 0 # Positive part
            W_upper_neg = copy.deepcopy(self.sym_upper[0])
            W_upper_neg[W_upper_neg > 0] = 0 # Negative part

            W_lower_pos = copy.deepcopy(self.sym_lower[0])
            W_lower_pos[W_lower_pos < 0] = 0 # Positive Part
            W_lower_neg = copy.deepcopy(self.sym_lower[0])
            W_lower_neg[W_lower_neg > 0] = 0 # Positive Part

            Z_lower = np.matmul(W_upper, self.bounds[0]) + np.matmul(W_lower, self.bounds[1])
            Z_upper = np.matmul(W_upper, self.bounds[1]) + np.matmul(W_lower, self.bounds[1])
            return Z_lower, Z_upper



    def print_init(self):
        """ print all varbles in a tuple """
        #print(( self.V_plus, self.V_minus, self.c_plus, self.c_minus, self.lower, self.upper ))
        print(f"V+ = {self.W_plus} \n")
        print(f"V- = {self.W_lower} \n")
        print(f"c+ = {self.b_upper} \n")
        print(f"c- = {self.b_low} \n")
        print(f"l = {self.lower} \n")
        print(f"u = {self.upper} \n")
        lower, upper = self.get_concrete_bound()
        print(f"  Concrete bounds: [{lower}, {upper}]")



# TODO RELU