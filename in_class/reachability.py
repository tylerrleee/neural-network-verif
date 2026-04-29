import numpy as np
import copy
from scipy.optimize import linprog



class HyperBox(object):
    """
        HyperBox Class for reachability
        author: Dung Tran
        date: 3/07/2026
        Representation of a HyperBox
        ==========================================================================
        A hyperbox is represented using lower bound l and upper bound u vectors

        B = <l, u>, x \in B -> l[i] <= x[i] <= u[i]
        ==========================================================================
    """


    def __init__(self, lb_vec, ub_vec):
        """
           Key attributes:
           lb ; % lower bound vector
           ub ; % upper bound vector
           dim; % dimension 
        
        """
        if np.any(ub_vec - lb_vec < 0):
            raise ValueError('the lower bound contains vvalues strictly larger than the associated upper bound value')
        
        assert isinstance(lb_vec, np.ndarray) and lb_vec.ndim == 1, \
            "lb_vec must be a 1D numpy array"
        assert isinstance(ub_vec, np.ndarray) and ub_vec.ndim == 1, \
            "ub_vec must be a 1D numpy array"
        
        assert lb_vec.shape == ub_vec.shape, \
            "lb_vec and ub_vec must have the same shape"
        assert np.all(lb_vec <= ub_vec), \
            "lb_vec must be <= ub_vec element-wise"
        
        self.lb : np.ndarray  = lb_vec
        self.ub : np.ndarray = ub_vec
        self.dim : int = lb_vec.shape[0]

    def __str__(self):

        print(f'{self.dim}-dimensional HyperBox:\n')
        print(f'lower bound vector: {self.lb}')
        print(f'\nupper bound vector: {self.ub}')


    def affineMap(self, A = None, b = None):
        """
        Affine mapping of a hyper box: X = A*self + b

        Args:
            A (np.ndarray): Mapping matrix (optional)
            b (np.ndarray): Offset vector (optional)

        Returns:
            HyperBox: New HyperBox overapprimating the exact set X
        
        """
        lb_new = self.lb.copy()
        ub_new = self.ub.copy()

        if A is not None:
            A_pos = np.maximum(A, 0.0) # W+
            A_neg = np.minimum(A, 0.0) # W-
            lb_new = A_pos @ self.lb + A_neg @ self.ub 
            ub_new = A_pos @ self.ub + A_neg @ self.lb
        
        if b is not None:
            lb_new += b
            ub_new += b
        
        return HyperBox(lb_new,ub_new)
    
        
    def intersect(self, alpha, gamma):
        """
        intersection with a halfspace defined by: H : alpha*x <= gamma: a1*x[1] + a2*x[2] + ...+ ai*x[i] <= gamma

        X = B intersect H

        Args: 
            alpha = [a1, ..., ai]
            gamma

        Return:
            A new hyperbox that is the smallest overapproximation of the exact set X
        
        """
        # halfspace = \alpha * x <= gamma
        assert isinstance(alpha, np.ndarray), \
            "lb_vec must be a  numpy array"

        lb_new = self.lb.copy()
        ub_new = self.ub.copy()

        alpha_pos = np.maximum(alpha, 0.0)
        alpha_neg = np.minimum(alpha, 0.0)

        max_val = alpha_pos @ self.ub + alpha_neg @ self.lb   # max of alpha·x in box
        min_val = alpha_pos @ self.lb + alpha_neg @ self.ub   # min of alpha*x in box

        if max_val <= gamma:
            return HyperBox(lb_new, ub_new)
        
        # Not within halfspace -> no intersect
        if min_val > gamma:
            return None

        # Tighten bounds by halfspace dim
        for i in range(self.dim):
            if alpha[i] == 0.0:
                continue
        
            rest_min = alpha_pos @ self.lb + alpha_neg @ self.ub \
                       - (alpha_pos[i] * self.lb[i] + alpha_neg[i] * self.ub[i])
            
            rest_max = alpha_pos @ self.ub + alpha_neg @ self.lb \
                       - (alpha_pos[i] * self.ub[i] + alpha_neg[i] * self.lb[i])

            if alpha[i] > 0.0:
                new_ub = (gamma - rest_min) / alpha[i]
                ub_new[i] = min(ub_new[i], new_ub)
            else:
                new_lb = (gamma - rest_max) / alpha[i]
                lb_new[i] = max(lb_new[i], new_lb)
        
        if np.any(lb_new > ub_new + 1e-9):
            return None
        
        return HyperBox(lb_new, ub_new)
   
    def ReLU(self):
        """
        apply ReLU activation on a hyperbox
        Return another hyperbox after ReLU
        y = ReLU(x) = max(x,0)
        """
        # clip anyhing < 0 to 0, else unchanged bound
        lb_new = np.maximum(self.lb, 0.0)
        ub_new = np.maximum(self.ub, 0.0)
        return HyperBox(lb_new, ub_new)

    def propagate(self, network):
        """
        propagate a hyperbox through a network
        construct a hyperbox output set

        network: a list of affinemap and ReLU operations

        """
        assert isinstance(network, list), 'error: network should be a list of operations'
        
        reachSet = [self]
        current = self
        
        for operation in network:
            if operation.operationType == 'affineMap':
                W, b = operation.operationPara
                current = current.affineMap(A = W, b = b)
            elif operation.operationType == 'relu':
                current = current.ReLU()
            
            else:
                raise ValueError(f"Unknown operation... {operation.operationType}")
            reachSet.append(current)

        return reachSet

class SymbolicBound(object):
    """
    SymbolicBound Class for reachability
        author: Dung Tran
        date: 3/07/2026
        Representation of a SymbolicBound
        ===================================================================================================
       
        S = <(alpha_ub,beta_ub), (alpha_lb, beta_lb), l, u>,

        y in S -->      alpha_lb * x + beta_lb <= y <=  alpha_ub * x + beta_ub, where, l[i] <= x[i] <= u[i] 
        ===================================================================================================
    """

    def __init__(self, W_ub, b_ub, W_lb, b_lb, lb, ub):
        self.W_ub = np.array(W_ub, dtype=np.float32)
        self.W_lb = np.array(W_lb, dtype = np.float32)
        self.b_ub = np.array(b_ub, dtype = np.float32)
        self.b_lb = np.array(b_lb, dtype = np.float32)
        self.lb   = np.array(lb, dtype = np.float32)
        self.ub   = np.array(ub, dtype = np.float32)
        self.in_dim = lb.shape[0] # n input dim
        self.out_dim = b_ub.shape[0] # m output dim
    
    @staticmethod
    def init_from_box(lb, ub):
        """
        Construct identity Symbolic Bound from Hyperbox, Star input
        """

        n = lb.shape[0]
        I = np.eye(n) # n x n
        b = np.zeros(n)
        return SymbolicBound(I, b, I, b, lb, ub)

    def __str__(self):

        print(f'SymbolicBound (in_dim={self.in_dim}, out_dim={self.out_dim}):')
        print(f'  W_ub =\n{self.W_ub}')
        print(f'  b_ub = {self.b_ub}')
        print(f'  W_lb =\n{self.W_lb}')
        print(f'  b_lb = {self.b_lb}')
        print(f'  input lb = {self.lb}')
        print(f'  input ub = {self.ub}')

    def affineMap(self, W, b):
        """
        Get Affine map y' = W*y + b
        """

        W_pos = np.maximum(W, 0.0)
        W_neg = np.minimum(W, 0.0)

        new_W_ub = W_pos @ self.W_ub + W_neg @ self.W_lb  
        new_b_ub = W_pos @ self.b_ub + W_neg @ self.b_lb + b

        new_W_lb = W_pos @ self.W_lb + W_neg @ self.W_ub   # (k, n)
        new_b_lb = W_pos @ self.b_lb + W_neg @ self.b_ub + b

        return SymbolicBound(new_W_ub, new_b_ub, new_W_lb, new_b_lb, self.lb, self.ub)

    def getConcreteBound(self):
        """
        Compute concrete lower + upper bounds from symbolic bounds
        """
        # 
        W_ub_pos = np.maximum(self.W_ub, 0.0) 
        W_ub_neg = np.minimum(self.W_ub, 0.0)
        Z_c_ub = W_ub_pos @ self.ub + W_ub_neg @ self.lb + self.b_ub
 
        W_lb_pos = np.maximum(self.W_lb, 0.0)
        W_lb_neg = np.minimum(self.W_lb, 0.0)
        Z_c_lb = W_lb_pos @ self.lb + W_lb_neg @ self.ub + self.b_lb
 
        return Z_c_lb, Z_c_ub
    
    def ReLU(self):
        """
        Code from HW 1-2
        Apply ReLU to symbolic bound where there are 3 cases:
        1. Always active
        2. Always inactive
        3. Cross zero
        """
        z_lb, z_ub = self.getConcreteBound()   
        
        m, n = self.W_ub.shape
        new_W_ub = np.zeros((m, n))
        new_b_ub = np.zeros(m)
        new_W_lb = np.zeros((m, n))
        new_b_lb = np.zeros(m)

        for i in range(m):
            l_i = z_lb[i]
            u_i = z_ub[i]

            # All bounds are POSITIVE
            if l_i >= 0.0:
                
                new_W_ub[i, :] = self.W_ub[i, :]
                new_b_ub[i]    = self.b_ub[i]
                new_W_lb[i, :] = self.W_lb[i, :]
                new_b_lb[i]    = self.b_lb[i]

            # All bounds are NEGATIVE 
            elif u_i <= 0.0:
                pass
                
            # Lb  negative & Ub positive
            else:
                s = u_i / (u_i - l_i)

                new_W_ub[i, :] = s * self.W_ub[i, :]
                new_b_ub[i]    = s * self.b_ub[i] - s * l_i

                # new lower bound and b_lb stays zero from np.zeros

        return SymbolicBound(new_W_ub, new_b_ub, new_W_lb, new_b_lb, self.lb, self.ub)

    def propagate(self, network):
        """
        propagate Symbolic Bound forward
        """

        reachSets = [self]
        current   = self

        for operation in network:
            if operation.operationType == 'affineMap':
                W, b = operation.operationPara
                current = current.affineMap(W, b)
            elif operation.operationType == 'relu':
                current = current.ReLU()
            
            else:
                raise ValueError(f"Unknown operation... {operation}")

            reachSets.append(current)
        return reachSets
    
    
class Zonotope(object):
    """
    A Zonotope Class for reachability
    """

    def __init__(self, c , V):
        assert isinstance(c, np.ndarray), \
            "c must be a numpy array"
        assert isinstance(V, np.ndarray), \
            "V must be a numpy array"

        if V.ndim == 1:
            V = V.reshape(-1, 1)
        assert c.shape[0] == V.shape[0], "c and V must share ambient dimension d"

        self.c      = c
        self.V      = V
        self.dim    = c.shape[0]
        self.n_gen  = V.shape[1]

    def __str__(self):
        print(f'Zonotope (dim={self.dim}, generators={self.n_gen}):')
        print(f'  center c = {self.c}')
        print(f'  generator matrix V =\n{self.V}')


    @staticmethod
    def from_box_to_zonotope(lb, ub):
        """
        Convert Hyperbox {lb, ub} to a Zonotope
        """
        assert isinstance(lb, np.ndarray), \
            "lb must be a numpy array"
        assert isinstance(ub, np.ndarray), \
            "ub must be a numpy array"
        
        c = (lb + ub) / 2.0
        V = np.diag((ub - lb) / 2.0)
        return Zonotope(c, V)
    
    def getConcreteBound(self):

        radius = np.sum(np.abs(self.V), axis = 1)
        z_lb, z_ub = self.c - radius, self.c + radius
        return z_lb, z_ub
    
    def affineMap(self, A, b):
        """
        Z' = A * Z + b
        c' = A * c + b
        V' = A * V
        """
        assert isinstance(A, np.ndarray) and A.ndim == 2, \
            "A must be a 2D numpy array"
        assert isinstance(b, np.ndarray), \
            "b must be a numpy array"
        
        c_p = A @ self.c + b
        V_p = A @ self.V

        return Zonotope(c_p, V_p)
    
    def ReLU(self):
        """
        Zonotope overapproximation
        """
        lb, ub = self.getConcreteBound()
        d, m   = self.V.shape
    
        new_c = np.zeros(d)
        new_V = np.zeros((d,m))
        xcols = []

        for i in range(d):
            l_i, u_i = lb[i], ub[i]

            # All active
            if l_i >= 0.0:
                new_c[i] = self.c[i]
                new_V[i, :] = self.V[i, :]
            
            elif u_i <= 0:
                pass
            else:
                s  = u_i / (u_i - l_i)
                dX = -u_i * l_i / (u_i - l_i) 

                new_c[i]    = s * self.c[i] + dX / 2.0
                new_V[i, :] = s * self.V[i, :]
 
                col     = np.zeros(d)
                col[i]  = dX / 2.0
                xcols.append(col)
        
        if xcols:
            new_V = np.hstack([new_V, np.column_stack(xcols)])

        return Zonotope(new_c, new_V)
    
    def propagate(self, network):
        """
        return list of Zonotope objects
        """

        reachSets = [self]
        current   = self
        for operation in network:
            if operation.operationType == 'affineMap':
                W, b = operation.operationPara
                current = current.affineMap(W, b)
            elif operation.operationType == 'relu':
                current = current.ReLU()
            else:
                raise ValueError(f"Unknown operation type: {operation.operationType}")
            
            reachSets.append(current)

        return reachSets
        


class Star(object):
    """
    A Star class for reachability
    """

    def __init__(self, c , V , C = None, d_vec = None, lb = None, ub = None):
        if V.ndim == 1:
            V = V.reshape(-1, 1)
        
        assert isinstance(c, np.ndarray), "c must be a numpy array"
        assert isinstance(V, np.ndarray), "V must be a numpy array"
        assert c.shape[0] == V.shape[0], \
            f"c (dim {c.shape[0]}) and V (rows {V.shape[0]}) must share dimension d"
        
        d, m = V.shape
        self.c = c
        self.V = V
        self.dim = d
        self.m = m #number of predicates

        if C is None:
            self.C = np.zeros((0, m))
            self.d_vec = np.zeros(0)
        
        else:
            self.C = C
            self.d_vec = d_vec
        
        if lb is None:
            self.lb = np.full(m, -np.inf)
        else:
            self.lb = lb
        
        if ub is None:
            self.ub = np.full(m, np.inf)
        else:
            self.ub = ub
    
    def __str__(self):
        print(f'Star (dim={self.dim}, generators={self.m}):')
        print(f'  center c    = {self.c}')
        print(f'  generator V =\n{self.V}')
        print(f'  predicate C =\n{self.C}')
        print(f'  predicate d = {self.d_vec}')
        print(f'  alpha lb    = {self.lb}')
        print(f'  alpha ub    = {self.ub}')

    @staticmethod
    def from_box(lb_x, ub_x):

        assert isinstance(lb_x, np.ndarray), "lb_x must be a numpy array"
        assert isinstance(ub_x, np.ndarray), "lb_ub_xx must be a numpy array"

        c       = (lb_x + ub_x) / 2.0
        V       = np.diag((ub_x - lb_x) / 2.0)
        lb_alpha = np.full(len(lb_x), -1.0)
        ub_alpha = np.full(len(ub_x),  1.0)
        return Star(c, V, lb=lb_alpha, ub=ub_alpha)

    def affineMap(self, W, b):
        assert isinstance(W, np.ndarray), "W must be a numpy array"
        assert isinstance(b, np.ndarray), "b must be a numpy array"

        new_c = W @ self.c + b
        new_V = W @ self.V

        return Star(new_c, new_V,
                    C = self.C.copy() if self.C.shape[0] > 0 else None,
                    d_vec = self.d_vec.copy() if self.C.shape[0] > 0 else None,
                    lb = self.lb.copy(), ub=self.ub.copy())

    def intersect(self, G, g):
        """
        Intersect Star with halfspaces
        phi ∩ H  =  <c, V, P and P_H>
            where P_H(alpha) = (G*V)*alpha <= g - G*c
        """
        assert isinstance(G, np.ndarray), "G must be a numpy array"
        assert isinstance(g, np.ndarray), "g must be a numpy array"

        C_new = G @ self.V
        d_new = g - G @ self.c

        if self.C.shape[0] > 0:
            C_combined = np.vstack([self.C, C_new])
            d_combined = np.concatenate([self.d_vec, d_new])
        
        else:
            C_combined = C_new
            d_combined = d_new
        
        return Star(self.c.copy(), self.V.copy(),
                    C = C_combined, d_vec = d_combined,
                    lb = self.lb.copy(), ub = self.ub.copy()) 
    
    def getEstimatedBounds(self):
        """
        Over-approximate Star bounds using predicate
        """

        # Clap infinite to finite values
        lb_a = np.where(np.isfinite(self.lb), self.lb, -1.0)
        ub_a = np.where(np.isfinite(self.ub), self.ub,  1.0)

        V_pos = np.maximum(self.V, 0.0)
        V_neg = np.minimum(self.V, 0.0)

        lb_x = self.c + V_pos @ lb_a + V_neg @ ub_a
        ub_x = self.c + V_pos @ ub_a + V_neg @ lb_a

        return lb_x, ub_x
    
    def getExactBound(self):
        """
        Exact bound of Star through Linear compute
        Returns:
            (lb, ub): exact lower and upper bound vectors, each shape (d,)

        """
        d, m = self.dim, self.m
        lb_x, ub_x = np.zeros(d), np.zeros(d)

        # Build bound tuple for linprog
        alpha_bounds = list(zip(
            np.where(np.isfinite(self.lb), self.lb, -1e9),
            np.where(np.isfinite(self.ub), self.ub,  1e9)
        ))

        # Constraints
        A_ub = self.C if self.C.shape[0] > 0 else None
        b_ub = self.d_vec if self.C.shape[0] > 0 else None

        for j in range(d):
            c_obj = self.V[j, :] # obj coefficients
 
            # Minimise
            res_min = linprog(c_obj, A_ub=A_ub, b_ub=b_ub,
                              bounds=alpha_bounds, method='highs')
            lb_x[j] = self.c[j] + res_min.fun if res_min.success else -np.inf
 
            # Maximise  (negate obj)
            res_max = linprog(-c_obj, A_ub=A_ub, b_ub=b_ub,
                              bounds=alpha_bounds, method='highs')
            ub_x[j] = self.c[j] - res_max.fun if res_max.success else np.inf
 
        return lb_x, ub_x
    
    def stepReLU(self, idx):
        """
        Apply ReLU to a neuron 
        When:
            1. All active : star unchanged
            2. All inactive: generate new Star to 0 | predicate unchanged
            3. crossing: Split Star w/ halfspace
        """

        lb_arr, ub_arr = self.getExactBound()
        l_i, u_i = lb_arr[idx], ub_arr[idx]

        if l_i >= 0.0:
            return [self]

        elif u_i <= 0.0:
            new_c = self.c.copy()
            new_c[idx] = 0.0
            new_V = self.V.copy()
            new_V[idx, :] = 0.0
            C_arg = self.C     if self.C.shape[0] > 0 else None
            d_arg = self.d_vec if self.C.shape[0] > 0 else None

            return [Star(new_c, new_V,
                        C = C_arg, d_vec = d_arg,
                         lb = self.lb.copy(), ub = self.ub.copy())]

        else:
            e_i = np.zeros((1, self.dim))
            e_i[0, idx] = 1.0
            # Active branch
            star_active = self.intersect(-e_i, np.array([0.0]))

            # inactive branch
            star_inactive = self.intersect(e_i, np.array([0.0]))

            # Zero dimension idx
            new_c = star_inactive.c.copy()
            new_c[idx] = 0.0
            new_V = star_inactive.V.copy()
            new_V[idx, :] = 0.0
            C_arg = star_inactive.C     if star_inactive.C.shape[0] > 0 else None
            d_arg = star_inactive.d_vec if star_inactive.C.shape[0] > 0 else None
            star_inactive = Star(new_c, new_V, C = C_arg, d_vec = d_arg,
                                 lb = star_inactive.lb.copy(),
                                 ub = star_inactive.ub.copy())
            return [star_active, star_inactive]
        
    def ReLU(self):
            """
            Exact ReLU on Star w/ stepReLU d by d
            """

            current_stars = [self]

            for i in range(self.dim):
                next_stars = []
                for s in current_stars:
                    next_stars.extend(s.stepReLU(i))
                current_stars = next_stars
 
            return current_stars

    @staticmethod
    def propagate(input_star, network):
        """
        Exact Star propagation through a network 
 
        """
        reach_sets = [[input_star]]
        current_stars = [input_star]
 
        for op in network:
            next_stars = []
 
            if op.operationType == 'affineMap':
                W, b = op.operationPara
                for s in current_stars:
                    next_stars.append(s.affineMap(W, b))
 
            elif op.operationType in ('relu', 'ReLU'):
                for s in current_stars:
                    next_stars.extend(s.ReLU())
 
            else:
                raise ValueError(f"Unknown operation type: {op.operationType}")
 
            current_stars = next_stars
            reach_sets.append(current_stars)
 
        return reach_sets

    @staticmethod
    def getOutputBound(star_list):
        """
        Compute exact lb/ub over a list of Stars.
 
        """
        all_lb = []
        all_ub = []
        for s in star_list:
            lb_s, ub_s = s.getExactBound()
            all_lb.append(lb_s)
            all_ub.append(ub_s)

        return np.min(np.stack(all_lb), axis=0), np.max(np.stack(all_ub), axis=0)

class Operation(object):
    """
    Neural network as a set/graph of operations
    """

    def __init__(self, operationType=None, operationPara=None, operationName=None):

        self.operationType = operationType
        self.operationPara = operationPara
        self.operationName = operationName

    @staticmethod
    def rand_ffnn(neurons=[], activations=[]):
        """
        Randomly generate a neural network

        """

        assert len(neurons) >= 3, 'error: list of neurons contain at least 3 values, number of inputs, number of neurons of the first (relu) layers, number of outputs'
        assert len(activations) == len(neurons) - 2, "error: inconsistency between the length of activations and the length of list of neurons"

        operations = []
        for i in range(0, len(neurons)-1):
            W = np.random.rand(neurons[i+1], neurons[i])
            b = np.random.rand(neurons[i+1],)
            operation = Operation('affineMap', (W, b))
            operations.append(operation)
            if i < len(neurons) - 2:
                operation = Operation(activations[i])
                operations.append(operation)

        return operations




class Test(object):
    """
       Testing reachability module
    """

    def __init__(self):

        self.n_fails = 0
        self.n_tests = 0

    def test_HyperBox_constructor(self):

        self.n_tests = self.n_tests + 1

        
        lb_vec = np.array([-1.0, -1.0])
        ub_vec = lb_vec + 2.0
        print('Testing HyperBox Constructor...')
    
        try:
            B = HyperBox(lb_vec, ub_vec)
        except Exception:
            print("Fail in constructing HyberBox object")
            self.n_fails = self.n_fails + 1
        else:
            print("Test Successfull!")

    def test_HyperBox_str(self):

        self.n_tests = self.n_tests + 1

        lb_vec = np.array([-1.0, -1.0])
        ub_vec = lb_vec + 2.0
        
        print('\nTesting __str__ method...')
        B = HyperBox(lb_vec, ub_vec)
        try:
            print(B.__str__())
        except Exception as e :
            print(f"Test Fail :( {e}!")
            self.n_fails = self.n_fails + 1
        else:
            print("Test Successfull!")


    def test_HyperBox_affineMap(self):

        self.n_tests = self.n_tests + 1

        lb_vec = np.array([-1.0, -1.0])
        ub_vec = lb_vec + 2.0
        B = HyperBox(lb_vec, ub_vec)

        A = np.random.rand(2, 2)
        b = np.random.rand(2,)

        print('\nTesting affine mapping method...')

        try:
            B1 = B.affineMap(A, b)
            print('original hyperbox:')
            print(B.__str__())
            print('new hyperbox:')
            print(B1.__str__())
        except Exception:
            print("Test Fail!")
            self.n_fails = self.n_fails + 1
        else:
            print("Test Successfull!")

    def test_rand_ffnn(self):
        
        self.n_tests = self.n_tests + 1

        neurons = [2, 3, 1]
        activations = ['relu']
        
        try:
            operations = Operation.rand_ffnn(neurons, activations)
            print('list of operations belong to the network:')
            for i in range(0, len(operations)):
                print('operations[{}]: {}'.format(i, operations[i].operationType))
        except Exception:
            print("Test Fail!")
            self.n_fails = self.n_fails + 1
        else:
            print("Test Successfull!")

    def test_HyperBox_propagate(self):
        
        self.n_tests = self.n_tests + 1

        # This is the network example from Lecture 4-5 Symbolic Interval & Abstract Interpretation Approaches

        W1 = np.array([[1., -1.], [-1., 2]])
        b1 = np.array([0., 0.])
        affineMap1 = Operation('affineMap', (W1, b1), 'affineMap1')
        relu1 = Operation('relu', operationName='relu1')
        W2 = np.array([[1., 1.],[-3., 2.]])
        b2 = np.array([0., 0.])
        affineMap2 = Operation('affineMap', (W2, b2), 'affineMap2')
        relu2 = Operation('relu', operationName='relu2')
        W3 = np.array([[-1., 1]])
        b3 = np.array([0.])
        affineMap3 = Operation('affineMap', (W3, b3), 'affineMap3')

        
        lb = np.array([-1, 1])
        ub = np.array([2,2])

        try:
            network = [affineMap1, relu1, affineMap2, relu2, affineMap3]
            inputSet = HyperBox(lb, ub)
            
            reachSet = inputSet.propagate(network)
            print('reachable sets of all operations:')
            for i in range(0, len(reachSet)-1):
                print('{}: lb = {}, ub = {}'.format(network[i].operationName, reachSet[i+1].lb, reachSet[i+1].ub))
        except Exception as e:
            print(f"Test Fail! {e}")
            self.n_fails = self.n_fails + 1
        else:
            print("Test Successfull!")

    def test_HyperBox_intersect(self):

        self.n_tests = self.n_tests + 1
        lb = np.array([1., 1.])
        ub = np.array([4., 3.])
        B = HyperBox(lb, ub)
        alpha = np.array([-0.9, -1.])
        gamma = -4.8

        try:
             B = HyperBox(lb, ub)
             alpha = np.array([-0.9, -1.])
             gamma = -4.8
             B1 = B.intersect(alpha, gamma)
             B1.__str__()
        except Exception:
            print("Test Fail!")
            self.n_fails = self.n_fails + 1
        else:
            print("Test Successfull!")

    ###
    ### SYMBOLIC BOUND TESTS
    ###
    def test_SB_constructor(self):
        self.n_tests += 1
        print('\nTesting SymbolicBound constructor (init_from_box)...')

        try:
            lb = np.array([-1.0, 1.0])
            ub = np.array([ 2.0, 2.0])
            S = SymbolicBound.init_from_box(lb, ub)
            S.__str__()
            assert S.in_dim  == 2
            assert S.out_dim == 2
        except Exception as e:
            print(f'Test Fail! ({e})')
            self.n_fails += 1
        else:
            print('Test Successfull!')

    def test_SB_affineMap(self):
        """Apply one affine map and check output dimension."""
        self.n_tests += 1
        print('\nTesting SymbolicBound affineMap...')
        try:
            lb = np.array([-0.0, 1.0])
            ub = np.array([ -1.0, 1.0])
            S  = SymbolicBound.init_from_box(lb, ub)
 
            W1 = np.array([[1., -1.], [-1., 2.]])
            b1 = np.array([0., 0.])
            S1 = S.affineMap(W1, b1)
            S1.__str__()
            assert S1.out_dim == 2
        except Exception as e:
            print(f'Test Fail! ({e})')
            self.n_fails += 1
        else:
            print('Test Successfull!')
 
    def test_SB_getConcreteBound(self):
        """Concrete bounds must be consistent with the HyperBox propagation."""
        self.n_tests += 1
        print('\nTesting SymbolicBound getConcreteBound...')
        try:
            lb = np.array([-1.0, 1.0])
            ub = np.array([ 2.0, 2.0])
            S  = SymbolicBound.init_from_box(lb, ub)
 
            W1 = np.array([[1., -1.], [-1., 2.]])
            b1 = np.array([0., 0.])
            S1 = S.affineMap(W1, b1)
 
            z_lb, z_ub = S1.getConcreteBound()
            print(f'  concrete lb = {z_lb}')
            print(f'  concrete ub = {z_ub}')
 
            # Cross-check against the HyperBox result from test_HyperBox_propagate
            # affineMap1 on lb=[-1,1], ub=[2,2]:
            #   expected lb = [-3, 0], ub = [1, 5]  (from lecture slides)
            expected_lb = np.array([-3., 0.])
            expected_ub = np.array([ 1., 5.])
            assert np.allclose(z_lb, expected_lb, atol=1e-9), \
                f'lb mismatch: got {z_lb}, expected {expected_lb}'
            assert np.allclose(z_ub, expected_ub, atol=1e-9), \
                f'ub mismatch: got {z_ub}, expected {expected_ub}'
        except Exception as e:
            print(f'Test Fail! ({e})')
            self.n_fails += 1
        else:
            print('Test Successfull!')
 
    def test_SB_ReLU(self):
        """ReLU concrete bounds must be >= 0 and consistent with HyperBox relu."""
        self.n_tests += 1
        print('\nTesting SymbolicBound ReLU...')
        try:
            lb = np.array([-1.0, 1.0])
            ub = np.array([ 2.0, 2.0])
            S  = SymbolicBound.init_from_box(lb, ub)
 
            W1 = np.array([[1., -1.], [-1., 2.]])
            b1 = np.array([0., 0.])
            S1 = S.affineMap(W1, b1)
            S2 = S1.ReLU()
 
            z_lb, z_ub = S2.getConcreteBound()
            print(f'  after ReLU concrete lb = {z_lb}')
            print(f'  after ReLU concrete ub = {z_ub}')
 
            # ReLU concrete ub must be >= 0, lb must be >= 0
            assert np.all(z_ub >= -1e-9), 'upper bound has negative entry after ReLU'
            assert np.all(z_lb >= -1e-9), 'lower bound has negative entry after ReLU'
 
            # Concrete ub must be <= HyperBox ub 
            expected_ub = np.array([1., 5.])   # relu1 ub from lecture
            assert np.all(z_ub <= expected_ub + 1e-9), \
                f'ub overapprox violated: {z_ub} > {expected_ub}'
        except Exception as e:
            print(f'Test Fail! ({e})')
            self.n_fails += 1
        else:
            print('Test Successfull!')
 
    def test_SB_propagate(self):
        
        self.n_tests += 1
        print('\nTesting SymbolicBound propagate (Lecture 4-5 network)...')
        try:
            W1 = np.array([[1., -1.], [-1., 2.]])
            b1 = np.array([0., 0.])
            affineMap1 = Operation('affineMap', (W1, b1), 'affineMap1')
            relu1 = Operation('relu', operationName='relu1')
            W2 = np.array([[1., 1.], [-3., 2.]])
            b2 = np.array([0., 0.])
            affineMap2 = Operation('affineMap', (W2, b2), 'affineMap2')
            relu2 = Operation('relu', operationName='relu2')
            W3 = np.array([[-1., 1.]])
            b3 = np.array([0.])
            affineMap3 = Operation('affineMap', (W3, b3), 'affineMap3')
 
            lb = np.array([-1., 1.])
            ub = np.array([ 2., 2.])
            S0 = SymbolicBound.init_from_box(lb, ub)
 
            network = [affineMap1, relu1, affineMap2, relu2, affineMap3]
            reach = S0.propagate(network)
 
            print('  Symbolic reachable sets (concrete bounds at each layer):')
            for i, op in enumerate(network):
                z_lb, z_ub = reach[i+1].getConcreteBound()
                print('  {}: lb = {}, ub = {}'.format(op.operationName, z_lb, z_ub))
 
            # Final output: HyperBox gives lb=[-6], ub=[10]
            # SymbolicBound must be a tighter or equal overapproximation
            z_lb_final, z_ub_final = reach[-1].getConcreteBound()
            assert z_lb_final[0] >= -6.0 - 1e-9, f'final lb too loose: {z_lb_final}'
            assert z_ub_final[0] <= 10.0 + 1e-9, f'final ub too loose: {z_ub_final}'
 
        except Exception as e:
            print(f'Test Fail! ({e})')
            self.n_fails += 1
        else:
            print('Test Successfull!')
    
    ###
    ### TEST ZONOTOPE BASED ON VALUE FROM L9
    ###

    def test_Zonotope_constructor(self):
        """Construct a Zonotope from_box and check dimensions"""
        self.n_tests += 1
        print('\nTesting Zonotope constructor (from_box)...')
        try:
            lb = np.array([-1.0,  1.0])
            ub = np.array([ 2.0,  2.0])
            Z  = Zonotope.from_box_to_zonotope(lb, ub)
            Z.__str__()
 
            expected_c = np.array([0.5, 1.5])
            expected_V = np.diag([1.5, 0.5])

            assert np.allclose(Z.c, expected_c), f'center mismatch: {Z.c}'
            assert np.allclose(Z.V, expected_V), f'V mismatch:\n{Z.V}'
            assert Z.dim == 2
            assert Z.n_gen == 2

        except Exception as e:
            print(f'Test Fail! ({e})')
            self.n_fails += 1
        else:
            print('Test Successfull!')
 
    def test_Zonotope_getConcreteBound(self):
        """Concrete bounds from a box-initialised Zonotope must recover lb/ub"""

        self.n_tests += 1
        print('\nTesting Zonotope getConcreteBound...')
        try:
            lb = np.array([-1.0, 1.0])
            ub = np.array([ 2.0, 2.0])
            Z  = Zonotope.from_box_to_zonotope(lb, ub)
            z_lb, z_ub = Z.getConcreteBound()

            print(f'  concrete lb = {z_lb}  (expected {lb})')
            print(f'  concrete ub = {z_ub}  (expected {ub})')
            assert np.allclose(z_lb, lb), f'lb mismatch: {z_lb}'
            assert np.allclose(z_ub, ub), f'ub mismatch: {z_ub}'

        except Exception as e:
            print(f'Test Fail! ({e})')
            self.n_fails += 1
        else:
            print('Test Successfull!')
 
    def test_Zonotope_affineMap(self):
        """affineMap result concrete bounds must agree with HyperBox affineMap."""
        self.n_tests += 1
        print('\nTesting Zonotope affineMap...')
        try:
            lb = np.array([-1.0, 1.0])
            ub = np.array([ 2.0, 2.0])
            Z  = Zonotope.from_box_to_zonotope(lb, ub)
 
            W1 = np.array([[1., -1.], [-1., 2.]])
            b1 = np.array([0., 0.])
            Z1 = Z.affineMap(W1, b1)
            Z1.__str__()
 
            z_lb, z_ub = Z1.getConcreteBound()
            print(f'  after affineMap1: lb = {z_lb}, ub = {z_ub}')
 
            # Must match HyperBox result: lb=[-3,0], ub=[1,5]
            expected_lb = np.array([-3., 0.])
            expected_ub = np.array([ 1., 5.])
            assert np.allclose(z_lb, expected_lb, atol=1e-9), f'lb mismatch: {z_lb}'
            assert np.allclose(z_ub, expected_ub, atol=1e-9), f'ub mismatch: {z_ub}'

        except Exception as e:
            print(f'Test Fail! ({e})')
            self.n_fails += 1
        else:
            print('Test Successfull!')
 
    def test_Zonotope_ReLU(self):
        """After ReLU, concrete bounds must be >= 0 and no looser than HyperBox"""
        self.n_tests += 1
        print('\nTesting Zonotope ReLU...')
        try:
            lb = np.array([-1.0, 1.0])
            ub = np.array([ 2.0, 2.0])
            Z  = Zonotope.from_box_to_zonotope(lb, ub)
 
            W1 = np.array([[1., -1.], [-1., 2.]])
            b1 = np.array([0., 0.])
            Z1 = Z.affineMap(W1, b1)    # concrete: lb=[-3,0], ub=[1,5]
            Z2 = Z1.ReLU()
            Z2.__str__()
 
            z_lb, z_ub = Z2.getConcreteBound()
            print(f'  after ReLU: lb = {z_lb}, ub = {z_ub}')
 
            # The overapproximation upper bound must be >= 0
            assert np.all(z_ub >= -1e-9), 'upper bound negative after ReLU'
 
        
            hbox_ub = np.array([1., 5.])
            assert np.all(z_ub <= hbox_ub + 1e-9), \
                f'Zonotope ub looser than HyperBox: {z_ub} > {hbox_ub}'
 
            # A new generator column must have been added for the crossing neuron
            assert Z2.n_gen > Z1.n_gen, \
                'Expected at least one extra generator for the crossing neuron'
            
        except Exception as e:
            print(f'Test Fail! ({e})')
            self.n_fails += 1
        else:
            print('Test Successfull!')
 
    def test_Zonotope_propagate(self):
        """Full propagation through the Lecture 4-5 network."""
        self.n_tests += 1
        print('\nTesting Zonotope propagate (Lecture 4-5 network)...')
        try:
            W1 = np.array([[1., -1.], [-1., 2.]])
            b1 = np.array([0., 0.])
            affineMap1 = Operation('affineMap', (W1, b1), 'affineMap1')
            relu1      = Operation('relu', operationName='relu1')
            W2 = np.array([[1., 1.], [-3., 2.]])
            b2 = np.array([0., 0.])
            affineMap2 = Operation('affineMap', (W2, b2), 'affineMap2')
            relu2      = Operation('relu', operationName='relu2')
            W3 = np.array([[-1., 1.]])
            b3 = np.array([0.])
            affineMap3 = Operation('affineMap', (W3, b3), 'affineMap3')
 
            lb = np.array([-1., 1.])
            ub = np.array([ 2., 2.])
            Z0 = Zonotope.from_box_to_zonotope(lb, ub)
 
            network = [affineMap1, relu1, affineMap2, relu2, affineMap3]
            reach   = Z0.propagate(network)
 
            print('  Zonotope reachable sets (concrete bounds at each layer):')
            for i, op in enumerate(network):
                z_lb, z_ub = reach[i+1].getConcreteBound()
                print('  {}: lb = {}, ub = {}, generators = {}'.format(
                    op.operationName, z_lb, z_ub, reach[i+1].n_gen))
 
            # Final output must be within HyperBox bounds lb=[-6], ub=[10]
            z_lb_f, z_ub_f = reach[-1].getConcreteBound()
            assert z_lb_f[0] >= -6.0 - 1e-9, f'final lb too loose: {z_lb_f}'
            assert z_ub_f[0] <= 10.0 + 1e-9, f'final ub too loose: {z_ub_f}'
 
            # Zonotope should give tighter (or equal) bounds than HyperBox fro L9
            print(f'\n  Summary — final output bounds:')
            print(f'    HyperBox:     lb=[-6.], ub=[10.]')
            print(f'    SymbolicBound:lb=[-5.], ub=[5.]')
            print(f'    Zonotope:     lb={z_lb_f}, ub={z_ub_f}')
 
        except Exception as e:
            print(f'Test Fail! ({e})')
            self.n_fails += 1
        else:
            print('Test Successfull!')

    ###
    ### STAR TESTS
    ###

    def test_Star_constructor(self):
        self.n_tests += 1
        print('\nTesting Star constructor (from_box)...')
        try:
            lb = np.array([0.0, -1.0])
            ub = np.array([1.0,  1.0])
            S  = Star.from_box(lb, ub)
            S.__str__()
            assert S.dim == 2
            assert S.m   == 2
            # center should be [0.5, 0.0], V = diag([0.5, 1.0])
            assert np.allclose(S.c, [0.5, 0.0]),     f'center wrong: {S.c}'
            assert np.allclose(S.V, np.diag([0.5, 1.0])), f'V wrong: {S.V}'
        except Exception as e:
            print(f'Test Fail! ({e})')
            self.n_fails += 1
        else:
            print('Test Successfull!')
 
    def test_Star_affineMap(self):
        self.n_tests += 1
        print('\nTesting Star affineMap...')
        try:
            lb = np.array([0.0, -1.0])
            ub = np.array([1.0,  1.0])
            S  = Star.from_box(lb, ub)
            
            W  = np.array([[1.0, -1.0], [-1.0, 1.0], [1.0, -2.0]])
            b  = np.array([0., 0., 0.])
            S1 = S.affineMap(W, b)
            assert S1.dim == 3, f'expected dim=3, got {S1.dim}'
            assert S1.m   == 2, f'expected m=2,   got {S1.m}'
            # Predicate unchanged
            
            assert S1.C.shape == S.C.shape
            lb_e, ub_e = S1.getExactBound()
            print(f'  after affineMap1: lb={lb_e}, ub={ub_e}')
        except Exception as e:
            print(f'Test Fail! ({e})')
            self.n_fails += 1
        else:
            print('Test Successfull!')
 
    def test_Star_stepReLU(self):
        self.n_tests += 1
        print('\nTesting Star stepReLU...')
        try:
            lb = np.array([0.0, -1.0])
            ub = np.array([1.0,  1.0])
            S  = Star.from_box(lb, ub)
            W  = np.array([[1.0, -1.0], [-1.0, 1.0], [1.0, -2.0]])
            b  = np.array([0., 0., 0.])
            S1 = S.affineMap(W, b)
 
            # dim 0: lb=-1, ub=2 → crossing → should split into 2
            result = S1.stepReLU(0)
            print(f'  stepReLU(dim=0): {len(result)} star(s) returned')
            assert len(result) in (1, 2), f'unexpected count: {len(result)}'
        except Exception as e:
            print(f'Test Fail! ({e})')
            self.n_fails += 1
        else:
            print('Test Successfull!')
 
    def test_Star_ReLU(self):
        self.n_tests += 1
        print('\nTesting Star ReLU...')
        try:
            lb = np.array([0.0, -1.0])
            ub = np.array([1.0,  1.0])
            S  = Star.from_box(lb, ub)
            W  = np.array([[1.0, -1.0], [-1.0, 1.0], [1.0, -2.0]])
            b  = np.array([0., 0., 0.])
            S1 = S.affineMap(W, b)
 
            stars_out = S1.ReLU()
            print(f'  ReLU produced {len(stars_out)} star(s)')
 
            # Compute union bounds
            lb_out, ub_out = Star.getOutputBound(stars_out)
            print(f'  union lb = {lb_out}')
            print(f'  union ub = {ub_out}')
            assert np.all(lb_out >= -1e-9), 'lb negative after ReLU'
            assert np.all(ub_out >= -1e-9), 'ub negative after ReLU'
        except Exception as e:
            print(f'Test Fail! ({e})')
            self.n_fails += 1
        else:
            print('Test Successfull!')


    def test_example_network(self):
        self.n_tests += 1
        print('\nTesting Problem 1 Part 4...')
        try:
            # input set
            lb = np.array([0.0, -1.0]) # [x_lb, y_lb]
            ub = np.array([1.0, 1.0])  # [x_ub, y_ub]

            W1 = np.array([[1.0, -1.0], [-1.0, 1.0], [1.0, -2.0]])
            b1 = np.array([0., 0., 0.])
            affineMap1 = Operation('affineMap', (W1, b1), 'affineMap1')
            relu1      = Operation('relu', operationName='relu1')

            W2 = np.array([[2., 2., -3.], [0.5, -1., -2.]])
            b2 = np.array([0., 0.])
            affineMap2 = Operation('affineMap', (W2, b2), 'affineMap2')
            relu2      = Operation('relu', operationName='relu2')

            W3 = np.array([[-1., 2.]])
            b3 = np.array([0.])
            affineMap3 = Operation('affineMap', (W3, b3), 'affineMap3')

            # List of operations
            network = [affineMap1, relu1, affineMap2, relu2, affineMap3]
            
            B0 = HyperBox(lb, ub)
            S0 = SymbolicBound.init_from_box(lb, ub)
            
            Z0 = Zonotope.from_box_to_zonotope(lb, ub)
            
            B0_reach   = B0.propagate(network)
            print('\n\n  HyperBox reachable sets:')
            for i in range(len(B0_reach)-1):
                print('  {}: lb = {}, ub = {}'.format(
                    network[i].operationName, B0_reach[i+1].lb, B0_reach[i+1].ub))
                
            S0_reach   = S0.propagate(network)
            print('  SymbolicBounds reachable sets:')
            for i, op in enumerate(network):
                z_lb, z_ub = S0_reach[i+1].getConcreteBound()
                print('  {}: lb = {}, ub = {}'.format(op.operationName, z_lb, z_ub))
                
            Z0_reach   = Z0.propagate(network)
            print('  Zonotope reachable sets :')
            for i, op in enumerate(network):
                z_lb, z_ub = Z0_reach[i+1].getConcreteBound()
                print('  {}: lb = {}, ub = {}, generators = {}'.format(
                    op.operationName, z_lb, z_ub, Z0_reach[i+1].n_gen))
            
            ST0       = Star.from_box(lb, ub)
            ST_reach  = Star.propagate(ST0, network)
            print('\n  Exact Star reachable sets:')
            for i, op in enumerate(network):
                stars = ST_reach[i+1]
                lb_s, ub_s = Star.getOutputBound(stars)
                print('  {}: lb = {}, ub = {}, |stars| = {}'.format(
                    op.operationName, lb_s, ub_s, len(stars)))
            

            
            # Get CB of the output neuron
            B0_lb, B0_ub = B0_reach[-1].lb, B0_reach[-1].ub
            S0_lb, S0_ub = S0_reach[-1].getConcreteBound()
            Z0_lb, Z0_ub = Z0_reach[-1].getConcreteBound()
            ST_lb, ST_ub   = Star.getOutputBound(ST_reach[-1])

            print(f"\n\n- HYPER RECTANGLE: [{B0_lb}, {B0_ub}]")
            print(f"- SYMBOLIC BOUND: [{S0_lb}, {S0_ub}]")
            print(f"- ZONOTOPE: [{Z0_lb}, {Z0_ub}]")
            print(f'  EXACT STAR: [{ST_lb}, {ST_ub}]')


        except Exception as e:
            print(f'Test Fail! ({e})')
            self.n_fails += 1
        else:
            print('Test Successfull!')


if __name__ == "__main__":

    test = Test()
    print('\n=======================\
    ================================\
    ================================\
    ===============================\n')
    """
    print("===HYPER_BOX TEST!!!!!===")
    test.test_HyperBox_constructor()
    test.test_HyperBox_str()
    test.test_HyperBox_affineMap()
    test.test_rand_ffnn()
    test.test_HyperBox_propagate()
    test.test_HyperBox_intersect()
    
    print('\n===SYMBOLIC BOUND TEST!!!!!===')
    test.test_SB_constructor()
    test.test_SB_affineMap()
    test.test_SB_getConcreteBound()
    test.test_SB_ReLU()
    test.test_SB_propagate()
    
    print('\n=== ZONOTOPE TEST!!!! ====')
    test.test_Zonotope_constructor()
    test.test_Zonotope_getConcreteBound()
    test.test_Zonotope_affineMap()
    test.test_Zonotope_ReLU()
    test.test_Zonotope_propagate()
    

    print('\n=== STAR UNIT TESTS ===')
    test.test_Star_constructor()
    test.test_Star_affineMap()
    test.test_Star_stepReLU()
    test.test_Star_ReLU()
    """
    test.test_example_network()
    """ OUTPUTs:

    HyperBox reachable sets:
    affineMap1: lb = [-1. -2. -2.], ub = [2. 1. 3.]
    relu1: lb = [0. 0. 0.], ub = [2. 1. 3.]
    affineMap2: lb = [-9. -7.], ub = [6. 1.]
    relu2: lb = [0. 0.], ub = [6. 1.]
    affineMap3: lb = [-6.], ub = [2.]
    SymbolicBounds reachable sets:
    affineMap1: lb = [-1. -2. -2.], ub = [2. 1. 3.]
    relu1: lb = [0. 0. 0.], ub = [2. 1. 3.]
    affineMap2: lb = [-9. -6.], ub = [4. 1.]
    relu2: lb = [0. 0.], ub = [4.        1.0000001]
    affineMap3: lb = [-4.], ub = [2.0000002]
    Zonotope reachable sets :
    affineMap1: lb = [-1. -2. -2.], ub = [2. 1. 3.], generators = 2
    relu1: lb = [-0.66666667 -0.66666667 -1.2       ], ub = [2. 1. 3.], generators = 5
    affineMap2: lb = [-7.66666667 -5.33333333], ub = [5.6        2.06666667], generators = 5
    relu2: lb = [-3.2361809  -1.48948949], ub = [5.6        2.06666667], generators = 7
    affineMap3: lb = [-4.81801802], ub = [3.60855328], generators = 7

        - affineMap1 for all methods is the same, which is consistent
        - ReLU is activated at ReLU 1, hence there should be more generators

    Final output:
    - HYPER RECTANGLE: [[-6.], [2.]]    
    - SYMBOLIC BOUND: [[-4.], [2.0000002]]
    - ZONOTOPE: [[-4.81801802], [3.60855328]]
    
        - Symbolic lower bound shows a tighter bound due to the fact that it tracks its dependencies from the original input variables
        - Zonotope looser upper bound shows its overapproximation of ReLU when introducing a new independent generator
    
    """
    print('\n========================\
    =================================\
    =================================\
    =================================\n')
    print('Testing HyperBox Class: fails: {}/{}, successfull: {}/{}, \
    total tests: {}'.format(test.n_fails, test.n_tests, \
                            test.n_tests - test.n_fails, \
                            test.n_tests, test.n_tests))
   
    

        



    
    


            
            
            
