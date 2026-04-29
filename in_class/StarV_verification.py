"""
Verification using StarV
Dung Tran
3/10/2026
"""

from StarV.net.network import NeuralNetwork
from StarV.verifier.verifier import reachExactBFS, checkSafetyStar
import numpy as np
import multiprocessing
from StarV.util.load import load_2017_IEEE_TNNLS, load_ACASXU
import time
from StarV.util.plot import plot_star, plot_3D_Star
from matplotlib import pyplot as plt
from StarV.set.star import Star
import os

class Test(object):
    """
    Testing module net class and methods
    """

    def __init__(self):

        self.n_fails = 0
        self.n_tests = 0

    def test_reach_2017_IEEE_TNNLS(self):
        """reachability analysis for 2017 IEEE TNNLS network"""

        self.n_tests = self.n_tests + 1

        print('Test exact reachability of 2017 IEEE TNNLS network...')
        
        try:
            lb = np.array([-1.0, -1.0, -1.0])
            ub = np.array([1.0, 1.0, 1.0])
            In = Star(lb, ub)
            inputSet = []
            inputSet.append(In)
            net = load_2017_IEEE_TNNLS()
            net.info()
            #pool = multiprocessing.Pool(8)
            #start = time.time()
            #S = reachExactBFS(net=net, inputSet=inputSet, pool=pool)
            #pool.close()
            #print('Number of output sets: {}'.format(len(S)))
            #end = time.time()
            #print('Reachability time = {}'.format(end - start))
            #plot_star(S)
        except Exception:
            print('Test Fail!')
            self.n_fails = self.n_fails + 1
        else:
            print('Test Successfull!')

    def test_reach_ACASXU(self, x, y, spec_id):
        """Reachability analysis of ACASXU network"""

        self.n_tests = self.n_tests + 1

        print('Test probabilistic reachability of ACASXU N_{}_{} network under specification {}...'.format(x, y, spec_id))
        
        try:
            net, lb, ub, unsafe_mat, unsafe_vec = load_ACASXU(x, y, spec_id)
            S = Star(lb, ub)
            inputSet = []
            inputSet.append(S)
            net.info()
            pool = multiprocessing.Pool(8)
            start = time.time()
            S = reachExactBFS(net=net, inputSet=inputSet, pool=pool)
            pool.close()
            print('Number of output sets: {}'.format(len(S)))
            end = time.time()
            print('Reachability time = {}'.format(end - start))

            # Plot reachable set in 3D using plot_3D_Star function
            
            # # x[0] = COC (Clear-of-Conflict)
            # # x[1] = Weak-Left
            # # x[2] = Weak-Right
            # # x[3] = Strong-Left
            # # x[4] = Strong-Right
            # # Look at load_ACASXU


            # project reach set to x[0] (COC), x[1] (WL), x[2] (WR)
            
            proj_mat = np.array([[1., 0., 0., 0., 0], [0., 1., 0., 0., 0.], [0., 0., 1, 0, 0]])
            S_3D = []
            for Si in S:
                S_3D.append(Si.affineMap(proj_mat))

            plot_3D_Star(S_3D)
        
        except Exception:
            print('Test Fail!')
            self.n_fails = self.n_fails + 1
        else:
            print('Test Successfull!')

    def  test_checkSafetyStar(self):

        self.n_tests = self.n_tests + 1
        print('Test intersectWithUnsafeRegion method...')
        
        
        try:
            lb = -np.random.rand(3,)
            ub = np.random.rand(3,)
            V = np.random.rand(3, 4)
            C = np.random.rand(3, 3)
            d = np.random.rand(3,)
            S = Star(V, C, d, lb, ub)
            unsafe_mat = np.random.rand(2, 3,)
            unsafe_vec = np.random.rand(2,)
            P = checkSafetyStar(unsafe_mat, unsafe_vec, S)
            S.__str__()
            if isinstance(P, Star):
                print('\nUnsafe Set')
                P.__str__()
            else:
                print('\nSafe')
        except Exception:
            print('Test Fail!')
            self.n_fails = self.n_fails + 1
        else:
            print('Test Successfull!')
   

if __name__ == "__main__":

    test_verifier = Test()
    print('\n=======================\
    ================================\
    ================================\
    ===============================\n')
    
    
    test_verifier.test_reach_2017_IEEE_TNNLS()
    #test_verifier.test_reach_ACASXU(x=3,y=7,spec_id=3)
    # test_verifier.test_checkSafetyStar()

    print('\n========================\
    =================================\
    =================================\
    =================================\n')
    print('Testing: fails: {}, successfull: {}, \
    total tests: {}'.format(test_verifier.n_fails,
                            test_verifier.n_tests - test_verifier.n_fails,
                            test_verifier.n_tests))

