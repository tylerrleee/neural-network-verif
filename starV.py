import numpy as np

class Star:
    def __init__(self, 
                 c: np.array , 
                 V: np.array , 
                 P , 
                 u: np.float32 = None, 
                 l: np.float32 = None):
        
        # < c, V P >
        self.c = c
        self.V: np.array = V
        self.P = P

        self.upper = u
        self.lower = l

    def affineMap(self, W: np.array, b: np.array):
        """
        W (np.array): some weight vector
        b (np.array): offset vector

        return:
            Star object
        """
        
        cp = np.matmul(W, self.c) + b
        Vp = np.matmul(W, self.V)
        Pp = self.P

        return Star(cp, Vp, Pp)

    def intersection(self, G):
        """
        Gs: Some halfspace
        $P_H (a) = (G x V) a <= g - G x c$

        """ 
        P_half = np.matmul(G, V) 
        
    def getEstimateBound(self):

    def stepReLU(self, index):
        'perform stepReLU at specific index'
        xmin = self.getBound(ids = {index}, option="lower-bound")

if __name__ == '__main__':
    star= Star()
    print(star)