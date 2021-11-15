import numpy as np

class Regularizer(object):
    def __init__(self, reg=0):
        super().__init__()
        self._lambd = reg
    @property    
    def lambd(self):
        return self._lambd
    @lambd.setter
    def lambd(self, value):
        self._lambd = value
    
    def loss(self, w):
        return 0
    def lossk(self, alpha, Z):
        return 0
    def gradient(self, w):
        return 0

class L2(Regularizer):
    def __init__(self, reg=0):
        super().__init__(reg)
    
    def loss(self, w):
        return self._lambd * np.power(np.linalg.norm(w, 2), 2)
    def lossk(self, alpha, Z):
        return self._lambd * np.dot(alpha.T, Z)
    
    def gradient(self, w):
        return 2 * self._lambd * w

class L1(Regularizer):
    def __init__(self, reg=0):
        super().__init__(reg)
    
    def loss(self, w):
        return self._lambd * np.linalg.norm(w, 1)
    def lossk(self, w):
        return self._lambd * np.linalg.norm(w, 1)
    def gradient(self, w):
        return self._lambd * np.sign(w)
    