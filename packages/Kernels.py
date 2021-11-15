import numpy as np

class Kernels(object):
    def __init__(self, X, Y, bw=0, deg=0):
        self._Xtr = X
        self._Ytr = Y
        self._bw = bw
        self._deg = deg
        self._n = X.shape[0]
        self.Ktr = np.zeros((X.shape[0], X.shape[0]))
    
    @property
    def n(self):
        return self._n
    def get_kernel(self, Xval):
        pass

class Polynomial(Kernels):
    def __init__(self, X, Y, bw=0, deg=0):
        super().__init__(X,Y,bw,deg)
        self.Ktr = self.get_kernel(self._Xtr)
    
    def get_kernel(self, Xval):
        return np.power( 1 + np.dot(Xval, self._Xtr.T), self._deg)
    
class Linear(Kernels):
    def __init__(self, X, Y, bw=0, deg=0):
        super().__init__(X,Y,bw,deg)
        self.Ktr = self.get_kernel(self._Xtr)
        
    def get_kernel(self, Xval):
        return np.dot(Xval, self._Xtr.T)
    
class RBF(Kernels):
    def __init__(self, X, Y, bw=0, deg=2): #use different default degree value for RBF kernel, it means degree of norm, usually set to 2.
        super().__init__(X,Y,bw,deg)
        self.Ktr = self.get_kernel(self._Xtr)
    
    def get_kernel(self, Xval):
        K = []
        for x in Xval:
            row = np.exp( -np.power(np.linalg.norm(x - self._Xtr, 2, axis=-1), self._deg) / self._bw )
            row = np.atleast_2d(row)
            K.append(row)
        Kpred = np.concatenate(K)
        return Kpred
    
    
    
    
    
    
    
    
    
    
    
    
    