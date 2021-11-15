# -*- coding: utf-8 -*-
"""
@author: pf
"""
import numpy as np
from .Kernels import Linear
from .Regularizers import Regularizer

class Optimizer(object):
    def __init__(self, X, Y, kernel=Linear, bw=-1, deg=0, reg=0, regularizer=Regularizer):
        self._Xtr = X
        self._Ytr = Y
        self._reg = reg
        self._regularizer = regularizer(reg)
        self._kernel = kernel(X, Y, bw, deg)
        self.n = X.shape[0]
        self._alpha = np.random.randn(self.n)
    
    @property
    def alpha(self):
        return self._alpha
    @alpha.setter
    def alpha(self, value):
        self._alpha = value
    @alpha.deleter
    def alpha(self):
        del self._alpha
    
    def predict(self, X=None, alpha=None):
        pass    
    def train_loss(self, alpha=None, indexes=None):
        pass
    def test_loss(self):
        pass
    def gradient(self, alpha=None, indexes=None):
        pass

class k_Perceptron(Optimizer):
    def __init__(self, X, Y, kernel=Linear, bw=-1, deg=0, reg=0, regularizer=Regularizer):
        super().__init__(X, Y, kernel, bw, deg, reg, regularizer)
    
    def predict(self, Xval, alpha=None):
        if alpha is None:
            alpha = self._alpha
            
        Kpred = self._kernel.get_kernel(Xval)
        return np.sign(np.dot(Kpred, alpha))
    
    def train_loss(self, alpha, indexes=None):
        if indexes is None:
            indexes = np.arange(self.n)
        Kalpha = np.dot(self._kernel.Ktr, alpha) 
        error = -Kalpha[indexes] * self._Ytr[indexes]
        error[error<0] = 0
        reg_loss = self._regularizer.lossk(alpha, Kalpha)
        return np.mean(error) + reg_loss
    
    def test_loss(self, Xtest, Ytest, alpha=None):
        if alpha is None:
            alpha =  np.arange(self.n)
        num = Xtest.shape[0]
        Kpred = self._kernel.get_kernel(Xtest)
        Z = np.dot(Kpred, alpha)
        error = -Z * Ytest
        error[error<0] = 0
        reg_loss = self._regularizer.lossk(alpha, Z)
        return np.mean(error) + reg_loss
    
    def n_misclassification(self, Xtest, Ytest, alpha=None): #number of misclassification
        if alpha is None:
            alpha =  self._alpha
        Z = self.predict(Xtest, alpha)
        error = -Z * Ytest
        error[error<0] = 0
        return np.sum(error)
     
    def gradient(self, alpha, indexes=None):
        if indexes is None:
            indexes =  np.arange(self.n)
        Kalpha = np.dot(self._kernel.Ktr, alpha)
        error = -Kalpha[indexes] * self._Ytr[indexes]
        gradient = - self._kernel.Ktr[indexes,:] * self._Ytr[indexes, np.newaxis]
        gradient[error<0] = 0
        reg_grad = self._regularizer.gradient(Kalpha)
        return np.mean(gradient, axis=0) + reg_grad
    
    def gradient1(self, alpha, indexes=None):
        if indexes is None:
            indexes =  np.arange(self.n)
        Kalpha = np.dot(self._kernel.Ktr, alpha)
        gradient = np.zeros(self._n)
        Z = Kalpha[indexes]
        error = -Z * self._Ytr[indexes]
        wrong_ind = indexes[error>=0]
        gradient[wrong_ind] = -self._Ytr[wrong_ind]
        reg_grad = self._regularizer.gradient(Kalpha)
        return gradient + reg_grad

class k_SVM(Optimizer):
    def __init__(self, X, Y, kernel=Linear, bw=-1, deg=0, reg=0, regularizer=Regularizer):
        super().__init__(X, Y, kernel, bw, deg, reg, regularizer)
    
    def predict(self, Xval, alpha=None):
        if alpha is None:
            alpha = self._alpha
            
        Kpred = self._kernel.get_kernel(Xval)
        return np.sign(np.dot(Kpred, alpha))
    
    def train_loss(self, alpha, indexes=None):
        if indexes is None:
            indexes = np.arange(self.n)
        Kalpha = np.dot(self._kernel.Ktr, alpha)    
        Z = Kalpha[indexes]
        error = 1 - Z * self._Ytr[indexes]
        error[error<0] = 0
        reg_loss = self._regularizer.lossk(alpha, Kalpha)
        return np.mean(error) + reg_loss
    
    def test_loss(self, Xtest, Ytest, alpha=None):
        if alpha is None:
            alpha =  np.arange(self.n)
        num = Xtest.shape[0]
        Kpred = self._kernel.get_kernel(Xtest)
        Z = np.dot(Kpred, alpha)
        error = 1 - Z * Ytest
        error[error<0] = 0
        reg_loss = self._regularizer.lossk(alpha, Z)
        return np.mean(error) + reg_loss
    
    def n_misclassification(self, Xtest, Ytest, alpha=None): #number of misclassification
        if alpha is None:
            alpha =  np.arange(self.n)
        Z = self.predict(Xtest, alpha)
        error = -Z * Ytest
        error[error<0] = 0
        return np.sum(error)
     
    def gradient(self, alpha, indexes=None):
        if indexes is None:
            indexes =  np.arange(self.n)
           
        Kalpha = np.dot(self._kernel.Ktr, alpha)
        Z = Kalpha[indexes]
        error = 1 - Z * self._Ytr[indexes]
        gradient = - self._kernel.Ktr[indexes,:] * self._Ytr[indexes, np.newaxis]
        gradient[error<0] = 0
        reg_grad = self._regularizer.gradient(Kalpha)
        return np.mean(gradient, axis=0) + reg_grad  
    
    def gradient1(self, alpha, indexes=None):
        if indexes is None:
            indexes =  np.arange(self.n)
        gradient = np.zeros(self._n)
        Kalpha = np.dot(self._kernel.Ktr, alpha)
        Z = Kalpha[indexes]
        error = 1 - Z * self._Ytr[indexes]
        wrong_ind = indexes[error>0]
        gradient[wrong_ind] = -self._Ytr[wrong_ind]
        reg_grad = self._regularizer.gradient(Kalpha)
        return gradient + reg_grad 

class k_Logistic(Optimizer):
    def __init__(self, X, Y, kernel=Linear, bw=-1, deg=0, reg=0, regularizer=Regularizer):
        super().__init__(X, Y, kernel, bw, deg, reg, regularizer)
    
    def predict(self, Xval, alpha=None):
        if alpha is None:
            alpha = self._alpha
            
        Kpred = self._kernel.get_kernel(Xval)
        z = np.dot(Kpred, alpha)
        return 1 / (1+np.exp(-z))
    
    def train_loss(self, alpha, indexes=None):
        if indexes is None:
            indexes = np.arange(self.n)
            
        Kalpha = np.dot(self._kernel.Ktr, alpha)    
        Z = Kalpha[indexes]
        error = np.log( 1+np.exp(-Z * self._Ytr[indexes]) ) #logistic loss        
        reg_loss = self._regularizer.lossk(alpha, Kalpha)
        return np.mean(error) + reg_loss
    
    def test_loss(self, Xtest, Ytest, alpha=None):
        if alpha is None:
            alpha =  np.arange(self.n)
        num = Xtest.shape[0]
        Kpred = self._kernel.get_kernel(Xtest)
        Z = np.dot(Kpred, alpha)
        error = np.log( 1+np.exp(-Z * Ytest) )
        reg_loss = self._regularizer.lossk(alpha, Z)
        return np.mean(error) + reg_loss
     
    def gradient(self, alpha, indexes=None): 
        if indexes is None:
            indexes =  np.arange(self.n)
        Kalpha = np.dot(self._kernel.Ktr, alpha)
        Z = Kalpha[indexes]
        z = - Z * self._Ytr[indexes]
        z_gd = - self._kernel.Ktr[indexes,:] * self._Ytr[indexes, np.newaxis]
        dim_expand = (np.exp(z) / (1+np.exp(z)))[:, np.newaxis]
        total_gd = z_gd * dim_expand
        reg_grad = self._regularizer.gradient(Kalpha)
        return np.mean(total_gd, axis=0) + reg_grad
    
class k_Regression(Optimizer):
    def __init__(self, X, Y, kernel=Linear, bw=-1, deg=0, reg=0, regularizer=Regularizer):
        super().__init__(X, Y, kernel, bw, deg, reg, regularizer)
        
    def predict(self, Xval, alpha=None):
        if alpha is None:
            alpha = self._alpha
            
        Kpred = self._kernel.get_kernel(Xval)
        return np.dot(Kpred, alpha)
    
    def train_loss(self, alpha, indexes=None):
        if indexes is None:
            indexes = np.arange(self.n)
        
        Kalpha = np.dot(self._kernel.Ktr, alpha)
        Z = Kalpha[indexes]
        error = np.square(np.linalg.norm(self._Ytr[indexes]-Z))#here we directly calculate the total residual error of all observations.
        reg_loss = self._regularizer.lossk(alpha, Kalpha)
        return error / indexes.size + reg_loss
    
    def test_loss(self, Xtest, Ytest, alpha=None):
        if alpha is None:
            alpha =  np.arange(self.n)
        num = Xtest.shape[0]
        Kpred = self._kernel.get_kernel(Xtest)
        Z = np.dot(Kpred, alpha)
        error = np.square(np.linalg.norm(Ytest-Z))
        reg_loss = self._regularizer.lossk(alpha, Z)
        return error / num + reg_loss
    
    def gradient(self, alpha, indexes=None):
        if indexes is None:
            indexes =  np.arange(self.n)
        
        Kalpha = np.dot(self._kernel.Ktr, alpha)
        Z = Kalpha[indexes]
        dif = self._Ytr[indexes] - Z
        gradient = - np.dot(self._kernel.Ktr[indexes,:].T, dif)# calculate the gradient in matrix form.
        reg_grad = self._regularizer.gradient(Kalpha)
        return gradient / indexes.size + reg_grad
     
    def closed(self):
        return np.dot(np.linalg.pinv(self._kernel.Ktr + self._reg*np.eye(self.n) ), self._Ytr)
            
            
            
            
            
            
        
        