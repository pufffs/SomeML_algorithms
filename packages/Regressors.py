# -*- coding: utf-8 -*-
"""
Created on Tue May  4 18:03:57 2021

@author: roone
"""
from .Regularizers import Regularizer

import numpy as np
class LinearReg(object):
    def __init__(self, x, y, reg=0, regularizer=Regularizer):
        self.X= x
        self.Y= y
        self.w= np.random.randn(x.shape[1])
        self.n= x.shape[0]
        self._reg = 0
        self._regularizer = regularizer
    
    def loss(self, w, index=None): #loss function is also a function of w.
        if index is None:
            index = np.arange(self.n)
        error = np.dot(self.X[index,:], w) - self.Y[index]
        loss = np.dot(error.T, error) / index.size
        return loss
    
    def gradient(self, w, index=None):
        if index is None:
            index = np.arange(self.n)
        error = np.dot(self.X[index,:], w) - self.Y[index]
        return np.dot(self.X[index, :].T, error) / index.size + self._regularizer.gradient(w)
    
    def w_closedform(self):
        return np.dot( np.linalg.pinv(np.dot(self.X.T, self.X) ), np.dot(self.X.T, self.Y) )
    
    def predict(self, w):
        return np.dot(self.X, w)
    
    def test_loss(self, x_test, y_test, w_new=None):
        if w_new is None:
            w_new = self.w_closedform #if no trained parameters given, use the closed form solution. 
        error = np.dot(x_test, w_new) - y_test
        return np.power(error, 2).mean()

class GLM(object): #for data at each dimension of x, so here computation is univariate.
    def __init__(self, x, y, reg, features):
        self.X = x
        self.Y = y
        self.n = x.shape[0]
        self.reg = reg
        self.features = features
        
    def selection(self):
        #the following are lists containing lambda functions.
        selection = list()
        poly = list()
        sin = list()
        exp = list()
        for i in range(4):
            poly.append(lambda x, j=i: np.power(x, j))
        for i in [1, 2, 5, 10]:
            sin.append(lambda x, j=i: np.sin(j*x))
            exp.append(lambda x, j=i: np.exp(j*x))
        if 'constant' in self.features:
            selection.append(poly[0])
        if 'linear' in self.features:
            selection.append(poly[1])
        if 'quadratic' in self.features:
            selection.append(poly[2])
        if 'cubic' in self.features:
            selection.append(poly[3])
        if 'sin(x)' in self.features:
            selection.append(sin[0])
        if 'sin(2x)' in self.features:
            selection.append(sin[1])
        if 'sin(5x)' in self.features:
            selection.append(sin[2])
        if 'sin(10x)' in self.features:
            selection.append(sin[3])
        if 'exp(x)' in self.features:
            selection.append(exp[0])
        if 'exp(2x)' in self.features:
            selection.append(exp[1])
        if 'exp(5x)' in self.features:
            selection.append(exp[2])
        if 'exp(10x)' in self.features:
            selection.append(exp[3])
        return selection
    
    def Phi(self, x):
        feature_space = self.selection()
        return np.atleast_2d( np.stack( [f(x) for f in feature_space] ) ).T
    
    def w_hat(self):
        X = self.Phi(self.X)
        Y = self.Y
        XTX = np.dot(X.T,X)
        return np.dot( np.linalg.pinv( XTX + self.reg * np.identity(X.shape[1]) ), np.dot(X.T, Y) )
    
    def test_loss(self, x_test, y_test):
        y_pred = np.dot(self.Phi(x_test), self.w_hat() )
        return np.power(y_pred-y_test,2).mean()

class RSP(object):
    def __init__(self, X_train, Y_train, order, df):
        self.order = order
        self.X = X_train
        self.Y = Y_train
        self.df = df
        self.knots = np.linspace(np.min(X_train), np.max(X_train), df+2)
    
    def Phi(self, X=None):
        if X is None:
            X = self.X
        knots_l = []
        for i in range(self.order+1):
            knots_l.append(X**i) 
            
        for knot in self.knots[1:(self.df+1)]:
            feature = np.power( (X-knot), 3)
            feature[feature<0] = 0
            knots_l.append(feature)
        return np.stack(knots_l, axis=1)
    
    def w_hat(self):
        X_map = self.Phi()
        XTX_inv = np.linalg.pinv( np.dot(X_map.T, X_map) )
        XTY = np.dot(X_map.T, self.Y)
        w_hat = np.dot(XTX_inv, XTY)
        return w_hat
    
    def predict(self, X_test):
        X_map1 = self.Phi(X_test)
        Y_test = np.dot(X_map1, self.w_hat())
        return Y_test
    
class NSP(object):
    def __init__(self, X_train, Y_train, order, df):
        self.order = order
        self.X = X_train
        self.Y = Y_train
        self.df = df
        self.knots = np.linspace(np.min(X_train), np.max(X_train), df+2)
    
    def d(self, k, x):
        eps_k = self.knots[k]; eps_K = self.knots[self.df]
        t1 = np.power(x-eps_k, self.order)
        t1[t1<0] = 0
        t2 = np.power(x-eps_K, self.order)
        t2[t2<0] = 0
        if k == self.df:
            return np.zeros_like(x)
        return (t1-t2)/(eps_K-eps_k)
    
    def Phi(self, X=None):
        if X is None:
            X = self.X
        knots_l = []
        for i in range(2):
            knots_l.append(X**i) 
        
        for k in range(1, self.df+1):
            feature = self.d(k, X) - self.d(self.df-1, X)
            knots_l.append(feature)
        return np.stack(knots_l, axis=1)
    
    def w_hat(self):
        X_map = self.Phi()
        XTX_inv = np.linalg.pinv( np.dot(X_map.T, X_map) )
        XTY = np.dot(X_map.T, self.Y)
        w_hat = np.dot(XTX_inv, XTY)
        return w_hat
    
    def predict(self, X_test):
        X_map1 = self.Phi(X_test)
        Y_test = np.dot(X_map1, self.w_hat())
        return Y_test

class GAMs(object):
    def __init__(self, x, y, orders, dfs):
        self.X = np.atleast_2d(x).T if len(x.shape) == 1 else x
        self.Y = y
        self.orders = orders
        self.dfs = dfs
        self.dim = self.X.shape[1]
        
    def get_knots(self):
        knots = []
        for i in range(self.dim):
            knots.append(np.linspace(np.min(self.X[:,i]), np.max(self.X[:,i]), self.dfs[i]+2))   
        return knots
    
    def d(self, k, x, knots, order):
        K = len(knots) - 2
        eps_k = knots[k]; eps_K = knots[K]
        t1 = np.power(x-eps_k, order)
        t1[t1<0] = 0
        t2 = np.power(x-eps_K, order)
        t2[t2<0] = 0
        if k == K:
            return np.zeros_like(x)
        return (t1-t2)/(eps_K-eps_k)
    
    def Phi(self, X=None):
        if X is None:
            X = self.X
        knots = self.get_knots()
        Mat = []
        for i in range(self.dim):
            for j in range(2):
                Mat.append(X[:,i]**j) 
            df = self.dfs[i]
            knots_i = knots[i]
            order_i = self.orders[i]
            for k in range(1, df+1):
                feature = self.d(k, X[:,i], knots_i, order_i) - self.d(df-1, X[:,i], knots_i, order_i)
                Mat.append(feature)
        return np.stack(Mat, axis=1)
    
    def w_hat(self):
        X_map = self.Phi()
        XTX_inv = np.linalg.pinv( np.dot(X_map.T, X_map) )
        XTY = np.dot(X_map.T, self.Y)
        w_hat = np.dot(XTX_inv, XTY)
        return w_hat
    
    def predict(self, X_test):
        X_test = np.atleast_2d(X_test).T if len(X_test.shape) == 1 else X_test
        X_map1 = self.Phi(X_test)
        Y_test = np.dot(X_map1, self.w_hat())
        return Y_test   
        
        
        
        
        
    
    
    
    
