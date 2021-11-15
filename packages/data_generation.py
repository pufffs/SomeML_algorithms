# -*- coding: utf-8 -*-
"""
Created on Tue May  4 18:05:04 2021

@author: roone
"""
import numpy as np
class DG(object):
    def __init__(self, point_num, noise, input_dim):
        self.n = point_num
        self.noise= noise
        self.dim = input_dim
        self.f1 = lambda x: 1 + x + 0.3 * np.sin(10*x) #used in the feature selection notebook.
        self.f2 = lambda x: x**2

    def polydata_generation(self, w_true): #used in the LR and GD notebook.
        dim = w_true.size- 1 #the dimension of feasure space excluding the intercept.
        x= np.random.normal(0., 1., size=(self.n,1))
        x1= np.ones_like(x)
        for i in range(1, dim+1):
            x1= np.concatenate( ( x1, np.power(x, i) ), axis= 1) #X= [1, x]
        y= np.dot(x1, w_true)+ np.random.normal(0., self.noise, self.n) #y= Xw+ eps
        return x1, y

    def Gaussianbased_data1(self, mean, var):
        x = np.random.normal(mean, var, size=(self.n, ))
        y = self.f1(x) + self.noise * np.random.randn(self.n)#here use self.f1 for convinience. can also define f2 for other types.
        return x, y
    
    def D2_data(self, mean, var):
        x = np.random.normal(mean, var, size=(self.n, 2))
        y = self.f1(x[:,0]) + self.f2(x[:,1]) + self.noise * np.random.randn(self.n)
        return x, y
    
    def linear_seperable(self, mean, var, offset, negative_num= None):
        """Default argument values are evaluated at function define-time, 
        but self is an argument only available at function call time. Thus 
        arguments in the argument list cannot refer each other."""
        if negative_num is None:
            negative_num = self.n 
        x_p = offset + np.random.normal(mean,var, size=(self.n, self.dim)) + np.atleast_2d(np.random.normal(0., self.noise, self.n)).T
        x_n = np.random.normal(mean, var, size=(negative_num, self.dim)) + np.atleast_2d(np.random.normal(0., self.noise, negative_num)).T
        y_p = np.ones((self.n), dtype=np.int)
        y_n = -1 * np.ones((negative_num), dtype=np.int)
        x = np.concatenate((x_p, x_n))
        x = np.concatenate( ( x, np.ones((x.shape[0],1)) ), axis=1)
        y = np.concatenate((y_p, y_n))
        return x, y