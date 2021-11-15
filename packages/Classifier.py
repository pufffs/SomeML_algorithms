from .Regularizers import Regularizer
import numpy as np

class Classifier(object):
    def __init__(self, X, Y, reg=0, regularizer=Regularizer):
        self._Xtr = X
        self._Ytr = Y
        self._Xtest = None
        self._Ytest = None
        self._w = None
        self._n = X.shape[0]
        self._regularizer = regularizer
        self._reg = reg
    
    @property
    def n(self):
        return self._n
    
    @property
    def XYtest(self):
        return self._Xtest, self._Ytest
    @XYtest.setter
    def XYtest(self, value):
        self._Xtest, self._Ytest = value

    @property
    def w(self):
        return self._w
    @w.setter
    def w(self, value):
        self._w = value
    @w.deleter
    def w(self):
        del self._w
        
    def loss(self):
        pass
    def gradient(self):
        pass
    def predict(self):
        pass
    def test_loss(self):
        pass
    
class Perceptron(Classifier):
    def __init__(self, X, Y):
        super().__init__(X, Y)
        self._w = np.random.randn(X.shape[1])
        
    def loss(self, w=None, indexes=None):
        if indexes is None:
            indexes = np.arange(self._n)
        if w is None:
            w = self._w
        error = -np.dot(self._Xtr[indexes,:], w) * self._Ytr[indexes]
        error[error<0] = 0
        return np.mean(error)
        
    def gradient(self, w=None, indexes=None):
        if indexes is None:
            indexes = np.arange(self._n)
        if w is None:
            w = self._w
        error = -np.dot(self._Xtr[indexes,:], w) * self._Ytr[indexes]
        gradient = - self._Xtr[indexes,:] * self._Ytr[indexes, np.newaxis]
        gradient[error<0] = 0
        return np.mean(gradient, axis=0) + self._regularizer.gradient(w)
        
    def predict(self, X, w=None):
        if w is None:
            w = self._w
        z = np.dot(X, w)
        return np.sign(z)
        
    def test_loss(self, Xtest, Ytest, w=None):
        if w is None:
            w = self._w
        num = Xtest.shape[0]
        error = -np.dot(Xtest, w) * Ytest
        error[error<0] = 0
        return np.mean(error)
    
    def toyfit(self, X, Y, eta=0.1,itera=1000):
        for i in range(itera):
            self._x = self._x - eta * self.gradient(self._x) 
    
    
class SVM(Classifier):
    def __init__(self, X, Y):
        super().__init__(X, Y)
        self._w = np.random.randn(X.shape[1])
        
    def loss(self, w=None, indexes=None):
        if indexes is None:
            indexes = np.arange(self._n)
        if w is None:
            w = self._w
        error = 1-np.dot(self._Xtr[indexes,:], w) * self._Ytr[indexes]
        error[error<0] = 0
        return np.mean(error)
        
    def gradient(self, w=None, indexes=None):
        if indexes is None:
            indexes = np.arange(self._n)
        if w is None:
            w = self._w
        error = 1-np.dot(self._Xtr[indexes,:], w) * self._Ytr[indexes]
        gradient = - self._Xtr[indexes,:] * self._Ytr[indexes, np.newaxis]
        gradient[error<0] = 0
        return np.mean(gradient, axis=0) + self._regularizer.gradient(w)
    """
    what if we change linear features to other type of features? say f^(x)!=w^T*x, instead we use some h(x)
    to replace linear features x: f^(x)=w^T*h(x). h(x) may contain [x1, x2^2, sin(x3)...] etc. And some
    data that is not linearly-seperable can be seperable on other features. for example circled data, if
    we map original features [x1,x2] to [x1^2, x2^2] and use this as the loss function to train and predict.
    Our final prediction will look like a circle when used to plot on original data(axis [x1,x2]). I tried
    simple modification on function gradient1(), changing its loss function and gradient. It turned out very
    well when I used the result to predict, it can also be used to classify linearly-seperable data by drawing
    circles.
    This way we can successfully split circled data which cannot be sovled by linear feature. So the question
    comes: can we find a way that can be deployed for any kind of features? Kernel method!
    """
    def gradient1(self, w=None, indexes=None):
        if indexes is None:
            indexes = np.arange(self._n)
        if w is None:
            w = self._w
        error = 1-np.dot(self._Xtr[indexes,:]**2, w) * self._Ytr[indexes]
        gradient = - (self._Xtr[indexes,:]**2) * self._Ytr[indexes, np.newaxis]
        gradient[error<0] = 0
        return np.mean(gradient, axis=0)
        
    def predict(self, X, w=None):
        if w is None:
            w = self._w
        z = np.dot(X, w)
        return np.sign(z)
        
    def test_loss(self, w=None):
        if w is None:
            w = self._w
        error = 1 - np.dot(self._Xtest, w) * self._Ytest
        error[error<0] = 0
        return np.mean(error)

class Logistic(Classifier):
    def __init__(self, X, Y):
        super().__init__(X, Y)
        self._w = np.random.randn(X.shape[1])
        
    def loss(self, w=None, indexes=None):
        if indexes is None:
            indexes = np.arange(self._n)
        if w is None:
            w = self._w
        z = -np.dot(self._Xtr[indexes,:], w) * self._Ytr[indexes] #can directly use the matrix form and divided by n because
        error = np.log(1+np.exp(z))                               #all operations are component-wise
        return np.mean(error,axis=0)
        
    def gradient1(self, w=None, indexes=None):
        if indexes is None:
            indexes = np.arange(self._n)
        if w is None:
            w = self._w
        z = np.dot(self._Xtr[indexes,:], w) * self._Ytr[indexes]
        z_gd = self._Xtr[indexes,:] * self._Ytr[indexes, np.newaxis]
        dim_expand = (np.exp(-z) / (1+np.exp(-z)))[:, np.newaxis]
        total_gd = -z_gd * dim_expand
        return np.sum(total_gd, axis=0) / indexes.size
    
    def gradient(self, w=None, indexes=None):
        if indexes is None:
            indexes = np.arange(self._n)
        if w is None:
            w = self._w
        z = -np.dot(self._Xtr[indexes,:], w) * self._Ytr[indexes]
        z_gd = -self._Xtr[indexes,:] * self._Ytr[indexes, np.newaxis]
        term = np.exp(z)/(1+np.exp(z))
        some = np.expand_dims(term, axis=1) * z_gd
        return np.mean(some, axis=0) + self._regularizer.gradient(w)
    
    def predict(self, X, w=None):
        if w is None:
            w = self._w
        z = -np.dot(X, w)
        return 1/(1+np.exp(z)) #we predict out the probability of positive prediction
        
    def test_loss(self, w=None):
        if w is None:
            w = self._w
        num = self._Xtest.shape[0]
        z = -np.dot(self._Xtest, w) * self._Ytest
        error = np.log(1+np.exp(z))
        return np.mean(error, axis=0)

        
        
        