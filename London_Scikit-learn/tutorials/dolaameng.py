#
# Forked from dolaameng's script
# Please refer to https://github.com/dolaameng/fun_with_kaggle/tree/master/sklearn-london
#

import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.decomposition import PCA


from autodiff import optimize
from scipy import stats
import pylab as pl

def soft_absolute(u):
	epsilon = 1e-8
	return np.sqrt(u*u+epsilon)

def logistic(u):
	return 1. / (1. + np.exp(-u))

class SparseFilter(BaseEstimator, TransformerMixin):
	def __init__(self, n_features=200, n_iterations=300, activate=soft_absolute):
		self.epsilon = 1e-8
		self.n_features = n_features
		self.n_iteration = n_iterations
		self.activate = activate

	def fit(self, X, y = None):
		n_samples, n_dim = X.shape
		W = np.random.randn(n_dim, self.n_features)
		b = np.random.randn(self.n_features)
		obj_fn = self.get_objective_fn(X)
		self.W_, self.b_ = optimize.fmin_l_bfgs_b(obj_fn, (W, b),
								iprint = 1,
								maxfun = self.n_iteration)
		return self

	def get_objective_fn(self, X):
		def _objective_fn(W, b):
			Y = self.activate(np.dot(X, W) + b)
			Y = Y / np.sqrt(np.sum(Y*Y, axis=0) + self.epsilon)
			Y = Y / np.sqrt(np.sum(Y*Y, axis=1)[:, np.newaxis] + self.epsilon)

			return np.sum(Y)
		return _objective_fn

	def transform(self, X):
		Y = self.activate(np.dot(X, self.W_) + self.b_)
		Y = Y / np.sqrt(np.sum(Y*Y, axis=0) + self.epsilon)
		Y = Y / np.sqrt(np.sum(Y*Y, axis=1)[:,np.newaxis] + self.epsilon)

		return Y
# end of class

