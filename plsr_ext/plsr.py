# Partial Least Squares (rPLSR)

import numpy as np
from sklearn.base import RegressorMixin
from sklearn.base import BaseEstimator
import math

class PLSR(BaseEstimator, RegressorMixin):
    """Partial least squares regression.
    
    Parameters
    ----------
    n_components : int, default=2
        Number of components to keep.

    max_iter : int, default=500
        The maximum number of iterations.

    tol : float, default=1e-06
        The tolerance used as convergence criteria in the power method: the
        algorithm stops whenever the squared norm of `u_i - u_{i-1}` is less
        than `tol`, where `u` corresponds to the left singular vector.

    scale : bool, default=True
        Whether to scale `X` and `Y`.

    Attributes
    ----------

    x_mean: ndarray of shape (1, n_features)
        Mean values of all input features.
    
    x_std: ndarray of shape (1, n_features)
        Standard deviation values of all input features

    y_mean: ndarray of shape (1, n_targets)
        Mean of all target variables.

    y_std: float
        Standard deviation of all target variables.

    n_samples : int
        Number of training samples.

    n_targets : int
        Number of target variables.

    n_features : int
        Number of input features.

    P : ndarray of shape (n_features, n_components)
        The loadings of `X`.

    W : ndarray of shape (n_features, n_components)
        The weighting matrix of `X`.

    Q : ndarray of shape (n_targets, n_components)
        The loadings of `Y`.

    T : ndarray of shape (n_samples, n_components)
        The matrix consisting of scores of `X`.

    U : ndarray of shape (n_samples, n_components)
        The matrix consisting of scores of `Y`.

    b : ndarray of shape (n_components,)
        The regression coefficients of all components.

    C : ndarray of shape (n_features, n_targets)
        The matrix of regression coefficients such that `Y` 
        is approximated as `Y = X @ C`.

    """
    def __init__(self, n_components=2, max_iter=500, tol=1e-06, scale=True):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.scale = scale

    def data_unit_scaling(self, X, Y):
        """
        Scaling the data to mean of 0 and unit variance.
        """
        self.x_mean = X.mean(axis=0)
        self.x_std = X.std(axis=0, ddof=1)

        self.y_mean = Y.mean(axis=0)
        self.y_std = Y.std(axis=0, ddof=1)

        X_norm = (X - self.x_mean) / self.x_std
        Y_norm = (Y - self.y_mean) / self.y_std

        return X_norm, Y_norm
    
    def _fit(self, X, Y):
        """
        Fit a PLSR model

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Normalised training vectors, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        Y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Normalised target vectors, where `n_samples` is the number of samples and
            `n_targets` is the number of response variables.

        Returns
        -------
        self : object
            Fitted model.

        """
        n_samples = X.shape[0]
        n_features = X.shape[1]
        n_targets = Y.shape[1]
        P = np.zeros((n_features, self.n_components))
        W = np.zeros((n_features, self.n_components))
        Q = np.zeros((n_targets, self.n_components))
        U = np.zeros((n_samples, self.n_components))
        T = np.zeros((n_samples, self.n_components))
        b = np.zeros((self.n_components))
        np.random.seed(0)
        for it in range(self.n_components):
            u_tmp = Y[:,np.random.randint(n_targets)].reshape(n_samples,1)      
            for _ in range(self.max_iter):
                p_tmp = np.matmul(X.T, u_tmp)
                p_tmp /= np.linalg.norm(p_tmp, 2)

                t_tmp = np.matmul(X, p_tmp)

                q_tmp = np.matmul(Y.T, t_tmp)
                q_tmp /= np.linalg.norm(q_tmp, 2)

                u_new = np.matmul(Y, q_tmp)
                if np.linalg.norm(u_tmp-u_new, 2) <= self.tol:
                    u_tmp = u_new
                    break
                u_tmp = u_new
            
            W[:,it] = p_tmp.reshape((n_features))
            Q[:,it] = q_tmp.reshape((n_targets))
            U[:,it] = u_tmp.reshape((n_samples))
            T[:,it] = t_tmp.reshape((n_samples))

            t_sum = np.sum(t_tmp*t_tmp)
            # Find the inner model
            b[it] = np.sum(u_tmp*t_tmp) / t_sum
            # Compute X loadings
            P[:,it] = (np.matmul(np.transpose(X), t_tmp) / t_sum).reshape((n_features))
            # Calculate the residuals
            X -= np.matmul(t_tmp, P[:,it].reshape(1, n_features))
            Y -= b[it] * np.matmul(t_tmp, q_tmp.T)      
        
        self.P = P[:,0:self.n_components]
        self.W = W[:,0:self.n_components]
        self.Q = Q[:,0:self.n_components]
        self.U = U[:,0:self.n_components]
        self.T = T[:,0:self.n_components]
        self.b = b[0:self.n_components]
        self.C = np.matmul( \
                    np.matmul(self.W, np.linalg.inv(np.matmul(self.P.T, self.W))), \
                    np.matmul(np.diag(self.b), self.Q.T) \
                    )
        
        return self

    def fit(self, X, Y):
        """Fit model to data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        Y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target vectors, where `n_samples` is the number of samples and
            `n_targets` is the number of response variables..

        Returns
        -------
        self : object
            Fitted model.
        """
        X = np.array(X)
        Y = np.array(Y)
        
        if X.ndim == 1:
            X = X.reshape(1, -1)

        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)

        rank_upper_bound = X.shape[1]
        
        if self.n_components > rank_upper_bound:
            raise ValueError(
                f"`max_n_components` upper bound is {rank_upper_bound}. "
                f"Got {self.n_components} instead. Reduce `max_n_components`."
            )
        
        self.n_samples, self.n_features = X.shape
        self.n_targets = Y.shape[1]
        
        if self.scale == True:
            # Performing a unit scaling for the training data
            X_norm, Y_norm = self.data_unit_scaling(X, Y)
            self._fit(X_norm, Y_norm)
        else:
            self._fit(X, Y)

        return self

    def predict(self, X):
        """Predict targets of given samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,) or (n_samples, n_targets)
            Returns predicted values.
        """
        typeX = type(X)

        if typeX == list:
            X = np.array(X)
        elif typeX == np.float64 or typeX == float or typeX == np.int64 or typeX == int:
            X = np.array([X])

        if self.scale == True:
            # Normalize
            X -= self.x_mean
            X /= self.x_std
            y_pred = self.y_mean + np.matmul(X, self.C) * self.y_std
        else:
            y_pred = np.matmul(X, self.C)
        
        if self.n_targets == 1:
            return y_pred.ravel()
        else:
            return y_pred
