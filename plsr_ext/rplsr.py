# Recursive Partial Least Squares (rPLSR)

import numpy as np
from sklearn.base import RegressorMixin
from plsr_ext import PLSR
import math

class RPLSR(RegressorMixin):
    """Recursive partial least squares regression.
    
    Parameters
    ----------
    n_components : int, default=2
        Number of components to keep.

    forgetting_lambda : float, default = 1
        Forgetting factor for old data.

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
    def __init__(self, n_components=2, forgetting_lambda=1, max_iter=500, tol=1e-06, scale=True):
        self.n_components = n_components
        self.forgetting_lambda = forgetting_lambda
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
    
    def update_mean_std(self, new_X, new_Y):
        """
        Update the mean and std values of training data according to new data

        Parameters
        ----------
        new_X : array-like of shape (1, n_features)
            Original new training vectors, where `n_samples` is the number of new samples and
            `n_features` is the number of features.

        new_Y : array-like of shape (1,) or (1, n_targets)
            Orignal new target vectors, where `n_samples` is the number of new samples and
            `n_targets` is the number of response variables.
        """
        typeY = type(new_Y)

        if typeY == list:
            new_Y = np.array(new_Y)
        elif typeY == np.float64 or typeY == float or typeY == np.int64 or typeY == int:
            new_Y = np.array([new_Y])

        old_X_mean = self.x_mean.copy()
        old_Y_mean = self.y_mean.copy()
        self.n_samples = self.n_samples + 1
        self.x_mean = 1/self.n_samples * (new_X + (self.n_samples - 1) * old_X_mean)
        self.y_mean = 1/self.n_samples * (new_Y + (self.n_samples - 1) * old_Y_mean)
        self.x_std = np.sqrt((self.n_samples - 2) / (self.n_samples - 1) * self.x_std * self.x_std + \
                                1/self.n_samples * (new_X - old_X_mean) *  (new_X - old_X_mean))
        self.y_std = np.sqrt((self.n_samples - 2) / (self.n_samples - 1) * self.y_std * self.y_std + \
                                1/self.n_samples * (new_Y - old_Y_mean) *  (new_Y - old_Y_mean)) 
        

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
        
        plsr_model = PLSR(n_components=self.n_components, max_iter=self.max_iter, tol=self.tol)
        if self.scale == True:
            # Performing a unit scaling for the training data
            X_norm, Y_norm = self.data_unit_scaling(X, Y)
            plsr_model._fit(X_norm, Y_norm)
        else:
            plsr_model._fit(X, Y)

        # store data
        self.P = plsr_model.P.copy()
        self.Q = plsr_model.Q.copy()
        self.W = plsr_model.W.copy()
        self.U = plsr_model.U.copy()
        self.T = plsr_model.T.copy()
        self.b = plsr_model.b.copy()
        self.C = plsr_model.C.copy() 

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
        else:
            X = X.copy()

        if self.scale == True:
            # Normalize
            scale_X = (X - self.x_mean) / self.x_std
            y_pred = self.y_mean + np.matmul(scale_X, self.C) * self.y_std
        else:
            y_pred = np.matmul(X, self.C)
        
        if self.n_targets == 1:
            return y_pred.ravel()
        else:
            return y_pred
        
    def update(self, new_X, new_Y, update_mean_std=True):
        """
        Update the trained model according to new input values

        Parameters
        ----------
        new_X : array-like of shape (n_samples, n_features)
            Original new training vectors, where `n_samples` is the number of new samples and
            `n_features` is the number of features.

        new_Y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Orignal new target vectors, where `n_samples` is the number of new samples and
            `n_targets` is the number of response variables.

        update_mean_std : bool, default=True
            Whether update mean and standard deviation of all learned data.
            This variable only has the power if scale parameter is True.

        Returns
        -------
        self : object
            Fitted model.
        """
        typeY = type(new_Y)

        if typeY == list:
            new_Y = np.array(new_Y)
        elif typeY == np.float64 or typeY == float or typeY == np.int64 or typeY == int:
            new_Y = np.array([new_Y])

        if new_X.ndim == 1:
            new_X = new_X.reshape(1, -1)

        if new_Y.ndim == 1:
            new_Y = new_Y.reshape(-1, 1)

        for i in range(new_X.shape[0]):
            if self.scale == True:
                scale_X_i = (new_X[i] - self.x_mean) / self.x_std
                scale_Y_i = (new_Y[i] - self.y_mean) / self.y_std

                X = np.vstack((self.forgetting_lambda * self.P.T , scale_X_i))
                Y = np.vstack((self.forgetting_lambda * np.matmul(np.diag(self.b), self.Q.T), scale_Y_i))

                if update_mean_std == True:
                    self.update_mean_std(new_X[i], new_Y[i])
                    X = (X - self.x_mean) / self.x_std
                    Y = (Y - self.y_mean) / self.y_std
            else:
                X = np.vstack((self.forgetting_lambda * self.P.T, new_X[i]))
                Y = np.vstack((self.forgetting_lambda * np.matmul(np.diag(self.b), self.Q.T), new_Y[i]))

            # Training the PLSR model
            plsr_model = PLSR(n_components=self.n_components, max_iter=self.max_iter, tol=self.tol)
            plsr_model._fit(X, Y)

            self.b = plsr_model.b.copy()
            self.P = plsr_model.P.copy()
            self.Q = plsr_model.Q.copy()

        # copy the final model
        self.P = plsr_model.P.copy()
        self.Q = plsr_model.Q.copy()
        self.W = plsr_model.W.copy()
        self.U = plsr_model.U.copy()
        self.T = plsr_model.T.copy()
        self.b = plsr_model.b.copy()
        self.C = plsr_model.C.copy() 

        return self
