# Just-in-Time Partial Least Squares (JIT-PLSR)

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.base import RegressorMixin
from plsr_ext.utils import find_K_nearest_samples
from plsr_ext.plsr import PLSR
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import GridSearchCV

class JIT_PLSR(RegressorMixin):
    """Just-in-time learning-integrated partial least squares regression.
    
    This implementation considered only one response variable in the
    target variable.

    Parameters
    ----------
    max_n_components : int, default=5
        Maximum number of components to keep. When building each local 
        PLSR model for each testing sample, its number of latent components
        will be fine-tuned in the range of [1, max_n_components].

    k_nearest : int, default=10
        Number of k nearest neighbors to execute the construction of LWPLSR.

    k_fold : int, default-5
        Number of k-fold cross-validation will be used to find the optimal 
        number of latent components for each local PLSR.

    scoring : str, callable, list, tuple or dict, default='neg_mean_absolute_percentage_error'
        Strategy to evaluate the performance of the cross-validated model on the validation set
        for hyper-parameter tuning of the local models.
        See more information in `link <https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics>`_

    sim_metric : str, or callable, default='euclidean'
        Distance metric is used in the similarity measure.
        If a string, the distance function can be 'braycurtis', 
        'canberra', 'chebyshev', 'cityblock', 'correlation', 'cosine',
        'dice', 'euclidean', 'hamming', 'jaccard', 'jensenshannon', 
        'kulczynski1', 'mahalanobis', 'matching', 'minkowski', 
        'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 
        'sokalsneath', 'sqeuclidean', 'yule'.
    
    n_jobs : int, default=1
        Number of jobs to run in parallel for parameter tuning.
        -1 means using all processors. See Glossary for more details.

    Attributes
    ----------
    x_train : ndarray of shape (n_samples, n_features)
        Input features of training data.

    y_train: ndarray of shape (n_samples, 1)
        Target variable of training data.

    x_train_mean: ndarray of shape (1, n_features)
        Mean values of all input features.
    
    x_train_std: ndarray of shape (1, n_features)
        Standard deviation values of all input features

    y_train_mean: float
        Mean of all target values.

    y_train_std: float
        Standard deviation of all target values

    scaled_x_train: ndarray of shape (n_samples, n_features)
        Input features of training data were scaled to unit variance.

    scaled_y_train: ndarray of shape (n_samples, 1)
        Target variable of training data were scaled to unit variance.

    """
    def __init__(self, max_n_components=5, k_nearest=10, k_fold=5, scoring='neg_mean_absolute_percentage_error', sim_metric='euclidean', n_jobs=1):
        self.max_n_components = max_n_components
        self.k_nearest = k_nearest
        self.k_fold = k_fold
        self.scoring = scoring
        self.sim_metric = sim_metric
        self.n_jobs = n_jobs

    def fit(self, X, y):
        """Fit model to data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vectors, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : array-like of shape (n_samples,)
            Target vector, where `n_samples` is the number of samples.

        Returns
        -------
        self : object
            Fitted model.
        """
        X = np.array(X)
        y = np.array(y)
        y = np.reshape(y, (len(y), 1))

        if X.ndim == 1:
            X = X.reshape(1, -1)

        rank_upper_bound = X.shape[1]
        
        if self.max_n_components > rank_upper_bound:
            raise ValueError(
                f"`max_n_components` upper bound is {rank_upper_bound}. "
                f"Got {self.max_n_components} instead. Reduce `max_n_components`."
            )
        
        if self.k_nearest > X.shape[0]:
            raise ValueError(
                f"`k_nearest` upper bound is {X.shape[0]}. "
                f"Got {self.k_nearest} instead. Reduce `k_nearest`."
            )
        
        # Performing a unit scaling for the training data
        self.x_train = X.copy()
        self.y_train = y.copy()

        self.x_train_mean = self.x_train.mean(axis=0)
        self.x_train_std = self.x_train.std(axis=0, ddof=1)

        self.y_train_mean = self.y_train.mean()
        self.y_train_std = self.y_train.std(ddof=1)

        self.scaled_x_train = (self.x_train - self.x_train_mean) / self.x_train_std
        self.scaled_y_train = (self.y_train - self.y_train_mean) / self.y_train_std

        return self
    
    def update(self, X, y):
        """Update model to new data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            New input features vectors, where `n_samples` is the number of samples and
            `n_features` is the number of features.

        y : array-like of shape (n_samples,)
            New target vector, where `n_samples` is the number of samples.

        Returns
        -------
        self : object
            Updated model.
        """
        X = np.array(X)
        y = np.array(y)
        y = np.reshape(y, (len(y), 1))

        if X.ndim == 1:
            X = X.reshape(1, -1)

        # Add new data to the list
        self.x_train = np.vstack((self.x_train, X))
        self.y_train = np.vstack((self.y_train, y))

        # Update the scaled data
        self.x_train_mean = self.x_train.mean(axis=0)
        self.x_train_std = self.x_train.std(axis=0, ddof=1)

        self.y_train_mean = self.y_train.mean()
        self.y_train_std = self.y_train.std(ddof=1)

        self.scaled_x_train = (self.x_train - self.x_train_mean) / self.x_train_std
        self.scaled_y_train = (self.y_train - self.y_train_mean) / self.y_train_std

        return self
    
    def _build_local_plsr(self, X_train, y_train):
        """
        Build a local PLSR model with k-fold cross-validation for 
        hyper-parameter tuning

        Parameters
        ----------

        X_train : array-like of shape (n_samples, n_features)
            Training input features vectors, where `n_samples` is the 
            number of samples and `n_features` is the number of features.

        y_train : array-like of shape (n_samples,)
            Training target vector, where `n_samples` is the number of samples.

        Returns
        -------
        local_model : object
            The trained local model.

        """
        parameters = {'n_components': np.arange(1, self.max_n_components + 1, 1)}
        plsr_model = PLSR(scale=False)
        clf = GridSearchCV(estimator=plsr_model, param_grid=parameters, scoring=self.scoring, cv=self.k_fold, refit=False, n_jobs=self.n_jobs)
        clf.fit(X_train, y_train)
        best_params = clf.best_params_
        local_model = PLSR(**best_params, scale=False)
        local_model.fit(X_train, y_train)
        return local_model

    
    def predict(self, X):
        """Predict targets of given samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Testing samples.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Returns predicted values.

        """
        X = np.array(X)

        if X.ndim == 1:
            X = X.reshape(1, -1)

        # Scaling testing data
        X = (X - self.x_train_mean) / self.x_train_std
        y_pred = np.zeros(X.shape[0])
        for test_sample_number in range(X.shape[0]):
            query_x_test = X[test_sample_number, :]
            query_x_test = np.reshape(query_x_test, (1, len(query_x_test)))

            ids_k_nearest, selected_x_train, _ = find_K_nearest_samples(self.scaled_x_train, query_x_test, self.k_nearest, self.sim_metric)
            selected_y_train = self.scaled_y_train[ids_k_nearest]

            # build a local PLSR model from the selected training samples
            local_model = self._build_local_plsr(selected_x_train, selected_y_train)
            pred_res = local_model.predict(query_x_test).ravel()
            pred_res = pred_res * self.y_train_std + self.y_train_mean
            y_pred[test_sample_number] = pred_res

        return y_pred
