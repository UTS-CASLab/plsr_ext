# Locally-Weighted Partial Least Squares (LWPLS)

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.base import RegressorMixin

class LWPLSR(RegressorMixin):
    """Locally-Weighted PLS regression.
    
    This implementation considered only one response variable in the
    target variable.

    Parameters
    ----------
    n_components : int, default=2
        Number of components to keep. Should be in `[1, min(n_samples,
        n_features)]`.

    lambda_in_similarity : float, default=1
        The scaling parameter in similarity matrix.

    sim_metric : str, or callable, default='euclidean'
        Distance metric is used in the similarity measure.
        If a string, the distance function can be 'braycurtis', 
        'canberra', 'chebyshev', 'cityblock', 'correlation', 'cosine',
        'dice', 'euclidean', 'hamming', 'jaccard', 'jensenshannon', 
        'kulczynski1', 'mahalanobis', 'matching', 'minkowski', 
        'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 
        'sokalsneath', 'sqeuclidean', 'yule'.

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

    y_pred_all_components: ndarray of shape (n_test_samples, n_components):
        Predicted outcomes of all components up to n_components for unseen testing data.

    """
    def __init__(self, n_components=2, lambda_in_similarity=1, sim_metric='euclidean'):
        self.n_components = n_components
        self.lambda_in_similarity = lambda_in_similarity
        self.sim_metric = sim_metric

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
        
        if self.n_components > rank_upper_bound:
            raise ValueError(
                f"`n_components` upper bound is {rank_upper_bound}. "
                f"Got {self.n_components} instead. Reduce `n_components`."
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

        Notes
        -----
        The similarity measure in this algorithm is:
        :math:`exp^{\\cfrac{-d}{\\lambda \\cdot \\sigma}}`

        """
        X = np.array(X)

        if X.ndim == 1:
            X = X.reshape(1, -1)

        # Scaling testing data
        X = (X - self.x_train_mean) / self.x_train_std

        self.y_pred_all_components = np.zeros((X.shape[0], self.n_components))
        distance_matrix = cdist(self.scaled_x_train, X, self.sim_metric)
        
        for test_sample_number in range(X.shape[0]):
            query_x_test = X[test_sample_number, :]
            query_x_test = np.reshape(query_x_test, (1, len(query_x_test)))
            distance = distance_matrix[:, test_sample_number]
            similarity = np.diag(np.exp(-distance / distance.std(ddof=1) / self.lambda_in_similarity))
            
            y_w = self.scaled_y_train.T.dot(np.diag(similarity)) / similarity.sum()
            x_w = np.reshape(self.scaled_x_train.T.dot(np.diag(similarity)) / similarity.sum(), (1, self.scaled_x_train.shape[1]))
            centered_y = self.scaled_y_train - y_w
            centered_x = self.scaled_x_train - np.ones((self.scaled_x_train.shape[0], 1)).dot(x_w)
            centered_query_x_test = query_x_test - x_w
            self.y_pred_all_components[test_sample_number, :] += y_w

            for component_number in range(self.n_components):
                w_a = np.reshape(centered_x.T.dot(similarity).dot(centered_y) / np.linalg.norm(centered_x.T.dot(similarity).dot(centered_y)), (self.scaled_x_train.shape[1], 1))
                t_a = np.reshape(centered_x.dot(w_a), (self.scaled_x_train.shape[0], 1))
                p_a = np.reshape(centered_x.T.dot(similarity).dot(t_a) / t_a.T.dot(similarity).dot(t_a), (self.scaled_x_train.shape[1], 1))
                q_a = centered_y.T.dot(similarity).dot(t_a) / t_a.T.dot(similarity).dot(t_a)
                t_q_a = centered_query_x_test.dot(w_a)
                
                self.y_pred_all_components[test_sample_number, component_number:] = self.y_pred_all_components[test_sample_number, component_number:] + t_q_a * q_a
                
                if component_number != self.n_components - 1:
                    centered_x = centered_x - t_a.dot(p_a.T)
                    centered_y = centered_y - t_a * q_a
                    centered_query_x_test = centered_query_x_test - t_q_a.dot(p_a.T)

        y_pred = self.y_pred_all_components[:, self.n_components - 1] * self.y_train_std + self.y_train_mean

        for i in range(self.n_components):
            self.y_pred_all_components[:, i] = self.y_pred_all_components[:, i] * self.y_train_std + self.y_train_mean

        return y_pred
    
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
