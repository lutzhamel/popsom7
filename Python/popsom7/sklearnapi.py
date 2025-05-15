from sklearn.base import BaseEstimator, ClusterMixin
import pandas as pd
from .maputils import (map_build, 
                       map_fitted, 
                       map_predict, 
                       map_position,
                       map_summary, 
                       map_starburst, 
                       map_significance, 
                       map_marginal)

class SOM(BaseEstimator, ClusterMixin):
    """
    A scikit-learn style wrapper for the SOM routines defined in map_utils.py.
    
    This unsupervised learning algorithm builds a self-organizing map (SOM) of the 
    training data. If labels (y) are provided during fit, they are used for reporting 
    (e.g. for majority voting in clusters). Otherwise, clusters are labeled numerically.
    
    Parameters
    ----------
    xdim : int, default=10
        Number of neurons along the x-dimension of the map.
    ydim : int, default=5
        Number of neurons along the y-dimension of the map.
    alpha : float, default=0.3
        Learning rate (0 < alpha <= 1).
    train : int, default=1000
        Number of training iterations.
    normalize : bool, default=False
        Whether to normalize the input data by row.
    seed : int or None, default=None
        Seed for reproducibility.
    
    Attributes
    ----------
    som_map_ : dict
        The fitted SOM model (the map object returned by map_build).
    """
    
    def __init__(self, xdim=10, ydim=5, alpha=0.3, train=1000, normalize=False, seed=None):
        self.xdim = xdim
        self.ydim = ydim
        self.alpha = alpha
        self.train = train
        self.normalize = normalize
        self.seed = seed
        self.som_map_ = None  # Will hold the fitted map

    def fit(self, X, y=None):
        """
        Fit the SOM model using the training data.
        
        Parameters
        ----------
        X : array-like or pd.DataFrame, shape (n_samples, n_features)
            Training data.
        y : array-like or pd.DataFrame, shape (n_samples,) or (n_samples, 1), default=None
            Optional labels for the training data (used for reporting purposes).
        
        Returns
        -------
        self : object
            Returns the instance itself.
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if y is not None and not isinstance(y, pd.DataFrame):
            y = pd.DataFrame(y)
            
        # Use the existing map_build routine.
        self.som_map_ = map_build(X, labels=y, xdim=self.xdim, ydim=self.ydim,
                                  alpha=self.alpha, train=self.train,
                                  normalize=self.normalize, seed=self.seed)
        return self

    def fit_predict(self, X, y=None):
        """
        Fit the SOM model and return cluster assignments for the training data.
        
        Parameters
        ----------
        X : array-like or pd.DataFrame, shape (n_samples, n_features)
            Training data.
        y : array-like or pd.DataFrame, default=None
            Optional labels for the training data.
        
        Returns
        -------
        labels : array-like
            Cluster labels for each training sample.
        """
        self.fit(X, y)
        # Use map_fitted to get the cluster assignments for the training data.
        return map_fitted(self.som_map_)

    def predict(self, X):
        """
        Map new samples to clusters.
        
        Parameters
        ----------
        X : array-like or pd.DataFrame, shape (n_samples, n_features)
            New data.
        
        Returns
        -------
        labels : pd.Series or list
            Cluster labels assigned to each sample.
        """
        if self.som_map_ is None:
            raise ValueError("Model is not fitted yet. Please call fit before predict.")
        if not isinstance(X, pd.DataFrame):
            # Ensure the new data has the same column names as the training data.
            X = pd.DataFrame(X, columns=self.som_map_['data'].columns)
        # map_predict returns a DataFrame with columns "Label" and "Confidence"
        predictions = map_predict(self.som_map_, X)
        return predictions["Label"]

    def transform(self, X):
        """
        Map the samples to their positions (coordinates) on the SOM grid.
        
        Parameters
        ----------
        X : array-like or pd.DataFrame, shape (n_samples, n_features)
            New data.
        
        Returns
        -------
        positions : pd.DataFrame
            A DataFrame with columns "x-dim" and "y-dim" indicating the grid positions.
        """
        if self.som_map_ is None:
            raise ValueError("Model is not fitted yet. Please call fit before transform.")
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.som_map_['data'].columns)
        return map_position(self.som_map_, X)

    def summary(self, verbose=True):
        """
        Print and return a summary of the SOM training parameters and quality assessments.
        
        Parameters
        ----------
        verbose : bool, default=True
            If True, prints the summary.
        
        Returns
        -------
        summary_dict : dict
            A dictionary containing summary information.
        """
        return map_summary(self.som_map_, verb=verbose)

    def starburst(self):
        """
        Display the starburst (heat map) representation of the SOM.
        """
        map_starburst(self.som_map_)

    def significance(self, graphics=False, feature_labels=True):
        """
        Compute and optionally plot the significance of each feature.
        
        Parameters
        ----------
        graphics : bool, default=False
            If True, produces a plot.
        feature_labels : bool, default=True
            If True, uses feature names in the plot.
        
        Returns
        -------
        significance : array-like or None
            A series of feature significance values (if graphics=False).
        """
        return map_significance(self.som_map_, graphics=graphics, feature_labels=feature_labels)

    def marginal(self, marginal):
        """
        Display a density plot of a chosen dimension overlayed with neuron density.
        
        Parameters
        ----------
        marginal : int or str
            The index or name of the data frame column.
        """
        return map_marginal(self.som_map_, marginal)
    
if __name__ == "__main__":
    import pandas as pd
    from sklearn import datasets

    iris = datasets.load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.DataFrame(iris.target_names[iris.target],columns=['species'])

    # Create and fit the SOM model
    som = SOM(xdim=20, ydim=15, train=100000, seed=42).fit(X, y)

    # View a summary of the SOM
    som.summary()

    # Map new data to clusters
    print(som.predict(X).head())

    # Get the grid coordinates for each sample
    positions = som.transform(X)

    # Display the starburst (heat map) representation
    som.starburst()

    # Optionally, display feature significance or marginal plots
    print(som.significance())
    som.marginal(2)



