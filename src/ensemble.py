# ensemble.py
#
# Base Ensemble Models in Machine Learning
# From MML Library by Nathmath

import numpy as np
try:
    import torch
except ImportError:
    torch = None

from copy import deepcopy

import matplotlib.pyplot as plt

from .objtyp import Object
from .tensor import Tensor
from .matrix import Matrix

from .baseml import MLBase, Regression, Classification


# Base Class for Ensemble Models
class Ensemble(Regression, Classification):
    
    __attr__ = "MML.Ensemble"
    
    def __init__(self, *, feature_names: Matrix | Tensor | None = None):
        """
        Initializes a Base ensemble model.

        Args:
            feature_names (Matrix | Tensor | None, optional): A Matrix or Tensor object containing strings representing the names of the features used by the model. 
                Defaults to None.  If provided, it should be a one-dimensional structure with a length equal to the number of columns in the data used for training.

        Attributes:
            feature_names (Matrix | Tensor | None): The feature names associated with the model, if provided during initialization.  Inherited from Ensemble.

        Returns:
            None
        """
        
        super().__init__()
        
        # Feature Names should be a 1 dimension Matrix or Tensor object of strings
        # It should be equal to the number of columns of data.
        self.feature_names = feature_names

    def fit(self):
        raise NotImplementedError("Fit is NOT implemented in the base class.")
        
    def predict(self):
        raise NotImplementedError("Predict is NOT implemented in the base class.")    
        
    def _is_fitted(self) -> bool:
        """
        Check if the Random Forest has been fitted or not.

        Returns
            -------
            bool, if the model is fitted or not.

        """
        # Base Class Attribute Test
        try:
            _ = self._estimators
        except AttributeError as e:
            raise RuntimeError(f"Invalid Inheritence! An Ensemble child class does NOT have `self._estimators` attribute, caused an error {e}.")
        
        return True if len(self._estimators) > 0 else False
        
    def plot_tree_id(self, tree_id: int, figsize = (14, 8), **kwargs):
        """
        Plot an image representing the structure of the `tree_id`-th decision tree within the forest.
        For leaf nodes, the label shows the prediction.
        It calls the tree's method to do the plot.
        
        Args:
            tree_id: int, the index of the tree to be plotted
            figsize: tuple, the size of the plot
        """
        # Base Class Attribute Test
        try:
            _ = self._estimators
        except AttributeError as e:
            raise RuntimeError(f"Invalid Inheritence! An Ensemble child class does NOT have `self._estimators` attribute, caused an error {e}.")
        
        
        # Check if the model has (at least partially) fitted or not
        if self._is_fitted() == False:
            raise RuntimeError("No weak learner is fitted. Call .fit() before plotting the tree structure.")
            
        if self._estimators.get(tree_id, None) is None:
            raise RuntimeError(f"Tree {tree_id} has NOT been trained. Please train or continue to train before plotting the tree structure.")
        self._estimators[tree_id].plot_tree(figsize = figsize)

    def plot_feature_importances_tree_id(self, tree_id: int, max_features: int = None, figsize=(8, 5), **kwargs) -> None:
        """
        Plot an image of the feature importance of ONE TREE as a bar chart.
        Each bar corresponds to the total loss reduction contributed by a feature.
        Values are extracted from self.feature_importance, converted to scalars using to_list()
        if necessary, and rounded to 4 digits.
        The bars are sorted in descending order by importance, similar to xgboost's plot.
        It calls the tree's method to do the plot.
        
        Args:
            tree_id: int, the index of the tree to be plotted
            max_features: int, the maximum number of features to plot (in importance descending)
            figsize: tuple, the size of the plot

        Returns
            -------
            None.

        """
        # Base Class Attribute Test
        try:
            _ = self._estimators
        except AttributeError as e:
            raise RuntimeError(f"Invalid Inheritence! An Ensemble child class does NOT have `self._estimators` attribute, caused an error {e}.")
        
        # Check if the model has (at least partially) fitted or not
        if self._is_fitted() == False:
            raise RuntimeError("No weak learner is fitted. Call .fit() before plotting the feature importance.")
            
        if self._estimators.get(tree_id, None) is None:
            raise RuntimeError(f"Tree {tree_id} has NOT been trained. Please train or continue to train before plotting the tree structure.")
        self._estimators[tree_id].plot_feature_importance(max_features = max_features, figsize = figsize)

    def plot_feature_importances_average(self, max_features: int = None, figsize=(8, 5), normalize: bool = False, **kwargs) -> None:
        """
        Plot an image of the feature importance of the averaged tree as a bar chart.
        Each bar corresponds to the total loss reduction contributed by a feature.
        Values are extracted from self.feature_importance, converted to scalars using to_list()
        if necessary, and rounded to 4 digits.
        The bars are sorted in descending order by importance, similar to xgboost's plot.
        
        Args:
            max_features: int, the maximum number of features to plot (in importance descending)
            figsize: tuple, the size of the plot

        Returns
            -------
            list of tuple (feature_idx, feature_importance)

        """
        # Base Class Attribute Test
        try:
            _ = self._estimators
        except AttributeError as e:
            raise RuntimeError(f"Invalid Inheritence! An Ensemble child class does NOT have `self._estimators` attribute, caused an error {e}.")
        try:
            _ = self.n_estimators_used
        except AttributeError as e:
            raise RuntimeError(f"Invalid Inheritence! An Ensemble child class does NOT have `self.n_estimators_used` attribute, caused an error {e}.")

        # Check if the model has (at least partially) fitted or not
        if self._is_fitted() == False:
            raise RuntimeError("No weak learner is fitted. Call .fit() before plotting the feature importance.")
            
        # Extract feature_idx-name map from any tree
        tree_ids = list(self._estimators.keys())
        feature_index_map = deepcopy(self._estimators[tree_ids[0]].feature_index_map)

        # Extract the sum of feature importance of every trained tree
        feature_importance_names = feature_index_map.values()
        n_features = len(feature_importance_names)
        feature_importance_values = np.repeat(0.0, n_features)
        for i in range(len(tree_ids)):
            right_feature_importance = self._estimators[tree_ids[i]].feature_importance
            for j in range(n_features):
                right_val = right_feature_importance.get(j, 0.0)
                if isinstance(right_val, Object):
                    right_val = right_val.to_list()
                feature_importance_values[j] = feature_importance_values[j] + right_val / self.n_estimators_used
        
        # Extract feature indexes and corresponding importance values as scalars.
        feat_imp_list = []
        for feature_name, val in zip(feature_importance_names, feature_importance_values):
            feat_imp_list.append((feature_name, val))
        
        # Sort the features by importance descending (largest first).
        feat_imp_list.sort(key=lambda x: x[1], reverse=True)
        
        # Truncate if employed
        if max_features is not None:
            feat_imp_list = feat_imp_list[0:max_features]
        
        # Unzip the sorted feature names and importance values.
        features, importances = zip(*feat_imp_list)
        
        # If normalize, then normalize the max to 1
        if normalize == True:
            importances = np.array(importances)
            importances = importances / min(importances)
            importances = importances.tolist()
        
        # Create a bar chart with the sorted feature importances.
        fig, ax = plt.subplots(figsize=figsize)
        ax.bar(features, importances)
        ax.set_xlabel("Features")
        ax.set_ylabel("Importance")
        ax.set_title("Average Feature Importance (Gain)")
        
        # Add text labels on top of the bars.
        for idx, v in enumerate(importances):
            ax.text(idx, v, str(round(v, 4)), ha='center', va='bottom')
        plt.show()
        
        return feat_imp_list

    def __repr__(self):
        return "Ensemble(Abstract Class)."


# Base Class for Bagging Models
class Bagging(Ensemble):
    
    __attr__ = "MML.Bagging"
    
    def __init__(self, *, feature_names: Matrix | Tensor | None = None):
        
        """
        Initializes a Bagging ensemble model.
        
        This class represents a bagging algorithm, which creates multiple instances of the 
        same base learner (e.g., decision tree) on different subsets of the training data and 
        aggregates their predictions to improve overall accuracy and reduce variance. It inherits from the 
        'Ensemble' base class and provides an optional way to specify feature names for improved interpretability.
        
        Args:
            feature_names (Matrix | Tensor | None, optional): A Matrix or Tensor object containing strings representing the names of the features used by the model. 
                Defaults to None.  If provided, it should be a one-dimensional structure with a length equal to the number of columns in the data used for training.
        
        Attributes:
            feature_names (Matrix | Tensor | None): The feature names associated with the model, if provided during initialization. Inherited from Ensemble.
        
        Returns:
            None
        """

        
        # Feature Names should be a 1 dimension Matrix or Tensor object of strings
        # It should be equal to the number of columns of data.
        super().__init__(feature_names = feature_names)
        # Assigned in the base class - Ensemble
        
    def _sample_bootstrapping(self, X: Matrix | Tensor, y: Matrix | Tensor, M: int, k: int | float | None, 
                              replace: bool = True, shuffle: bool = True,
                            *, random_state: int | None = None, container: type = list) -> list | tuple | dict:
        """
        Generate M bootstrapped datasets containing k entries from the original data.
        k must be smaller than or equal to the 1st dim of data, or error will be raised.
        
        Parameters:
            -----------
            X: Matrix | Tensor, the input Feature Matrix or Tensor to be bootstrapped.
            y: Matrix | Tensor, the input Target Matrix or Tensor to be bootstrapped.
            M: int, Number of bootstrapped datasets to generate, the length of the container.
            k: int | float | None, Number of entries for each subset, or None for k = X.shape[0].
            replace: bool, When choosing samples, if using sampling WITH REPLACEMET. Default True.
            shuffle: bool, If to shuffle the indices of rows, or make them in an increasing manner.
            Optional:
                random_state: int | None, the random seed set to numpy backend to do the sampling.
                container: type, the container type to contain the output data, can be list, tuple, or dict.
            
        Returns:
            -----------
            Container of tuple of subset (X, y, row_indices), like:
                [
                    (X_0, y_0, row_indices),
                    (X_1, y_1, row_indices),
                    ...
                ]
                where X_i, y_i are Matrix | Tensor subset of data,
                and  row_indices is also a Matrix | Tensor that is flatten.
        """
        
        # Type Check (must be an Object type).
        if isinstance(X, Object) == False or isinstance(y, Object) == False:
            raise ValueError("Input dataset must be either Matrix and Tensor. Use Matrix(data) or Tensor(data) to convert.")
        if type(X) != type(y):
            raise ValueError("Input feature `X` and target `y` must have the same type, either Matrix or Tensor.")
        
        # Integer validity Check.
        if M < 0 or k < 0 or k > X.shape[0]:
            raise ValueError("Input integer set (M, k) are not valid. Make sure you understand them!")
        
        # Dimension Check
        if len(X.shape) != 2:
            raise ValueError("Input feature `X` must be a tabular data with two dimensions.")
        if len(y.shape) == 1:
            raise ValueError("Input target `y` must also be a 2d data. If only one label or value, use data.reshape([-1, 1])")
        if X.shape[0] != y.shape[0]:
            raise ValueError("Input feature `X` must have the same size of shape[0] with the input target `y`.")
        
        # Container Type Check.
        if container not in (list, tuple, dict):
            raise ValueError("The container type must be one of `list`, `tuple`, `dict`.")
        
        # Number of entries reset.
        n_samples = X.shape[0]
        n_features = X.shape[1]
        if k is None:
            k = n_samples
        if isinstance(k, float):
            k = int(round(n_samples * k))
            
        # Random State
        if random_state is not None:
            np.random.seed(random_state)
        
        # The created, bootstrapped datasets.
        bootstrapped_datasets = container() if container in (list, dict) else []
        
        for i in range(M):
            indices = np.random.choice(n_samples, size=k, replace=replace)
            if shuffle == False:
                indices.sort()
            X_i = X[indices]
            y_i = y[indices]
            if container in (list, tuple):
                bootstrapped_datasets.append((X_i, y_i, type(X)(indices, backend=X._backend, dtype=int, device=X.device)))
            else:
                bootstrapped_datasets[i] = (X_i, y_i, type(X)(indices, backend=X._backend, dtype=int, device=X.device))
            np.random.seed(np.random.randint(0, int(2**31 - 1)))
                
        return bootstrapped_datasets if container in (list, dict) else tuple(bootstrapped_datasets)
    
    def _feature_random_select(self, M: int, N: int, q: int | float, replace: bool = False, *,
                               random_state: int | None = None, container: type = list) -> list | tuple | dict:
        """
        Generate M dataset feature indices by randomly selecting q features from N features
        
        Parameters:
        M: int, Number of datasets to generate.
        N: int, Number of features contained in the dataset.
        q: int, Number of features to select in each dataset, or
           float, proportion of features to select in each dataset.
        replace: bool, When choosing features, if using sampling WITH REPLACEMET. Default False.
            Optional:
                random_state: int | None, the random seed set to numpy backend to do the sampling.
                container: type, the exterior container type to contain the output data, can be list, tuple, or dict.
            
        
        Returns:
            -----------
            Container of np.array of feature indices, like:
                [
                    np.array([0,1,2,3,10]), # Must be 1 dimensional and increasing manner
                    np.array([1,2,4,8,9])   # Must be 1 dimensional and increasing manner
                ]
                where the internal data type is a np.array of selected feature indices.
                # Note, it is a np.array NOT Matrix or Tensor either
                
            Note: why using feature indices?
                  answer: to be compatible to most of MML's ensemble-abled models, like CART or LinearRegression.
        """
        
        # Integer validity Check.
        if M < 0 or N < 0 or q < 0 or q > N:
            raise ValueError("Input integer set (M, N, q) are not valid. Make sure you understand them!")
        
        # Container Type Check.
        if container not in (list, tuple, dict):
            raise ValueError("The container type must be one of `list`, `tuple`, `dict`.")
            
        # Number of entries reset.
        if q is None:
            q = N
            
        # If q is float, convert to N
        if isinstance(q, float):
            q = min(int(round(N * q)), N)
            
        # Random State.
        if random_state is not None:
            np.random.seed(random_state)
            
        # The selected feature-index container.
        feature_indices = container() if container in (list, dict) else []
        
        for i in range(M):
            indices = np.random.choice(N, size=q, replace=replace)
            indices.sort()
            if container in (list, tuple):
                feature_indices.append(indices)
            else:
                feature_indices[i] = (indices)
            np.random.seed(np.random.randint(0, int(2**31 - 1)))
        
        return feature_indices if container in (list, dict) else tuple(feature_indices)
    
    def __repr__(self):
        return "Bagging(Abstract Class)."
    
    
# Base Class for Boosting Models
class Boosting(Ensemble):
    
    __attr__ = "MML.Boosting"
    
    def __init__(self, *, feature_names: Matrix | Tensor | None = None):
        """
        Initializes a Boosting ensemble model.

        This class represents a boosting algorithm, which combines multiple weak learners 
        (typically decision trees) to create a strong predictive model.  It inherits from the 
        'Ensemble' base class and provides an optional way to specify feature names for improved interpretability.

        Args:
            feature_names (Matrix | Tensor | None, optional): A Matrix or Tensor object containing strings representing the names of the features used by the model. 
                Defaults to None.  If provided, it should be a one-dimensional structure with a length equal to the number of columns in the data used for training.

        Attributes:
            feature_names (Matrix | Tensor | None): The feature names associated with the model, if provided during initialization.  Inherited from Ensemble.

        Returns:
            None
        """
        
        # Feature Names should be a 1 dimension Matrix or Tensor object of strings
        # It should be equal to the number of columns of data.
        super().__init__(feature_names = feature_names)
        # Assigned in the base class - Ensemble
    
    def __repr__(self):
        return "Boosting(Abstract Class)."
    
    
# Some Basic Tests
if __name__ == "__main__":
    
    bag = Bagging()
    typeclass = Matrix
    
    m10 = typeclass([
        [10, 12, 15],
        [7.1, 4, 20],
        [25, 14, 19],
        [42, 2821, 0],
        [17, 4, 1216],
        [20, 13, 727]], backend = "torch")
    
    n10 = typeclass([
        [1.2],
        [2.4],
        [7.1],
        [0.9],
        [0.2],
        [2.2]], backend = "torch")
    
    # Bootstrapping
    samples = bag._sample_bootstrapping(m10, n10, 3, k = 4, random_state=42)
    
    # Feature selecting
    features = bag._feature_random_select(4, 3, q = 2, random_state=None)
    
    # See what is going on (in practice, DO NOT do it)
    print(
        samples[0][0][:,features[0]]
        )
