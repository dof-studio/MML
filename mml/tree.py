# tree.py
#
# Tree models in Machine Learning
# From MML Library by Nathmath

import numpy as np
try:
    import torch
except ImportError:
    torch = None

from copy import deepcopy

import lzma
import random
from collections import Counter
import matplotlib.pyplot as plt

from .dump import save, load

from .objtyp import Object
from .tensor import Tensor
from .matrix import Matrix

from .metrics import RegressionMetrics
from .metrics import BinaryClassificationMetrics
from .metrics import MultiClassificationMetrics
from .baseml import MLBase, Regression, Classification


# Base Class for Tree Models
class BaseTree(Regression, Classification):
    
    __attr__ = "MML.BaseTree"
    
    def __init__(self, *, feature_names: Matrix | Tensor | None = None, **kwargs):
        
        super().__init__()
        
        # Feature Names should be a 1 dimension Matrix or Tensor object of strings
        # It should be equal to the number of columns of data.
        self.feature_names = feature_names
        
        # Dictionary to keep track of total loss reduction per feature (for feature importance)
        self.feature_importance = {} # Key: feature index
                                     # Value: Total Information Gain
        
        # Build a mapping from feature index to feature name.
        self.feature_index_map = {}  # Key: feature index
                                     # Value: feature name
        
        # All features used in fit.
        self.use_features_idx = None
        
    def _gini_impurity(self, y: Matrix | Tensor, one_hot: bool | None = None, floattype: type = float):
        """
        Compute the Gini impurity for a set of class labels.
        
        This function accepts either:
          1. A 1D Matrix | Tensor of integer labels (e.g., [0, 1, 0, 1, 2, 1, 3, ...])
          2. A 2D Matrix | Tensor in one-hot encoded format (e.g., [[1, 0], [0, 1], ...])
             or a count matrix in which each row represents a sample with counts 
             (typically one-hot for classification problems).
        
        Parameters:
            --------
            y : Matrix | Tensor
                The labels which can be of shape (n_samples,) or (n_samples, n_classes).
            one_hot : bool, optional
                Indicates whether the input is one-hot encoded. If None, the function will
                attempt to automatically detect one-hot encoding when y is 2D.
        
        Returns:
            --------
            Matrix | Tensor
                The computed Gini impurity, scalar, stored in the container
        """
        if len(y.shape) == 1:
            # Input is a 1D array of labels
            counts = y.bincount()
            
        elif len(y.shape) == 2:
            # If one_hot is explicitly provided or can be inferred.
            if one_hot is None:
                if len(y.unique().flatten()) > 2:
                    raise ValueError("You have more than 2 unique values. Make sure it is one-hot encoded.")
                one_hot = True
            
            if one_hot == False:
                counts = y.flatten().bincount()
                    
            else:
                # One-hot encoded data: count occurrences per class by summing columns
                counts = y.sum(axis=0)
        else:
            raise ValueError("Input array 'y' must be either a 1D or a 2D data.")
            
        total = counts.sum()
        # Avoid division by zero for empty input
        if total == 0:
            return type(y)(0.0, backend = y._backend, dtype = floattype, device=y.device)
        
        # Compute impurity
        probabilities = counts / total
        impurity = 1.0 - (probabilities ** 2).sum()
        return impurity
        
    def _create_feature_index_map(self, n_features: int, *, regime: str = "Feature_{}"):
        """
        Creates a mapping from column indices to feature names or default names.
        
        Args:
            regime (str): A format string for generating default feature names if no names are provided.
                          Default is 'Feature_{index}'.
        
        Returns:
            self
        
        """
        # If feature names were provided, create a map from column index to feature name.
        # Otherwise, generate default names (e.g., "Feature0", "Feature1", ...)
        if self.feature_names is not None:
            # If it is an Object
            if isinstance(self.feature_names, Object):
                feature_names = self.feature_names.flatten().to_list()
            # Or iterative containers
            else:
                feature_names = list(self.feature_names)
                
            # Map it from 0
            self.feature_index_map = {}
            for i in range(n_features):
                self.feature_index_map[i] = feature_names[i]
        else:
            # Map it from 0 using the default regime
            self.feature_index_map = {}
            for i in range(n_features):
                self.feature_index_map[i] = regime.format(i)
        return self
    
    def fit(self, **kwargs):
        raise NotImplementedError("Fit is NOT implemented in the base class.")
        
    def predict(self, **kwargs):
        raise NotImplementedError("Predict is NOT implemented in the base class.")    
    
    def plot_feature_importance(self, max_features: int = None, figsize=(8, 5)):
        """
        Plot an image of the feature importance as a bar chart.
        Each bar corresponds to the total loss reduction contributed by a feature.
        Values are extracted from self.feature_importance, converted to scalars using to_list()
        if necessary, and rounded to 4 digits.
        The bars are sorted in descending order by importance, similar to xgboost's plot.
        
        Args:
            max_features: int, the maximum number of features to plot (in importance descending)
            figsize: tuple, the size of the plot
        """
    
        if not self.feature_importance:
            print("No feature importance data available. Please train the tree first.")
            return
    
        # Extract feature indexes and corresponding importance values as scalars.
        feat_imp_list = []
        for feat_idx, imp in self.feature_importance.items():
            # Get feature name from feature_index_map.
            feature_name = self.feature_index_map.get(feat_idx, f"Feature_{feat_idx}")
            if isinstance(imp, Object):
                val = imp.to_list()  # extract raw value(s)
            else:
                val = imp
            if isinstance(val, list):
                val = val[0] if len(val) == 1 else val
            feat_imp_list.append((feature_name, val))
        
        # Sort the features by importance descending (largest first).
        feat_imp_list.sort(key=lambda x: x[1], reverse=True)
        
        # Truncate if employed
        if max_features is not None:
            feat_imp_list = feat_imp_list[0:max_features]
        
        # Unzip the sorted feature names and importance values.
        features, importances = zip(*feat_imp_list)
        
        # Create a bar chart with the sorted feature importances.
        fig, ax = plt.subplots(figsize=figsize)
        ax.bar(features, importances)
        ax.set_xlabel("Features")
        ax.set_ylabel("Importance")
        ax.set_title("Feature Importance (Gain)")
        
        # Add text labels on top of the bars.
        for idx, v in enumerate(importances):
            ax.text(idx, v, str(round(v, 4)), ha='center', va='bottom')
        plt.show()
    
    def __repr__(self):
        return "BaseTree(Abstract Class)."
    

# Classfication and Regression Tree (CART)
class CART(BaseTree):
    
    __attr__ = "MML.CART"
    
    def __init__(self, task: str = 'regression', tree_id: int = None, *,
                 loss: str = None,
                 max_depth: int = 12,
                 min_samples_split: float | int = 0.001,
                 grid_accelerator: int | None = 10,
                 grid_use_percentile: bool = False,
                 grid_point_variant: float | None = None,
                 prune: bool = True,
                 prune_alpha: float = 0.001,
                 random_state: int | None = None,
                 floattype: type = float,
                 feature_names: Matrix | Tensor | list | tuple | None = None,
                 **kwargs):
        """
        Initialize a CART tree that can be used for regression or classification.

        Parameters:
            task: str, 'regression' or 'classification'.
            tree_id: int, the identifiation number for this tree. If None, then 0.
            loss: str, the name of the loss function applied.
                  For regression, choose 'mse', 'rmse', 'wmse', 'wrmse', or any RegressionMetrics;
                  For classification, 'gini' or 'logloss' may be used.
                  If not provided, defaults to 'mse' for regression and 'logloss' for classification.
            max_depth: int, Maximum depth of the tree.
            min_samples_split: float | int, when float, it is the minimum fraction required to split a node. Recommend [0.00, 0.02]
                               or when it is int, minimum number of samples.
            grid_accelerator: int | None, if non-None, use grid based splitor instead of exhaustive test. May lose accuracy. Minimum 5.
            grid_use_percentile: bool, if True, grid search will be based on percentiles, otherwise equally splited. Setting False is faster.
            grid_point_variant: float, if given, then points in the grid will be randomly move within this scale, Recommend [None, 0, 0.5]
            prune: bool, whether to prune the tree or not
            prune_alpha: float, L1 regularization on the number of leaves in the nested structure. Recommended values [0.00, 0.005]
            random_state: int, random seed for grid searching, controlling the fitting process, including the sequence of selecting features,
                          and the random movements applied on grid points, which is further controlled by grid_point_variant.
            floattype: type, The target float type that is going to be used in training
            feature_names: Matrix | Tensor, A 1D Matrix or Tensor of string names for features.
            **kwargs: other key word arguments, reserved for compatibility use.
        """
        super().__init__(feature_names=feature_names)
        
        # Task and Loss Check
        self.tree_id = tree_id if tree_id is not None else 0
        self.task = task.lower()
        if self.task == 'regression':
            self.loss = loss if loss is not None else 'mse'
        elif self.task == 'classification':
            self.loss = loss if loss is not None else 'logloss'
        else:
            raise ValueError("Unsupported task. Choose 'regression' or 'classification'.")
            
        # Invalid grid accelerator
        if grid_accelerator is not None and grid_accelerator < 10:
            raise ValueError("Argument grid_accelerator should be an integer >= 10 to make the tree work properly. Or leave None to do an exahustive search.")
            
        # Regularization Related
        self.prune = prune
        self.prune_alpha = prune_alpha
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        
        # Split Related 
        self.random_state = random_state
        self.grid_accelerator = grid_accelerator
        self.grid_percentile = grid_use_percentile
        self.grid_point_variant = grid_point_variant
        
        # Feature and Target info
        self.n_classes = None
        self.total_samples = None
        
        # Original Dataset (assigned in fit())
        self.original_X = None
        self.original_y = None
        self.original_weights = None
        self.floattype = floattype
        
        # KWargs Accepted
        self.kwargs = kwargs
        
        # Tree structure - a Dict of dict ...
        self.root = None # Tree object
        self.unpruned_root = None
                         # It is the core thing to see whether the tree is trained or not
                         
        # Record the total number of leaves
        self.total_num_leaves = 0
                         # Note, the number of leaves are made sure to be unique but NOT consecutive
                         # This DOES NOT show the number of leaves but also stands for a number counter for leaves

        # Record the type (Matrix or Tensor) of the input data for later reconstruction.
        self.typeclass = None
        
        # Record if it is one hot
        self.one_hot = None
        
    def fit(self, X: Matrix | Tensor, y: Matrix | Tensor, 
            one_hot: bool = True, 
            use_features_idx: tuple | list | Matrix | Tensor | None = None, 
            weights: Matrix | Tensor | None = None,
            **kwargs):
        """
        Fit the CART tree to the data.
        
        Parameters:
            X: Matrix | Tensor, the feature matrix (each row is a sample).
            y: Matrix | Tensor, the target values (for regression, numerical; for classification, one-hot or multi-label).
            one_hot: bool, whether target y is one hot encoded
            use_features_idx, Matrix | Tensor | tuple | list of indices or None (all features)
            weights: Matrix | Tensor | None, if non-None, it will pass to Metrics/Loss and should be used with Metrics like "wmse" or "wrmse".
        
        Returns:
            self
        """
        # Type Check (must be an Object type).
        if isinstance(X, Object) == False or isinstance(y, Object) == False:
            raise ValueError("Input dataset must be either Matrix and Tensor. Use Matrix(data) or Tensor(data) to convert.")
        if type(X) != type(y):
            raise ValueError("Input feature `X` and target `y` must have the same type, either Matrix or Tensor.")
        
        # Dimension Check
        if len(X.shape) != 2:
            raise ValueError("Input feature `X` must be a tabular data with two dimensions.")
        if len(y.shape) == 1:
            raise ValueError("Input target `y` must also be a 2d data. If only one label or value, use data.reshape([-1, 1])")
                    
        # Copy Training data
        self.original_X = X.to(backend=X._backend, dtype = self.floattype, device=X.device)
        self.original_y = y.to(backend=y._backend, dtype = self.floattype, device=y.device)
        self.original_weights = weights if weights is None else weights.to(backend=weights._backend, dtype = self.floattype, device=weights.device)
        
        # Check the shape of weights if existance
        if self.original_weights is not None:
            if isinstance(self.original_weights, Object) == False:
                raise ValueError("Input weights must be either Matrix and Tensor. Use Matrix(data) or Tensor(data) to convert.")
            if type(self.original_weights) != type(y):
                raise ValueError("Input `weights` and target `y` must have the same type, either Matrix or Tensor.")
        
        # Record the one_hot case
        self.one_hot = one_hot
        
        # Record the input type.
        self.typeclass = type(X)
        
        # Get the number of classes
        self.n_classes = len(self._to_labels(y).flatten().unique())
        
        # Store total number of training samples to check against min_samples_split.
        self.total_samples = X.shape[0]
        
        # If feature names were provided, create a map for feature names
        self._create_feature_index_map(X.shape[1])
        
        # Set features used (will be converted to a list).
        self.use_features_idx = use_features_idx
        if self.use_features_idx is None:
            # Use all
            self.use_features_idx = list(range(X.shape[1]))
        else:
            if isinstance(self.use_features_idx, Object):
                self.use_features_idx = self.use_features_idx.flatten().to_list()
            elif isinstance(self.use_features_idx, tuple):
                self.use_features_idx = list(self.use_features_idx)
            elif isinstance(self.use_features_idx, np.ndarray):
                self.use_features_idx = self.use_features_idx.tolist()
                
        #######################################################################
        # 
        # CART Special Fitting logic
                
        # Build the tree recursively starting from the root.
        self.unpruned_root = self._build_tree(self.original_X, self.original_y, self.original_weights, depth = 0)
        # Note, this _build_tree uses _find_best_split, which is desperately slow.
        # It may subject to future optimization using batched Tensor algebra.
        # By Nathmath Huang
        
        # If needs prune, then prune the tree and transfer the node
        if self.prune == True:
            self.root = deepcopy(self.unpruned_root)
            self.root, _, _, _ = self._prune_tree(self.root, alpha = self.prune_alpha)
            # For simplicity, we only provide alpha and let program to determine X,y,weights
        else:
            self.root = deepcopy(self.unpruned_root)
            
        # Create feature importance mapping
        self._create_feature_importance(None, recursive=False)
            
        return self
    
    def _create_feature_importance(self, node = None, *, recursive: bool = False, use_prune: bool = True, **kwargs):
        """
        Creates a feature importance dict.
        
        Args:
            recursive: bool, if it is called by a recursive function or called externally.
            use_prune: bool, if using pruned tree or not.
        
        Returns:
            None
        
        """
        # Not recursive, first call, clear the self.feature_importance
        if recursive == False:
            self.feature_importance = {}
        
        # Not recursive, initially get the root
        if recursive == False and node is None:
            node = self.root if use_prune else self.unpruned_root
            
        # If the leaf node, return it
        if node.get("prediction", None) is not None:
            return
        
        # Feature improvement for feature importance.
        feat = node["feature_index"]
        self.feature_importance[feat] = self.feature_importance.get(feat, 0.0) + node["improvement"]
        
        # Recursively call left and right
        self._create_feature_importance(node["left"], recursive=True)
        self._create_feature_importance(node["right"], recursive=True)
        
        return
    
    def _build_tree(self, X: Matrix | Tensor, y: Matrix | Tensor, weights: Matrix | Tensor| None = None, *, depth: int, **kwargs):
        """
        Recursively build the tree structure.
        
        Args:
            X: Matrix | Tensor, the splited data of features at this time.
            y: Matrix | Tensor, the splited target at this time.
            weights: Matrix | Tensor | None, if non-None, it will pass to Metrics/Loss and should be used with Metrics like "wmse" or "wrmse",
                                                          and will be used in calculating the weighted leaf.
        
        Returns:
            dict, A dictionary representing the node. Internal nodes have keys:
                'feature_index' and 'threshold' and pointers 'left' and 'right'.
            Leaf nodes have a key 'prediction' storing the constant prediction.
        """
        # It is in the internal loop, we don't do type check here
        # 
        
        n_samples = X.shape[0]

        # Recursion Stops #####################################################
        # 
        # Determine if this node should be a leaf:
        # (1) Maximum depth reached, 
        # (2) too few samples to split, or 
        # (3) node is pure (no need to do further classification or regression)
        #  etc...
        if depth >= self.max_depth:
            return self._create_leaf(y, weights = weights)
            # Stopping critera 1)
        
        if n_samples < (self.min_samples_split if self.min_samples_split >= 1 else self.total_samples * self.min_samples_split) or n_samples == 1:
            return self._create_leaf(y, weights = weights)
            # Stopping critera 2)
        
        if self._is_pure(y):
            return self._create_leaf(y, weights = weights)
            # Stopping critera 3)
        
        # Find the best split given the data at this node.
        if self.grid_accelerator is None:
            best_split = self._find_best_split(X, y, weights = weights)
        else:
            best_split = self._find_best_split_gridsearch(X, y, weights = weights, 
                                               grid_k=self.grid_accelerator, percentile=self.grid_percentile)
        
        # If NO improvement or subset count == 0, still stop here.
        if best_split["improvement"].data <= 0 or best_split["left_count"] == 0 or best_split["right_count"] == 0:
            return self._create_leaf(y, weights = weights)
            # Stopping critera 4)
        
        # Partition the data into left and right children using the best split.
        left_mask = self._get_mask(X, best_split["feature_index"], best_split["threshold"], left=True)
        right_mask = self._get_mask(X, best_split["feature_index"], best_split["threshold"], left=False)
        
        # Use the masks to obtain subsets of the data.
        X_left = X[left_mask.to_numpy_array()]
        y_left = y[left_mask.to_numpy_array()]
        X_right = X[right_mask.to_numpy_array()]
        y_right = y[right_mask.to_numpy_array()]
        
        # Use the masks to obtain subsets of the weights if having.
        if weights is not None:
            left_weights = weights[left_mask.to_numpy_array()]
            right_weights = weights[right_mask.to_numpy_array()]
        else:
            left_weights = None
            right_weights = None
            
        #######################################################################
        # Recursive!
        #
        # Recursively build the left and right subtrees.
        left_node = self._build_tree(X_left, y_left, weights = left_weights, depth = depth + 1)
        right_node = self._build_tree(X_right, y_right, weights = right_weights, depth = depth + 1)
        
        # Return the internal node as a dictionary.
        return {
            "feature_index": best_split["feature_index"],
            "threshold": best_split["threshold"],
            "improvement": best_split["improvement"],
            "weighted_loss": best_split["weighted_loss"],
            "left": left_node,
            "right": right_node
        }
    
    def _create_leaf(self, y: Matrix | Tensor, *, weights: Matrix | Tensor | None = None, keepdims: bool = False, **kwargs):
        """
        Create a leaf node by computing the optimal constant prediction for the node.
        For regression, this is typically the mean; 
        For classification, it is the average distribution.
        
        Returns:
            dcit, dict of prediction
        """
        # For regression, we compute the arithmetic mean of y.
        # For classification, we compute the arithmetic mean (same) of the one-hot outputs.
        # We STRONGLY recommend users to convert to one-hot data.
        
        # If having weights, we compute weighted average with weights normalization
        if weights is None:
            prediction = y.mean(axis=0)
        else:
            norm_weights = weights / weights.sum(axis=0)
            prediction = (y * norm_weights).sum(axis=0)
            
        # Increase the total number of leaves (just a counter)
        leaf_no = int(self.total_num_leaves)
        self.total_num_leaves += 1
        
        return {
                 "prediction": prediction if keepdims == False else prediction.reshape([1, -1]).repeat(y.shape[0], axis=0),
                 "leaf_no": leaf_no
               }    
    
    def _is_pure(self, y: Matrix | Tensor, *, std_thres: float = 1e-10, **kwargs):
        """
        Check whether the given target values are pure.
        For regression, this may check that the std devation is nearly zero (< std_thres).
        For classification, it may check whether all labels are identical.
        
        Returns:
            bool, whether the node is pure (perfectly classfied or regresed) or not.
        """
        if self.task == "regression":
            # Compute the std dev.
            stdev = y.std(axis=0)
            counts = self.typeclass(stdev.data <= 1e-8, backend=y._backend, device=y.device)
            if isinstance(counts, Object):
                return counts.all() == True
            else:
                return counts == True
        
        else:
            # For classification, if y is one-dimensional (integer labels), check if all are the same.
            # If one-hot, compare the argmax.
            if self.one_hot == True:
                labels = y.argmax(axis=1).flatten()
            else:
                labels = y.flatten()
            first = labels[0]
            counts = labels == first
            if isinstance(counts, Object):
                return counts.all() == True
            else:
                return counts == True

    def _find_best_split(self, X: Matrix | Tensor, y: Matrix | Tensor, weights: Matrix | Tensor | None = None, **kwargs):
        """
        Iterate over all features and candidate threshold values to identify the best split.
        This is the UN-optimized version using a O(n^2) nested loop.
        
        Special Args:
            weights: Matrix | Tensor | None, if non-None, it will pass to Metrics/Loss and should be used with Metrics like "wmse" or "wrmse",
                                                          and will be used in calculating the weighted leaf.
        
        Returns:
            A dict like:
                {
                    "feature_index": index of feature used for split,
                    "threshold": candidate threshold,
                    "improvement": improvement in loss,
                    "weighted_loss": weighted loss that is the best,
                    "left_count": number of samples on the left side,
                    "right_count": number of samples on the right side.
                }
        """
        
        # Future Optimization:
        # > If note categorical data, you may use binary search instead of looping
        #   to find the optimal split within one feature.
        # > It will reduce O(n^2) to O(n * log(n))
        
        # Create placeholders.
        n_samples = X.shape[0]
        best_split = {
            "feature_index": None,
            "threshold": None,
            "improvement": Matrix(-float("inf"), backend=X._backend, dtype=float, device=X.device),
            "weighted_loss": Matrix(-float("inf"), backend=X._backend, dtype=float, device=X.device),
            "left_count": 0,
            "right_count": 0
        }
        
        # Compute parent prediction and its loss before splitting.
        parent_pred = self._create_leaf(y, weights = weights, keepdims=True)["prediction"]
        parent_loss = self._compute_loss(parent_pred, y, weights = weights, **kwargs)
        
        n_features = X.shape[1]
        
        # Test on a feature and a threshold
        def _test_feature_threshold(feature_i: int, threshold: float, parent_loss_: float, best_improvement: Matrix | Tensor) -> dict:
            """
            Test a split and see if it is a better split against a given benchmark.
            
            Returns
            -------
            dict
                if this is a better threshold and feature, return the information creating a split;
                otherwise, returns an empty dict.

            """
            # If threshold is on the egde of range, return empty.
            X_col = X[:,feature_i]
            if threshold >= X_col.max().data or threshold <= X_col.min().data:
                return {}
            
            # Obtain boolean masks for the split.
            mask_left = self._get_mask(X, feature_i, threshold, left=True)
            mask_right = self._get_mask(X, feature_i, threshold, left=False)
                
            # Get the counts; if either side is empty, continue.
            left_count = mask_left.sum()
            right_count = mask_right.sum()
            if left_count == 0 or right_count == 0:
                return {}
                
            # Extract left and right splits.
            y_left = y[mask_left.to_list()]
            y_right = y[mask_right.to_list()]
            
            # Use the masks to obtain subsets of the weights if having.
            if weights is not None:
                left_weights = weights[mask_left.to_numpy_array()]
                right_weights = weights[mask_right.to_numpy_array()]
            else:
                left_weights = None
                right_weights = None
                
            # Compute predictions for left and right groups.
            left_pred = self._create_leaf(y_left, weights=left_weights, keepdims=True)["prediction"]
            right_pred = self._create_leaf(y_right, weights=right_weights, keepdims=True)["prediction"]
                
            # Compute losses using the external loss functions.
            left_loss = self._compute_loss(left_pred, y_left, weights = left_weights)
            right_loss = self._compute_loss(right_pred, y_right, weights = right_weights)
                
            # Compute weighted loss.
            weighted_loss_ = (left_count / n_samples) * left_loss + (right_count / n_samples) * right_loss
                
            # Compute improvement (loss reduction).
            improvement_ = parent_loss_ - weighted_loss_
                
            # Check if the improvement is the best so far.
            if improvement_.data > best_improvement.data:
                
                return {
                    "feature_index": feature_i,
                    "threshold": threshold,
                    "improvement": improvement_,
                    "weighted_loss": weighted_loss_,
                    "left_count": left_count,
                    "right_count": right_count,
                }
            else:
                return {}

        # Iterate over all features
        for i in range(n_features):
            
            # If the feature is NOT used, continue
            if i not in self.use_features_idx:
                continue
            
            # Get the entire column i as a Matrix/Tensor.
            feature_column = X[:, i]
            unique_values = feature_column.flatten().unique().sort().to_list()
            
            # Skip if there is only one unique value.
            if len(unique_values) <= 1:
                continue
            
            # Consider candidate thresholds between adjacent unique values.
            for j in range(len(unique_values) - 1):
                
                # Use the midpoint as candidate threshold.
                threshold_candidate = (unique_values[j] + unique_values[j + 1]) / 2
               
                # Conduct a test on this 
                _test_result = _test_feature_threshold(i, threshold_candidate, parent_loss, best_improvement=best_split["improvement"])
                    
                if len(_test_result) > 0:
                    best_split = _test_result
                
        return best_split
    
    def _find_best_split_gridsearch(self, X: Matrix | Tensor, y: Matrix | Tensor, weights: Matrix | Tensor | None = None, 
                                    *, 
                                    grid_k: int = 10, 
                                    random_state:int | None = None,
                                    variant: float | None = None, 
                                    percentile: bool = False,
                                    **kwargs):
        """
        Iterate over all features and candidate threshold values to identify the best split.
        Uses a combination of grid search (with progressive interval refinement)
        to efficiently explore the candidate threshold space.
        
        Special Args:
            weights: Matrix | Tensor | None, if non-None, it will pass to Metrics/Loss and should be used with Metrics like "wmse" or "wrmse",
                                                          and will be used in calculating the weighted leaf.
        
        Control Args:
            grid_k: int, the number of grid points within a specific range, default is 50, recommend 20, 50, 100, 200.
            random_state: int, the random seed controlling the sequence of inspecting features and grid variants, if None, then randomly draw.
            variant: float, the size (in decimal) of random noise to add on the splited grid points, if None, then add nothing.
            percentile: bool, whether to use fix grid points or computed percentile values (as the base) to check the thresholds.
        
        Returns:
            A dict like:
                {
                    "feature_index": index of feature used for split,
                    "threshold": candidate threshold,
                    "improvement": improvement in loss,
                    "weighted_loss": weighted loss that is the best,
                    "left_count": number of samples on the left side,
                    "right_count": number of samples on the right side.
                }
        """        
        # Set the global seed if passed
        if random_state is not None:
            random.seed(random_state)
            np.random.seed(random_state)
            random_state += self.total_num_leaves
        
        # Create placeholders.
        n_samples = X.shape[0]
        best_split = {
            "feature_index": None,
            "threshold": None,
            "improvement": Matrix(-float("inf"), backend=X._backend, dtype=float, device=X.device),
            "weighted_loss": Matrix(-float("inf"), backend=X._backend, dtype=float, device=X.device),
            "left_count": 0,
            "right_count": 0
        }
        
        # Compute parent prediction and its loss before splitting.
        parent_pred = self._create_leaf(y, weights = weights, keepdims=True)["prediction"]
        parent_loss = self._compute_loss(parent_pred, y, weights = weights, **kwargs)
        
        n_features = X.shape[1]
        
        # Randomly choose the sequence of features (including non-used, but anyway, we will prevent using them)
        seq_features = list(range(n_features))
        random.shuffle(seq_features)
        
        #######################################################################
        # Test on a feature and a threshold
        def _test_feature_threshold(feature_i: int, threshold: float, parent_loss_: float, best_improvement: Matrix | Tensor) -> dict:
            """
            Test a split and see if it is a better split against a given benchmark.
            
            Returns
            -------
            dict
                if this is a better threshold and feature, return the information creating a split;
                otherwise, returns an empty dict.

            """
            # If threshold is on the egde of range, return empty.
            X_col = X[:,feature_i]
            if threshold >= X_col.max().data or threshold <= X_col.min().data:
                return {}
            
            # Obtain boolean masks for the split.
            mask_left = self._get_mask(X, feature_i, threshold, left=True)
            mask_right = self._get_mask(X, feature_i, threshold, left=False)
                
            # Get the counts; if either side is empty, continue.
            left_count = mask_left.sum()
            right_count = mask_right.sum()
            if left_count == 0 or right_count == 0:
                return {}
                
            # Extract left and right splits.
            y_left = y[mask_left.to_list()]
            y_right = y[mask_right.to_list()]
            
            # Use the masks to obtain subsets of the weights if having.
            if weights is not None:
                left_weights = weights[mask_left.to_numpy_array()]
                right_weights = weights[mask_right.to_numpy_array()]
            else:
                left_weights = None
                right_weights = None
                
            # Compute predictions for left and right groups.
            left_pred = self._create_leaf(y_left, weights = left_weights, keepdims = True)["prediction"]
            right_pred = self._create_leaf(y_right, weights = right_weights, keepdims = True)["prediction"]
                
            # Compute losses using the external loss functions.
            left_loss = self._compute_loss(left_pred, y_left, weights = left_weights)
            right_loss = self._compute_loss(right_pred, y_right, weights = right_weights)
                
            # Compute weighted loss.
            weighted_loss_ = (left_count / n_samples) * left_loss + (right_count / n_samples) * right_loss
                
            # Compute improvement (loss reduction).
            improvement_ = parent_loss_ - weighted_loss_
                
            # Check if the improvement is the best so far.
            if improvement_.data > best_improvement.data:
                
                return {
                    "feature_index": feature_i,
                    "threshold": threshold,
                    "improvement": improvement_,
                    "weighted_loss": weighted_loss_,
                    "left_count": left_count,
                    "right_count": right_count,
                }
            else:
                return {}

        #######################################################################
        # Grid search accelerator
        def _grid_search_threshold(feature_i: int, unique_values: list,
                                   X: Matrix | Tensor, y: Matrix | Tensor, weights: Matrix | Tensor | None,
                                   parent_loss_: float, best_improvement: Matrix | Tensor,
                                   grid_k: int = 10, tol_points: int | None = None, 
                                   random_state: int | None = None, variant:float | None = None, percentile: bool = True) -> dict:
            """
            Perform a multi-resolution grid search within the range spanned by unique_values.
            - First, sample grid_k thresholds (linearly across [min, max]).
            - Find the grid candidate with the best improvement.
            - If the candidate is not on the boundary, narrow the interval to the neighbors of this candidate.
            - When the number of unique values inside the refined interval is below tol_points,
              fall back to exhaustive search on those unique values.
              
            Returns the best candidate for this feature as a dict.
            """
            # Grid k minimum 5, but default 10, to make the searching algorithm works
            if grid_k is None:
                grid_k = 10
            elif grid_k < 5:
                grid_k = 5
            
            # If tol_points not initialized
            if tol_points is None:
                tol_points = int(grid_k * 1.5) if grid_k >= 25 else grid_k * 2
            
            low = unique_values[0]
            high = unique_values[-1]
            candidate = {}
            
            # If unique values is smaller than tol_points
            if len(unique_values) < tol_points:
                
                # Perform exhaustive search on these.
                for it, th in enumerate(unique_values):
                    if it == 0:
                        continue
                    thres = (unique_values[it-1] + unique_values[it]) / 2
                    res = _test_feature_threshold(feature_i, thres, parent_loss_, best_improvement)
                    if len(res) > 0:
                        candidate = res
                        best_improvement = res["improvement"]
                return candidate                    
            
            # Percentile helper
            def _percentile_within(data: np.ndarray, percentiles: list, lower_threshold: float, upper_threshold: float):
                """
                Computes specified percentile values for data within a given threshold range.
            
                Parameters:
                    data (np.ndarray): Input 1D data array.
                    percentiles (np.ndarray or list): Percentiles to compute, e.g., [25, 50, 75].
                    lower_threshold (float): Lower bound of the data to include.
                    upper_threshold (float): Upper bound of the data to include.
            
                Returns:
                    np.ndarray: Computed percentile values of the filtered data.
                """
                if not isinstance(data, np.ndarray):
                    data = np.array(data)
            
                filtered_data = data[(data >= lower_threshold) & (data <= upper_threshold)]
                
                if filtered_data.size == 0:
                    return np.array([]) # Empty, error happened
            
                return np.percentile(filtered_data, percentiles)
            
            # Initialize the random machine
            if random_state is not None:
                np.random.seed(random_state)
            
            # If variant is too large (> 0.5), then make as 0.5
            if variant is not None:
                variant = 0.5 if variant > 0.5 else variant
            
            # Percentiles we have
            percentiles = np.arange(0.0, 100 + 1e-10, 100.0 / grid_k)

            while True:
                # Create grid points between low and high.
                if percentile == False:
                    grid_points = [low + (high - low) * i / (grid_k - 1) for i in range(grid_k)]
                else:
                    grid_points = list(_percentile_within(unique_values, percentiles, low, high))
                    if len(grid_points) == 0:
                        break
                
                # If add random noise as variance
                if variant is not None:
                    def diff_stdev(x):
                        diff_values = np.diff(x)
                        result = np.insert(diff_values, 0, 0)
                        return result.std()
                    grid_points_len = len(grid_points)
                    grid_diff_stdev = diff_stdev(grid_points)
                    grid_points = (np.array(grid_points) + np.random.randn(grid_points_len) * variant * grid_diff_stdev).tolist()
                
                best_local = {}
                
                # Test each candidate in the grid.
                for thres in grid_points:
                    res = _test_feature_threshold(feature_i, thres, parent_loss_, best_improvement)
                    if len(res) > 0:
                        best_local = res
                        best_improvement = res["improvement"]
                
                # If no candidate was found in this grid, exit.
                if len(best_local) == 0:
                    break
                
                # Find the index of the best candidate in grid_points.
                idx = grid_points.index(best_local["threshold"])
                
                # If at boundary, use single side refinement
                if idx == 0 or idx == len(grid_points) - 1:
                    if idx == 0:
                        new_low = grid_points[idx]
                        new_high = grid_points[idx + 1]
                    else:
                        new_low = grid_points[idx - 1]
                        new_high = grid_points[idx]
                else:
                    # Define new search boundaries as the adjacent grid points.
                    new_low = grid_points[idx - 1]
                    new_high = grid_points[idx + 1]
                    
                # Count how many unique values lie within the new interval.
                interval_points = [v for v in unique_values if new_low <= v <= new_high]
                
                # If the interval is sufficiently resolved, perform exhaustive search on these.
                if len(interval_points) < tol_points:
                    best_exhaustive = {}
                    for it, th in enumerate(interval_points):
                        if it == 0:
                            continue
                        thres = (interval_points[it-1] + interval_points[it]) / 2
                        res = _test_feature_threshold(feature_i, thres, parent_loss_, best_improvement)
                        if len(res) > 0:
                            best_exhaustive = res
                            best_improvement = res["improvement"]
                    candidate = best_exhaustive if len(best_exhaustive) > 0 else best_local
                    break
                else:
                    # Update the boundaries and repeat.
                    low, high = new_low, new_high
                    candidate = best_local
                    
            return candidate

        # Iterate over features.
        for i, feat in enumerate(seq_features):
            
            # If the feature is NOT used, continue
            if feat not in self.use_features_idx:
                continue
            
            # Get sorted unique values for feature feat.
            feature_column = X[:, feat]
            unique_values = feature_column.flatten().unique().sort().to_list()
            if len(unique_values) <= 1:
                continue
            
            # Run grid search on this feature.
            candidate = _grid_search_threshold(feat, unique_values, X, y, weights, 
                                               parent_loss_=parent_loss.copy(), 
                                               best_improvement=best_split["improvement"].copy(), 
                                               grid_k=grid_k, 
                                               random_state=random_state,
                                               variant=variant,
                                               percentile=percentile)
            
            # Update best_split if this candidate is better.
            if len(candidate) > 0 and candidate["improvement"].data > best_split["improvement"].data:
                best_split = candidate
                
        return best_split
    
    def _get_mask(self, X: Matrix | Tensor, feature_index: int, threshold: float, *, left=True, **kwargs):
        """
        Return a boolean mask for the rows of X based on the threshold at the given feature index.
        
        Parameters:
            left: if True, returns mask for rows where X[:, feature_index] <= threshold;
                  otherwise, returns mask for rows where X[:, feature_index] > threshold.
        
        Args:
            Matrix | Tensor, with boolean indicating whether smaller or greater than the threshold.
        """
        if left:
            return self.typeclass(X[:, feature_index].data <= threshold, backend=X._backend, dtype=bool, device=X.device)
        else:
            return self.typeclass(X[:, feature_index].data > threshold, backend=X._backend, dtype=bool, device=X.device)
    
    def _compute_loss(self, prediction: Matrix | Tensor, y: Matrix | Tensor, *, weights: Matrix | Tensor | None = None, **kwargs):
        """
        Compute the loss between the prediction (a constant for a node) and the true target y.
        Dispatch to the corresponding external function.
        
        Special Args:
            weights: Matrix | Tensor | None, if non-None, it will pass to Metrics/Loss and should be used with Metrics like "wmse" or "wrmse",
                                                          and will be used in calculating the weighted leaf.
        
        Returns:
            Matrix | Tensor, even if the metrics is a scalar.
        """        
        if self.task == "regression":
            metrics = RegressionMetrics(prediction, y, metric_type = self.loss)
            return metrics.compute(weights = weights, **kwargs)
        else:
            # If gini, then we use the built-in gini
            if self.loss == 'gini':
                return self._gini_impurity(y, one_hot = self.one_hot)
            else:
                if self.n_classes > 2:
                    metrics = MultiClassificationMetrics(prediction, y, metric_type = self.loss)
                    return metrics.compute(floattype = self.floattype, weights = weights, **kwargs)
                else:
                    metrics = BinaryClassificationMetrics(prediction, y, metric_type = self.loss)
                    return metrics.compute(floattype = self.floattype, weights = weights, **kwargs)
                    
    def _prune_tree(self, node=None, X=None, y=None, weights=None, *, alpha: float = 0.001, **kwargs):
        """
        Recursively prune the tree using cost-complexity pruning.
        At each node, we compare:
        
            cost_if_pruned = (n_samples * avg_loss_at_node) + alpha * 1
            subtree_cost = (cost from left subtree + cost from right subtree)
        
        The loss is assumed to be an average per sample, so we multiply by the number
        of samples in that node (n). For leaves, the cost is computed as:
        
            leaf_cost = n * (avg loss at that leaf) + alpha
        
        Parameters:
            node: Dict, The current subtree (as a dict). If None, uses self.root.
            X: Matrix | Tensor, The subset of features (a Matrix or Tensor) for samples that reached this node.
               If None, uses self.original_X (the entire training set).
            y: Matrix | Tensor, The corresponding targets for the samples in X.
               If None, uses self.oroginal_y.
            weights: Matrix | Tensor | None, The weights to sample or to each prediction, can be None.
            alpha: float, The complexity parameter to penalize the number of leaves.
        
        Returns:
            A tuple (pruned_node, cost,  n_samples, num_leaves), where:
                pruned_node: the (possibly pruned) subtree.
                cost: the total cost = loss + ccp_alpha * (number of leaves) for this subtree.
                n_samples: total samples at this node
                num_leaves: the total number of leaves in the pruned subtree.
        """
        # On the first call, set node to root and X, y to the original training set.
        if node is None:
            node = self.root
        if X is None:
            X = self.original_X
        if y is None:
            y = self.original_y
        if weights is None and self.original_weights is not None:
            weights = self.original_weights
            
        # number of samples in current node
        n = X.shape[0]
        
        # loss scaling factor
        scaling_factor = 1 / self.total_samples

        # If the node is a leaf, compute its cost.
        if "prediction" in node:
            # For a leaf node, compute loss on the training samples that reached this leaf.
            leaf_pred = node["prediction"]
            loss_leaf = self._compute_loss(leaf_pred.reshape([1,-1]).repeat(y.shape[0], axis=0), y, weights = weights)
            return node, scaling_factor * n * loss_leaf + alpha * 1, n, 1

        # Otherwise, the node is internal.
        # Retrieve the splitting criterion stored in the node.
        feat_idx = node["feature_index"]
        thresh = node["threshold"]

        # Obtain the boolean masks for splitting X into left and right subsets.
        left_mask = self._get_mask(X, feat_idx, thresh, left=True)
        right_mask = self._get_mask(X, feat_idx, thresh, left=False)

        # Use the mask to partition the data. (This uses vectorized slicing, so there is no Python loop.)
        X_left = X[left_mask.to_numpy_array()]
        y_left = y[left_mask.to_numpy_array()]
        X_right = X[right_mask.to_numpy_array()]
        y_right = y[right_mask.to_numpy_array()]
        
        # Use the masks to obtain subsets of the weights if having.
        if weights is not None:
            left_weights = weights[left_mask.to_numpy_array()]
            right_weights = weights[right_mask.to_numpy_array()]
        else:
            left_weights = None
            right_weights = None

        # Recursively prune the left and right subtrees.
        pruned_left, cost_left, n_left, leaves_left = self._prune_tree(node["left"], X_left, y_left, left_weights, alpha=alpha)
        pruned_right, cost_right, n_right, leaves_right = self._prune_tree(node["right"], X_right, y_right, right_weights, alpha=alpha)

        # Combine the cost for the subtree as currently structured.
        subtree_cost = cost_left + cost_right
        subtree_leaves = leaves_left + leaves_right

        # Now compute the cost if we were to prune (collapse) this internal node into a single leaf.
        # Use all samples that reach the current node (X, y) to compute the aggregated prediction.
        pruned_leaf = self._create_leaf(y, weights = weights, keepdims = False)
        # returns a dict with key "prediction"
        loss_pruned = self._compute_loss(pruned_leaf["prediction"].reshape([1,-1]).repeat(y.shape[0], axis=0), y, weights = weights)
        pruned_cost = scaling_factor * n * loss_pruned + alpha * 1  # complexity: one leaf

        # Decide whether to prune this node.
        # > pruned_cost: use this node as a leaf, the cost computed
        # > subtree_cost: use the original nested structure, the cost computed
        if pruned_cost.data <= subtree_cost.data:
            # Prune it and set this as the leaf
            return pruned_leaf, pruned_cost, n, 1
        else:
            # Otherwise, update this node with the subtrees.
            node["left"] = pruned_left
            node["right"] = pruned_right
            node["pruned_cost"] = subtree_cost
            node["num_leaves"] = subtree_leaves
            return node, subtree_cost, n, subtree_leaves
                
    def predict(self, X: Matrix | Tensor, **kwargs) -> Matrix | Tensor:
        """
        Predict target values for samples in X.
        
        Returns:
            Matrix | Tensor, output of predictions.
        """
        if not self.root:
            raise TypeError("Please train the tree first.")
        
        # Type Check (must be an Object type).
        if isinstance(X, Object) == False:
            raise ValueError("Input dataset must be either Matrix and Tensor. Use Matrix(data) or Tensor(data) to convert.")
        
        # Dimension Check
        if len(X.shape) != 2:
            raise ValueError("Input feature `X` must be a tabular data with two dimensions.")
        if X.shape[1] != self.original_X.shape[1]:
            raise ValueError(f"Input feature `X` must have the same number of columns as the training data, which is {self.X.shape[1]}, but you have {X.shape[1]}")
        
        # Create a float typed X
        X_float = X.to(backend=X._backend, dtype = self.floattype, device=X.device)
        
        # Create a container for predictions with the same number of samples as X.
        predictions = self.typeclass.zeros_like(self.original_y[0], backend = X._backend, dtype = float).reshape([1, -1]).repeat(X.shape[0], axis=0)
        
        # Predict recursively.
        self._predict_recursive(X_float, self.root, predictions, indices=self.typeclass(list(range(X.shape[0])), backend=X._backend))
        
        return predictions

    def _predict_recursive(self, X: Matrix | Tensor, node: dict, predictions: Matrix | Tensor, indices: Matrix | Tensor, **kwargs):
        """
        Recursively traverse the tree: at each node, assign predictions to the corresponding indices.
        
        Parameters:
            X: Feature matrix for all samples.
            node: The current node (dictionary). If a leaf, 'prediction' key exists.
            predictions: A preallocated Matrix or Tensor object to store predictions.
            indices: Indices (or a mask) of rows in X that fall into this node.
            
        Returns: 
            None   
        
        """
        # If the node is a leaf, assign the leaf's prediction to all indices.
        if "prediction" in node:
            if len(indices) > 0:
                predictions[indices.to_numpy_array()] = node["prediction"]
            return
        
        # If no indices, return 
        if len(indices) == 0:
            return
        
        # Get the splitting criteria.
        feature_index = node["feature_index"]
        threshold = node["threshold"]
        
        X_subset = X[indices.to_list()]
        left_mask = self._get_mask(X_subset, feature_index, threshold, left=True)
        right_mask = self._get_mask(X_subset, feature_index, threshold, left=False)
        
        # Convert the boolean masks to index Matrix | Tensors
        left_indices = indices[self.typeclass.where(left_mask.flatten().data == True, backend=X._backend).flatten().to_numpy_array()]
        right_indices = indices[self.typeclass.where(right_mask.flatten().data == True, backend=X._backend).flatten().to_numpy_array()]
        
        # Recurse on the left and right children.
        self._predict_recursive(X, node["left"], predictions, left_indices)
        self._predict_recursive(X, node["right"], predictions, right_indices)
    
    def apply(self, X: Matrix | Tensor, **kwargs) -> tuple:
        """
        Apply the tree searching and return the index of the node each sample is corresponding to.
        > Deprecated: Legacy Apply method. But I haven't done the new one.
        
        Returns:
            Tuple of [
                Matrix | Tensor: the index of the node, unique to any given X, 1 dim, integers,
                np.array: the string representation (always in NUMPY) of the node, unique to any X given this tree, strings,
                ]
        """
        if not self.root:
            raise TypeError("Please train the tree first.")
        
        # Type Check (must be an Object type).
        if isinstance(X, Object) == False:
            raise ValueError("Input dataset must be either Matrix and Tensor. Use Matrix(data) or Tensor(data) to convert.")
        
        # Dimension Check
        if len(X.shape) != 2:
            raise ValueError("Input feature `X` must be a tabular data with two dimensions.")
        if X.shape[1] != self.original_X.shape[1]:
            raise ValueError(f"Input feature `X` must have the same number of columns as the training data, which is {self.X.shape[1]}, but you have {X.shape[1]}")
        
        # Create a float typed X
        X_float = X.to(backend=X._backend, dtype = self.floattype, device=X.device)
        
        # Create a container for node indices with 1 column but the same row number as X
        node_indices = self.typeclass.zeros((1), backend = X._backend, dtype = int).reshape([1, 1]).repeat(X.shape[0], axis=0)
        node_representation = node_indices.to_numpy_array().astype(str)
        node_representation[...] = "" # Initialize them with empty strings
        
        # Apply recursively.
        self._apply_recursive(X_float, self.root, node_representation, indices=self.typeclass(list(range(X.shape[0])), backend=X._backend))
        
        # Given the node_representation has been calculated, we calculate the indices as integers
        list_node_representation = node_representation.flatten().tolist()
        def _to_count_order_dict(lst):
            """
            Generate a dict where each key is a unique value from lst.
            The value is a tuple: (count of occurrences, rank in sorted order, starting at 1).
            """
            counts = Counter(lst)
            unique_sorted = sorted(counts)
            return {value: (counts[value], index + 1)
                    for index, value in enumerate(unique_sorted)}

        # This dict stores the sorted values and encountered times
        count_order_dict = _to_count_order_dict(list_node_representation)
        list_indices = [count_order_dict[item][1] for item in list_node_representation]
        
        return self.typeclass(list_indices, backend = X._backend, dtype = int).reshape([-1, 1]), node_representation

    def _apply_recursive(self, X: Matrix | Tensor, node: dict, node_representation: np.ndarray, indices: Matrix | Tensor, **kwargs):
        """
        Recursively traverse the tree: Assign the path (0 for left 1 for right N for node) to find the leaf node for each sample.
        
        Parameters:
            X: Feature matrix for all samples.
            node: The current node (dictionary). If a leaf, 'prediction' key exists.
            node_representation: An empty np column array (n, 1) represent the string-ized node path, but lefr '' empty string when passed in.
                  In this function, algorithms will add pathes to the `node_representation`.
            indices: Indices (or a mask) of rows in X that fall into this node.
            
        Returns: 
            None   
        
        """
        # Helper Function: Append some character to a slice of representation
        def _append_sliced(representation: np.ndarray, slices: Matrix | Tensor, column: int = 0, to_append: str = "0"):
                representation[slices, column] = np.char.add(representation[slices.flatten().to_numpy_array(), column], to_append)
                
        # If the node is a leaf, assign the leaf's prediction to all indices.
        if "prediction" in node:
            if len(indices) > 0:
                _append_sliced(node_representation, indices, column = 0, to_append = "N")
            return
        
        # If no indices, return 
        if len(indices) == 0:
            return
        
        # Get the splitting criteria.
        feature_index = node["feature_index"]
        threshold = node["threshold"]
        
        X_subset = X[indices.to_list()]
        left_mask = self._get_mask(X_subset, feature_index, threshold, left=True)
        right_mask = self._get_mask(X_subset, feature_index, threshold, left=False)
        
        # Convert the boolean masks to index Matrix | Tensors
        left_indices = indices[self.typeclass.where(left_mask.flatten().data == True, backend=X._backend).flatten().to_numpy_array()]
        right_indices = indices[self.typeclass.where(right_mask.flatten().data == True, backend=X._backend).flatten().to_numpy_array()]
        
        # Create left and right path
        _append_sliced(node_representation, left_indices, column = 0, to_append = "0")
        _append_sliced(node_representation, right_indices, column = 0, to_append = "1")
        
        # Recurse on the left and right children.
        self._apply_recursive(X, node["left"], node_representation, left_indices)
        self._apply_recursive(X, node["right"], node_representation, right_indices)
    
    def plot_tree(self, figsize = (14, 8), **kwargs):
        """
        Plot an image representing the structure of the decision tree.
        
        This function traverses the tree recursively, assigns (x,y) positions to each node,
        and then uses matplotlib to render the tree structure. For internal nodes, the label
        is in the form "Feature <= threshold" (with threshold rounded to 4 digits).
        
        For leaf nodes, the label shows the prediction.
        
        Args:
            figsize: tuple, the size of the plot
        """
        if self.root is None:
            raise ValueError("Tree has not been built yet. Please fit the tree first.")

        # Dictionaries to store node positions and labels, and a list for edges.
        node_positions = {}
        node_labels = {}
        edges = []
        
        # Mutable counters to keep track of node id and horizontal position.
        counter = [0]        # For node IDs.
        current_x = [0]      # For assigning x-coordinate in an in-order fashion.

        def assign_positions(node, depth):
            """
            Recursively assign positions to each node.
            
            Returns:
                A tuple of (x_position, node_id) for the current node.
            """
            my_id = counter[0]
            counter[0] += 1

            # If leaf node, assign current x position and label using prediction.
            if "prediction" in node:
                pos = (current_x[0], -depth)
                node_positions[my_id] = pos
                # Extract prediction; if it is a Matrix/Tensor, extract its scalar list.
                pred = node["prediction"]
                if hasattr(pred, "to_list"):
                    pred = pred.to_list()
                # Format prediction as a rounded list (if list) or scalar.
                def force_round(num, n=4):
                    return f"{num:.{n}f}"
                if isinstance(pred, list):
                    pred_str = ",\n ".join(str(force_round(x, 4)) for x in pred) + ","
                else:
                    pred_str = str(round(pred, 4))
                node_labels[my_id] = pred_str
                current_x[0] += 1
                return node_positions[my_id], my_id

            # Else, process an internal node.
            else:
                # Recursively assign position to the left subtree.
                left_pos, left_id = assign_positions(node["left"], depth + 1)
                # Reserve current node id for the internal node.
                current_node_id = my_id
                # Recursively assign position to the right subtree.
                right_pos, right_id = assign_positions(node["right"], depth + 1)
                # Set current node's x position as the average of its children's x positions.
                x = (node_positions[left_id][0] + node_positions[right_id][0]) / 2
                pos = (x, -depth)
                node_positions[current_node_id] = pos

                # Format the internal node label using the feature name and threshold.
                feature_name = self.feature_index_map.get(node["feature_index"], f"F{node['feature_index']}")
                thresh_val = node["threshold"]
                if hasattr(thresh_val, "to_list"):
                    thresh_val = thresh_val.to_list()
                    if isinstance(thresh_val, list):
                        thresh_val = thresh_val[0] if len(thresh_val) == 1 else thresh_val
                node_labels[current_node_id] = f"{feature_name} <= {round(thresh_val, 4)}"
                # Record edges from the current node to its left and right children.
                edges.append((current_node_id, left_id))
                edges.append((current_node_id, right_id))
                return pos, current_node_id

        # Build node positions and labels from the tree starting at the root at depth 0.
        _, root_id = assign_positions(self.root, depth=0)

        # Create the plot.
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot each edge.
        for parent, child in edges:
            x_vals = [node_positions[parent][0], node_positions[child][0]]
            y_vals = [node_positions[parent][1], node_positions[child][1]]
            ax.plot(x_vals, y_vals, 'k-', lw=1)
        # Plot the nodes with their labels.
        for node_id, (x, y) in node_positions.items():
            ax.text(x, y, node_labels[node_id], ha='center', va='center',
                    bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'))
        ax.set_title(f"CART Tree (ID: {self.tree_id})")
        ax.axis('off')
        plt.show()

    def __repr__(self):
        return f"CART(task = {self.task}, loss = {self.loss}, n_features = {self.original_X.shape[1]})."


if __name__ == "__main__":
    
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import classification_report, accuracy_score
    import pandas as pd
    
    ###########################################################################
    #
    # Generate the data
    def generate_binary_classification_data(n_samples=10000, n_features=30, n_informative=10, n_redundant=0, random_state=None):
        X, y = make_classification(n_samples=n_samples,
                                   n_features=n_features,
                                   n_informative=n_informative,
                                   n_redundant=n_redundant,
                                   n_classes=3,
                                   random_state=random_state)
        return pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])]), pd.Series(y, name='label')
    
    # Data
    X, y = generate_binary_classification_data()
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
    
    ###########################################################################
    #
    # Reference
    # CART model using sklearn
    cart_model = DecisionTreeClassifier(criterion='gini', max_depth=12, random_state=2821)
    cart_model.fit(X_train, y_train)
    
    # Predictions
    y_pred = cart_model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    ###########################################################################
    
    ###########################################################################
    #
    # Construct my CART tree
    backend = "numpy"
    device = "cpu"
    cart = CART("classification", max_depth=12, loss="gini",
                min_samples_split=0.0002, 
                prune=True,
                prune_alpha=0.0004,
                random_state=None,
                grid_accelerator=10,
                grid_use_percentile=False,
                grid_point_variant=0.05
                )
    re_train_y = cart._to_onehot(Matrix(y_train.to_numpy(), backend, device=device), 3).astype(int)
    
    # Fit the tree
    cart.fit(Matrix(X_train.to_numpy(), backend, device=device), re_train_y, use_features_idx=None)
    
    # Or fit a weighted tree
    weights = Matrix(X_train.abs().to_numpy(), backend, device=device)[:,[0]]
    cart.fit(Matrix(X_train.to_numpy(), backend, device=device), re_train_y, weights=weights, use_features_idx=None)
    
    # Predict the labels
    n = 9999
    y_pred_ = cart.predict(Matrix(X_test[0:n].to_numpy(), backend, device=device))
    y_actual_ = Matrix(y_test[0:n].to_numpy(), backend, device=device).reshape([-1, 1])
    
    from metrices import MultiClassificationMetrics
    ct = MultiClassificationMetrics(y_pred_, y_actual_, metric_type="accuracy", )
    print("CART Accuracy:", ct.compute())
    
    # Plot the tree
    cart.plot_tree([40,10])
    
    # Plot the feature importance
    cart.plot_feature_importance(10, [15, 9])
    
    