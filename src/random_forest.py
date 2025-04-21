# random_forest.py
#
# Random Forest Classfier and Regressor
# From MML Library by Nathmath

import math
import lzma

import numpy as np
try:
    import torch
except ImportError:
    torch = None

from objtyp import Object
from tensor import Tensor
from matrix import Matrix

from dump import save, load

from copy import deepcopy
from typing import List, Dict, Tuple, Any

from threadpool import Mutex, ThreadPool

from baseml import MLBase, Regression, Classification

from metrices import RegressionMetrics
from metrices import BinaryClassificationMetrics
from metrices import MultiClassificationMetrics

from tree import BaseTree, CART
from tree_wrapper import LRTW
from ensemble import Ensemble, Bagging
from aggregation import ClassificationAggregation
from aggregation import RegressionAggregation

import matplotlib.pyplot as plt


# Bagging‑style Random Forest Implementation
class RandomForest(Bagging):

    __attr__ = "MML.RandomForest"    

    def __init__(self, task: str = "classification", agg_method: str = "mean",
                       *,
                       tree_type: type = CART,
                       n_estimators: int = 10,
                       max_features: str | int | float | None = 0.6,
                       bootstrap_ratio: float = 1.0,
                       replace: bool = True,
                       shuffle: bool = True,
                       floattype: type = float,
                       random_state: int | None = None,
                       n_workers: int | None = None,
                       feature_names: Matrix | Tensor | None = None,
                       tree_kwargs: dict = {},
                       agg_kwargs: dict = {},
                       **kwargs) -> None:
        """
        Parameters
        ----------
        task : str, one of {"classification", "regression"}, showing the learning task.
        agg_method: str, the name of aggregation method. See aggregation.py.
                    for classification, common ones are: "mean", "hard_vote", "soft_vote", ...
                    for regression, common ones are: "mean", "median", "percentile", "weighted", ...
        Optional:
            tree_type: type, showing the type of tree you are intending to use. Default, CART. 
                       You may pass in some other tree-compatible classes, like LRTW.
            n_estimators : int, indicating the maximum number of trees.
            max_features : {None, "sqrt", "log2", "over3"} | int | float, number of columns available to each tree.
            bootstrap_ratio : float, fraction of samples drawn per bootstrap sample, [0, 1].
            replace, shuffle : bool, if to shuffle the bootstrapped samples, passed to `_sample_bootstrapping`.
            floattype : type, numerical precision stored in internal matrices.
            random_state : int | None, global seed for reproducibility.
            n_workers: int | None, the number of threads working concurrently when building trees.
            feature_names : Matrix | Tensor | None, optional feature labels, in Object type. May be useful in feature importance.
            tree_kwargs : dict, extra hyperparameters forwarded to every tree model.
            agg_kwargs : dict, extra hyperparameters forwarded to the aggregation instance.
            **kwargs: other key word arguments, reserved for compatibility use.
        """
        
        super().__init__(feature_names = feature_names)

        # Record task and aggregation
        self.task  = task.lower()
        if self.task not in ("classification", "regression"):
            raise ValueError("Input `task` must be 'classification' or 'regression'")
        self.agg_method = agg_method
        
        # Random Forest Key arguments
        self.n_estimators = int(n_estimators) # Specified Number of Estimators
        self.n_estimators_used = 0            # Trainned Number of Estimators
        # Note, early stopping may prevent using all of the estimators
        
        # Random Forest Data ROW/COL settings
        self.max_features = max_features
        if  max_features is None:
            max_features = 1.0
        self.bootstrap_ratio = float(bootstrap_ratio)
        if self.bootstrap_ratio > 1.0 or self.bootstrap_ratio <= 0.0:
            raise ValueError("Extrapulate for bootstrap ratio is NOT allowed. You must set a value in (0, 1]")
            
        # Bootstrap and Feature Selecting arguments
        self.replace = replace
        self.shuffle = shuffle
        
        # Global random_state (Use _random_state_next() to move it forward)
        self.random_state = random_state
        
        # Threadpool Primitive
        self.n_workers = n_workers

        # Type Related Variables.
        self.tree_type = tree_type
        if isinstance(self.tree_type(), BaseTree) == False:
            raise ValueError(f"The given weak leaner type {tree_type} must be a child class of BaseTree.")
        self.floattype = floattype
        self.typeclass = None # Determined in fit()

        # KWargs for Trees and Aggregation
        self.tree_kwargs = tree_kwargs
        self.agg_kwargs = agg_kwargs   
        self.kwargs = kwargs
        
        # Original Dataset.
        self.original_X = None
        self.original_y = None
        
        # Processed Datasets.
        self._feature_sets = {}      # Key: tree id, starting from 0
                                     # Value: feature indices used
        self._bootstrapped = {}      # Key: tree id, starting from 0
                                     # Value: bootstrapped data tuple (see. bagging.py)

        # Runtime Containers (Forest Ensemble).
        self._estimators = {}        # Key: tree id, starting from 0
                                     # Value: tree instance
    
    def fit(self, X: Matrix | Tensor, y: Matrix | Tensor,
            *,
            one_hot: bool = True,
            verbosity: int | None = None,
            sequential_batch: int | None = None,
            evalset: Dict[str, Tuple[Matrix | Tensor, Matrix | Tensor]] | None = None,
            evalmetrics: List[str] | str | None = None,
            early_stop: int | None = None,
            early_stop_logic: str = "some",
            continue_to_train: bool | None = None):
        """
        Train n_estimators independent trees.
        You may want to evaluate datasets while training. If so, please do the following things:
            1. set `verbosity` = 1 to print the evaluation
            2. set a proper `sequential_batch`, for 1 or 4 depending on cores of your CPU
            3. set the `evalset` to a dict of tuples of your dataset that is going to be evaluated
            4. set the `evalmetrics` either to a string of metrics or a list of strings
        You may want the algorithm to decide to stop training automatically. If so, please do things above, plus:
            1. set `early_stop` to a number of batches, like 1 or 2, which acts like: 
                if the metrics for all/some/any/most of the evaluation sets do not decrease anymore, 
                the training process will be terminated and return
            2. set `early_stop_logic` to determine the way of processing non-decreasing datasets/metrics
            3. If you hope to continue to train again, call this `fit` again with `continue_to_train` set to True

        Parameters:
            ----------
            X: Matrix | Tensor, the feature matrix (each row is a sample).
            y: Matrix | Tensor, the target values (for regression, numerical; for classification, one-hot or multi-label).
            Optional:
                one_hot : bool, if y is one-hot encoded for classification tasks.
                verbosity: int | None, if >= 1 and having `evalset`, then will report metrics each batch.
                sequential_batch: int | None, if int, then train `batch` estimators as a batch before evaluating.
                evalset: Dict[name : Tuple[X, y],
                              ...], | None, if provided, it may be used as evaluation set. XGBoost style.
                evalmetrics: list of str | str | None, metrics used to do the evaluation. Will be printed.
                early_stop: int | None, if non-None, then if metrics NOT gained for `early_stop` times, the forest will stop training.
                early_stop_logic: str, the logic when deciding on multiple metrics, can be {"any", "some", "most", "all"}.
                continue_to_train: bool | None, if non-None and True, the machine will try to restore the place it was and continue
                                   to train new estimators until a new stopping criterion meets or until reaches the max number of allowed estimators.
                
        Returns:
            ----------
            self
        """

        # Type Check (must be an Object type).
        if isinstance(X, Object) == False or isinstance(y, Object) == False:
            raise ValueError("Input dataset must be either Matrix and Tensor. Use Matrix(data) or Tensor(data) to convert.")
        if type(X) != type(y):
            raise ValueError("Input feature `X` and target `y` must have the same type, either Matrix or Tensor.")
        
        # Dimension Check.
        if len(X.shape) != 2:
            raise ValueError("Input feature `X` must be a tabular data with two dimensions.")
        if len(y.shape) == 1:
            raise ValueError("Input target `y` must also be a 2d data. If only one label or value, use data.reshape([-1, 1])")
                    
        # Stopping Logic Check.
        if early_stop_logic not in ("any", "some", "most", "all"):
            raise ValueError("Stopping logic `early_stop_logic` must be one of ('any', 'some', 'most', 'all')")
            
        # Prepare the subsets and sub-features
        self._fit_prep(X = X, y = y)
        n_samples, n_features = X.shape
        
        # Special evalmetrics type conversion
        if isinstance(evalmetrics, str) == True:
            evalmetrics = [evalmetrics]
            
        # Verbosity Conversion
        verbosity = verbosity if verbosity is not None else 0
        
        # Helper: Print and decide the evaluated results
        def _decide_stop_with_print(batch: int, undecreased_no: int, eval_dict: dict, last_eval_dict: dict):
            """
            Compare the metics and decide if

            Parameters
                ----------
                batch: int, batch no, for printing uses.
                undecreased_no : int, cumulative number that loss did NOT decrease before evaluation.
                eval_dict : dict, the passed evaluation dict.

            Returns
                -------
                Tuple of (int, bool):
                    int, updated undecreased_no
                    bool, whether to stop (True) training or continue (False)

            """
            # Dict is empty, abort
            if len(eval_dict) == 0:
                return undecreased_no, False
            if len(last_eval_dict) == 0:
                return undecreased_no, False
            
            # Difference dict copy
            diff_dict = deepcopy(eval_dict)
            
            # Calculate the difference (this - last)
            # and
            # If verbosity, print the new evaluation dict
            undes_count = 0
            allmetric_count = 0
            if verbosity >= 1:
                for evalset_name in eval_dict.keys():
                    eval_result = eval_dict[evalset_name]
                    print("Evalset: [", evalset_name, " : Metrics {", end = " ", sep = "")
                    for metric_name in eval_result.keys():
                        metric_value = eval_result[metric_name]
                        diff_dict[evalset_name][metric_name] = metric_value - last_eval_dict[evalset_name][metric_name]
                        if diff_dict[evalset_name][metric_name].to_list() > 0:
                            undes_count += 1
                        allmetric_count += 1
                        print(metric_name, ":", round(metric_value.to_list(), 4), ", ", end = " ", sep = "")
                    print("}]", end = "\n")
                    
            # If no early stop, directly return 0, False
            if early_stop is None:
                return 0, False
                    
            # If meets the requirement, stop training
            if early_stop_logic == "any":
                if undes_count > 0:
                    undecreased_no += 1
                    if undecreased_no >= early_stop:
                        return undecreased_no, True
            elif early_stop_logic == "some":
                if undes_count * 3 >= allmetric_count:
                    undecreased_no += 1
                    if undecreased_no >= early_stop:
                        return undecreased_no, True
            elif early_stop_logic == "most":
                if undes_count * 2 >= allmetric_count:
                    undecreased_no += 1
                    if undecreased_no >= early_stop:
                        return undecreased_no, True
            elif early_stop_logic == "all":
                if undes_count * 1 >= allmetric_count:
                    undecreased_no += 1
                    if undecreased_no >= early_stop:
                        return undecreased_no, True
                    
            # If survives here, return 0, False to refresh the undecreased_no
            return 0, False
        
        round_ = 0
        tree_id = 0
        undecreased_no = 0
        last_eval_dict = {} # Please use deepcopy() here to avoid being errorly referred
        
        # Continue to train? Restore the last point
        if continue_to_train is not None:
            if continue_to_train == True:
                tree_id = self.n_estimators_used
        
        # Build up the forest - semi-sequentially or parallelly
        while tree_id < self.n_estimators:
            
            # Verbosity
            if verbosity >= 1:
                print(f"Training on Round: {round_}, starting from tree: {tree_id}, with {sequential_batch if sequential_batch is not None else 1} trainned concurrently.")
            
            # Train
            if sequential_batch is not None:
                # Use batch training
                tree_ids = []
                if tree_id + sequential_batch > self.n_estimators:
                    while tree_id < self.n_estimators:
                        tree_ids.append(tree_id)
                        tree_id += 1
                else:
                    tree_ids = [x for x in range(tree_id, tree_id + sequential_batch)]
                    tree_id += sequential_batch
                # Semi-sequantially train the trees
                self._train_semi_sequential(tree_ids, one_hot = one_hot)
                
            else:
                # Sequentially train the trees
                self._train_one_tree(tree_id, one_hot = one_hot)
                tree_id += 1
                
            # Evaluate and decide
            eval_dict = self._eval_one_batch(sequential_batch, evalset = evalset, evalmetrics = evalmetrics, one_hot = one_hot)
            
            # Try stop maker and receive the advice
            undecreased_no, decision = _decide_stop_with_print(round_, undecreased_no = undecreased_no, eval_dict = eval_dict, last_eval_dict = last_eval_dict)
            
            # Copy last evaluated dict
            last_eval_dict = deepcopy(eval_dict)
            
            # Count self increasing
            round_ += 1
            
            # Make decision to terminate or not
            if decision == True:
                break
            
        return self

    def _fit_prep(self, X: Matrix | Tensor, y: Matrix | Tensor):
        """
        Prepare datasets for n_estimators and selected features.

        Parameters:
            ----------
            X: Matrix | Tensor, the feature matrix (each row is a sample).
            y: Matrix | Tensor, the target values (for regression, numerical; for classification, one-hot or multi-label).
        
        Returns:
            ----------
            self
        """
        # Type Check (must be an Object type).
        if isinstance(X, Object) == False or isinstance(y, Object) == False:
            raise ValueError("Input dataset must be either Matrix and Tensor. Use Matrix(data) or Tensor(data) to convert.")
        if type(X) != type(y):
            raise ValueError("Input feature `X` and target `y` must have the same type, either Matrix or Tensor.")
        
        # Dimension Check.
        if len(X.shape) != 2:
            raise ValueError("Input feature `X` must be a tabular data with two dimensions.")
        if len(y.shape) == 1:
            raise ValueError("Input target `y` must also be a 2d data. If only one label or value, use data.reshape([-1, 1])")
        
        # Copy Training data.
        self.original_X = X.to(backend=X._backend, dtype = self.floattype, device=X.device)
        self.original_y = y.to(backend=y._backend, dtype = self.floattype, device=y.device)
        
        # Resolve Number of Features for each weak learner.
        n_samples, n_features = X.shape
        self.max_features = self._resolve_max_features(n_features, q = self.max_features)
        
        # Bootstrap row indices, generate row-sliced subsets.
        # Prepare all n_estimators even somethimes not fully used.
        self._bootstrapped = self._sample_bootstrapping(
            X, y,
            M = self.n_estimators,
            k = self.bootstrap_ratio,
            replace = self.replace,
            shuffle = self.shuffle,
            random_state = self._random_state_next(),
            container = dict)

        # Retrieve the subset of features.
        # Prepare all n_estimators even somethimes not fully used.
        self._feature_sets = self._feature_random_select(
            M = self.n_estimators,
            N = n_features,
            q = self.max_features,
            replace = False,  # Default standarad
            random_state = self._random_state_next(),
            container = dict)
        
        # Restore trained trees and trained numbers
        self._estimators = {}
        self.n_estimators_used = 0
        
        return self

    def _resolve_max_features(self, n_features_all: int, q: int | float | str | None) -> int:
        """
        Translate max_features spec into an integer 1 ≤ q ≤ n_features_all.
        
        Parameters:
            ----------
            n_features_all: int, number of total features you have.
            q: int | float | str | None, feature number/proportion/description or None
        
        Returns:
            ----------
            int, number of features selected (parsed from the input, which may not be an int)
        """
        p = n_features_all
        
        # String: sqrt or log2
        if isinstance(q, str):
            q = q.lower()
            if q == "sqrt":
                return max(1, int(p ** 0.5))
            if q == "log2":
                return max(1, int(math.log2(p)))
            if q == "over3":
                return max(1, int(p // 3))
            raise ValueError("Resolving the max features, max_features `q` may be None, int, 'sqrt', or 'log2'")
        if isinstance(q, int):
            if q <= 0:
                raise ValueError("Resolving the max features, max_features `q` must be positive")
            return min(q, p)
        if isinstance(q, float):
            if q > 1:
                raise ValueError("Resolving the max features, max_features `q` must be less than 1 if float type")
            return max(1, int(p * q))
        raise ValueError("Unsupported max_features `q` type. Coult be int, float, str and None")

    def _train_one_tree(self, tree_id: int, one_hot: bool, *, _mutex : Mutex | None = None) -> None:
        """
        Train one tree based on the split data.

        Parameters
        ----------
            tree_id : int, the tree_id you expect to train.
                      The tree must NOT be initialied, else error.
            one_hot: bool, if classification and using one-hot coding.

        Returns
            -------
            None.

        """
        # Tree id validity check
        if tree_id > self.n_estimators:
            raise ValueError(f"You are training tree_id {tree_id}, but you only have {self.n_estimators} reserved.")
        
        # If the tree is trained
        if self._estimators.get(tree_id, None) is not None:
            raise ValueError(f"You are training tree_id {tree_id}, but the tree has been already trained.")
            
        # If datasets are not ready
        if self._bootstrapped.get(tree_id, None) is None or self._feature_sets.get(tree_id, None) is None:
            raise ValueError(f"You are training tree_id {tree_id}, but datasets/feature indices are not prepared.")
            
        X_selected, y_selected, _indices = self._bootstrapped[tree_id]
        feature_idx = self._feature_sets[tree_id]
        
        tree = self.tree_type(task = self.task, tree_id = tree_id, 
                              random_state = self._random_state_next(), floattype = self.floattype, feature_names = self.feature_names, **self.tree_kwargs)
        tree.fit(X_selected, y_selected, one_hot = one_hot, use_features_idx = feature_idx)

        # Apply lock to protect the resources
        if _mutex is not None:
            with _mutex:
                self._estimators[tree_id] = tree
                self.n_estimators_used += 1
            return
        else:
            self._estimators[tree_id] = tree
            self.n_estimators_used += 1

    def _train_semi_sequential(self, tree_ids: List[int] | None, one_hot: bool):
        """
        Train one batch of trains using ThreadPool based on the split data.

        Parameters
        ----------
            tree_ids : list of int, the tree_id you expect to train.
                       The tree must NOT be initialied, else error.
            one_hot: bool, if classification and using one-hot coding.

        Returns
            -------
            None.
        """
        # Task IDs
        task_ids = []
        
        # Without Check, just invoke the TheadPool
        mutex = Mutex()
        fittingpool = ThreadPool(self.n_workers if self.n_workers is not None else 1)
        for tree_id in tree_ids:
            task_id = fittingpool.execute(self._train_one_tree, tree_id = tree_id, one_hot = one_hot, _mutex = mutex)
            task_ids.append(task_id)
            
        # Wait until all 
        for task_id in task_ids:
            fittingpool.waituntil(task_id)
        return
        
    def _eval_one_batch(self, sequential_batch: int | None = None, evalset: Dict[str, Tuple[Matrix | Tensor, Matrix | Tensor]] | None = None, evalmetrics: List[str] | str | None = None, one_hot: bool = True):
        """
        Evaluate the `evalset` after training for one batch.    

        Returns
            -------
            result_dict : dict  # Key: evalset name
                                # Value dict {metric_name: metric_value}
            or 
            {} if failed or did not evaluate
        """
        # If:
        # 1. sequential_batch non-None
        # 2. evalset is at least len = 1
        # 3. evalmetrics is non-None and at least len = 1
        # Do evaluation
        result_dict = {}
        if sequential_batch is not None and evalmetrics is not None and evalset is not None:
            if len(evalset) > 0 and len(evalmetrics) > 0:
                # Record the result for each eval group
                result_dict = {}    # Key: evalset name
                                    # Value dict {metric_name: metric_value}
                for eval_name in evalset.keys():
                    X_sub, y_sub = evalset[eval_name]
                    y_pred = self.predict(X_sub)
                    
                    # Inner metric dict, for values of result dict
                    metrics = {}
                    for metric_name in evalmetrics:
                        
                        # Evaluation: regression
                        if self.task == "regression":
                            eval_metric = RegressionMetrics(y_pred, y_sub, metric_type = metric_name).compute()
                            # Matrix | Tensor
                        
                        # Evaluation: this is classification
                        else:
                            if y_pred.shape[1] == 2 and one_hot == False:
                                # Binary and non-one hot
                                eval_metric = BinaryClassificationMetrics(self._to_binary_prob(y_pred), y_sub, metric_type = metric_name).compute()
                            else:
                                # Since the aggregation output is alway one-hot, use Multiple then
                                eval_metric = MultiClassificationMetrics(y_pred, y_sub, metric_type = metric_name).compute()
                            # Matrix | Tensor
                        metrics[metric_name] = eval_metric
                    # For all metrics, put them into result_dict
                    result_dict[eval_name] = metrics
        
        # If it is empty, then exit since nothing valid
        if len(result_dict) == 0:
            return {}
        
        # Else, return the dict
        else:
            return result_dict

    def predict(self, X: Matrix | Tensor) -> Matrix | Tensor:
        """
        Predict target values for samples in X.
        If classification, the output must be one-hot (even binary cases). Please kindly note.
        To convert one-hot to labels, use ._to_labels()
        
        Returns:
            Matrix | Tensor, output of predictions.
        """
        
        # Check if the model has (at least partially) fitted or not
        if self._is_fitted() == False:
            raise RuntimeError("No weak learner is fitted. Call .fit() before .predict()")
            
        # Type Check (must be an Object type).
        if isinstance(X, Object) == False:
            raise ValueError("Input dataset must be either Matrix and Tensor. Use Matrix(data) or Tensor(data) to convert.")
        
        # Dimension Check
        if len(X.shape) != 2:
            raise ValueError("Input feature `X` must be a tabular data with two dimensions.")
        if X.shape[1] != self.original_X.shape[1]:
            raise ValueError(f"Input feature `X` must have the same number of columns as the training data, which is {self.X.shape[1]}, but you have {X.shape[1]}")
        
        # Get the first trained tree model
        first_trained_model = list(self._estimators.keys())
        first_trained_model.sort()
        first_trained_model = self._estimators[first_trained_model[0]]
        
        # Get the first prediction
        first_prediction = first_trained_model.predict(X)
        
        # Collect predictions from estimators -> separate
        # !! Important: make the reuslt stacked over axis = 1
        #    with shape shape (n_estimators, n_samples, n_outputs) -> which IS a 3D array
        predictions_shape = (self.n_estimators_used, first_prediction.shape[0], first_prediction.shape[1])
        # Create an empty Matrix | Tensor Wrapper with 0 padded
        predictions = type(X).zeros(predictions_shape, backend = first_prediction._backend, dtype = first_prediction.dtype)
        predictions = predictions.to(backend = first_prediction._backend, dtype = first_prediction.dtype, device = first_prediction.device)
        
        # Iteratively predict
        estm_id = 0
        for tree_id in self._estimators.keys():
            estimator = self._estimators[tree_id]
            predictions[estm_id] = estimator.predict(X)
            estm_id += 1
        
        # choose an aggregation rule
        if self.task == "classification":
            agg = ClassificationAggregation(predictions, method = self.agg_method, floattype = self.floattype, **self.agg_kwargs)
        else:
            agg = RegressionAggregation(predictions, method = self.agg_method, floattype = self.floattype, **self.agg_kwargs)
        
        # Compute the aggreagated prediction
        return agg.compute()
    
    def _is_fitted(self) -> bool:
        """
        Check if the Random Forest has been fitted or not.

        Returns
            -------
            bool, if the model is fitted or not.

        """
        return True if len(self._estimators) > 0 else False

    def update_tree_kwargs(self, tree_kwargs : dict | None = None) -> None:
        """
        Update the tree-build key word arguments in whole.
        If you hope to replace or remove, please first get and then set by this.

        Returns
            -------
            None

        """
        if tree_kwargs is not None:
            self.tree_kwargs = tree_kwargs
            
    def update_agg_kwargs(self, agg_kwargs: dict | None = None) -> None:
        """
        Update the aggregation related key word arguments in whole.
        If you hope to replace or remove, please first get and then set by this.

        Returns
            -------
            None

        """
        if agg_kwargs is not None:
            self.agg_kwargs = agg_kwargs

    def plot_tree_id(self, tree_id: int, figsize = (14, 8)):
        """
        Plot an image representing the structure of the `tree_id`-th decision tree within the forest.
        For leaf nodes, the label shows the prediction.
        It calls the tree's method to do the plot.
        
        Args:
            tree_id: int, the index of the tree to be plotted
            figsize: tuple, the size of the plot
        """
        # Check if the model has (at least partially) fitted or not
        if self._is_fitted() == False:
            raise RuntimeError("No weak learner is fitted. Call .fit() before plotting the tree structure.")
            
        if self._estimators.get(tree_id, None) is None:
            raise RuntimeError(f"Tree {tree_id} has NOT been trained. Please train or continue to train before plotting the tree structure.")
        self._estimators[tree_id].plot_tree(figsize = figsize)

    def plot_feature_importances_tree_id(self, tree_id: int, max_features: int = None, figsize=(8, 5)) -> None:
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
        # Check if the model has (at least partially) fitted or not
        if self._is_fitted() == False:
            raise RuntimeError("No weak learner is fitted. Call .fit() before plotting the feature importance.")
            
        if self._estimators.get(tree_id, None) is None:
            raise RuntimeError(f"Tree {tree_id} has NOT been trained. Please train or continue to train before plotting the tree structure.")
        self._estimators[tree_id].plot_feature_importance(max_features = max_features, figsize = figsize)

    def plot_feature_importances_average(self, max_features: int = None, figsize=(8, 5)) -> None:
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
            None.

        """
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
            feat_imp_list.append((feature_name, round(val, 4)))
        
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
        ax.set_title("Average Feature Importance (Gain)")
        
        # Add text labels on top of the bars.
        for idx, v in enumerate(importances):
            ax.text(idx, v, str(v), ha='center', va='bottom')
        plt.show()

    def __repr__(self):
        return f"RandomForest(task = {self.task}, tree_type = {str(self.tree_type)}, n_estimators = {self.n_estimators})."
    
    
if __name__ == "__main__":
    
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    import pandas as pd
    
    ###########################################################################
    #
    # Generate the data
    def generate_binary_classification_data(n_samples=4000, n_features=40, n_informative=20, n_redundant=0, random_state=None):
        X, y = make_classification(n_samples=n_samples,
                                   n_features=n_features,
                                   n_informative=n_informative,
                                   n_redundant=n_redundant,
                                   n_classes=2,
                                   random_state=random_state)
        return pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])]), pd.Series(y, name='label')
    
    # Data
    X, y = generate_binary_classification_data()
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
    
      
    ###########################################################################
    #
    # Reference
    # RandomForest from sklearn
    clf = RandomForestClassifier(n_estimators = 10, 
                                 max_features = 20,
                                 max_depth = 20,
                                 random_state = None)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    # Report accuracy
    print(f"Test set accuracy: {accuracy_score(y_test, y_pred):.4f}")
    
    ###########################################################################
    #
    # Construct my RandomForest
    frontend = Matrix
    backend = "numpy"
    device = "cpu"
    tree_kwargs = {"max_depth":20, "loss":"logloss",
                   "min_samples_split":0.005, 
                   "prune":True, "prune_alpha":0.0006, 
                   "grid_accelerator":20, "grid_use_percentile":False,
                   "grid_point_variant":None}
    
    # Initialize a RandomForest
    X_train_x = frontend(X_train.to_numpy(), backend, device=device)
    re_train_y = frontend(y_train.to_numpy(), backend, device=device).reshape([-1, 1])
    X_test_x = frontend(X_test.to_numpy(), backend, device=device)
    y_actual_ = frontend(y_test.to_numpy(), backend, device=device).reshape([-1, 1])
    
    rf = RandomForest(task = "classification", agg_method = "soft_vote",
                      n_estimators = 10,
                      n_workers = 2,
                      max_features = 0.6,
                      bootstrap_ratio = 1.0,
                      random_state = None,
                      tree_kwargs = tree_kwargs,
                      agg_kwargs = {})
    # Fit the tree
    rf.fit(X_train_x, 
           re_train_y, 
           one_hot = False,
           verbosity = 1,
           evalset = {"train": (X_train_x, re_train_y), "test": (X_test_x, y_actual_)},
           evalmetrics = ["accuracy", "logloss"],
           sequential_batch = 1, continue_to_train=True)
    
    # Predict the labels
    n = 9999
    y_pred_ = rf.predict(X_test_x[0:n])
    binary_y_pred_ = rf._to_binary_prob(y_pred_)
    
    # Evaluate the test set
    ct = BinaryClassificationMetrics(binary_y_pred_, y_actual_[0:n], metric_type="accuracy", )
    print("RandomForest Accuracy:", ct.compute())
    
    # Plot the n-th tree
    rf.plot_tree_id(0, [35,20])
    
    # Plot the average feature importance
    rf.plot_feature_importances_average(12)
    