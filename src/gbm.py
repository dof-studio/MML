# gbm.py
#
# Gradient Boosting Models
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
from threadpool import Mutex, ThreadPool

from copy import deepcopy
from typing import List, Dict, Tuple, Any

from baseml import MLBase, Regression, Classification

from ensemble import Bagging, Boosting
from tree import BaseTree, CART
from tree_wrapper import LRTW
from aggregation import ClassificationAggregation
from aggregation import RegressionAggregation

from metrices import RegressionMetrics
from metrices import BinaryClassificationMetrics
from metrices import MultiClassificationMetrics


# Base Class for Gradient Boosting Models
class BaseGradientBoosting(Bagging, Boosting):
    
    __attr__ = "MML.BaseGradientBossting"
    
    def __init__(self, task: str = "regression", init_method: str = "mean", 
                 *, feature_names: Matrix | Tensor | None = None, **kwargs):
        """
        Initializes a Gradient Boosting model.

        This class represents a base gradient boosting algorithm, which fits multiple weak learners 
        (typically decision trees) on residuals to create a strong predictive model.  It inherits from the 
        'Boosting' base class and provides an optional way to specify feature names for improved interpretability.

        Args:
            task (str): one of {"classification", "regression"}, showing the learning task.
            init_method (str): The method name for creating the initial values. Can be used like an aggregation method.
                In fact, we just reshape the target result and use Aggregation classes to perform the initialization.
                for classification, recommended methods are {"mean", "hard_vote", "soft_vote", ...}
                for regression, recommended methods are {"mean", "median", "trimmed_mean", "quantile", ...}
            feature_names (Matrix | Tensor | None, optional): A Matrix or Tensor object containing strings representing the names of the features used by the model. 
                Defaults to None.  If provided, it should be a one-dimensional structure with a length equal to the number of columns in the data used for training.

        Attributes:
            feature_names (Matrix | Tensor | None): The feature names associated with the model, if provided during initialization.  Inherited from Ensemble.
            initial_values (Matrix | Tensor | None): The initial values for gradient boosting models, is None if not initialized. Can be computed by _compute_initial_values().

        Returns:
            None
        """
        
        # Feature Names should be a 1 dimension Matrix or Tensor object of strings
        # It should be equal to the number of columns of data.
        super().__init__(feature_names = feature_names)
        # Assigned in the base class - Ensemble
        
        # Record task name and initial method
        self.task = task
        self.init_method = init_method
        
        # Initial Values for the first round before fitting any GBTrees.
        # Matrix or Tensor type but left None if not initialized
        self.initial_values = None
    
    @staticmethod
    def _compute_initial_values(real_y: Matrix | Tensor, task: str = "regression", init_method: str = "mean", 
                                *, keepdims: bool = False, **kwargs) -> Matrix | Tensor:
        """
        Computes the initial values for gradient boosting models.
        
        This method calculates the initial predictions used as a starting point in the boosting process.
        It performs data validation and aggregation based on the specified task and initialization method.
        
        Args:
            real_y (Matrix | Tensor): The target variable matrix or tensor to compute initial values from.
            task (str, optional): The learning task ("classification" or "regression"). Defaults to "regression".
            init_method (str, optional): The aggregation method for computing initial values (e.g., "mean", "median"). Defaults to "mean".
            keepdims (bool, optional): Whether to keep the dimensions of the aggregated result. Defaults to False.
            **kwargs: Additional keyword arguments passed to the aggregation class.
        
        Returns:
            Matrix | Tensor: The computed initial values as a Matrix or Tensor.
        
        Raises:
            ValueError: If the input `real_y` is not a Matrix or Tensor, has an invalid shape (not 2D),
                        or contains integer data types instead of floating-point numbers.
        """

        # Type Check (must be an Object type).
        if isinstance(real_y, Object) == False:
            raise ValueError("Input real values must be either Matrix and Tensor. Use Matrix(data) or Tensor(data) to convert.")
        
        # Dimension Check
        if len(real_y.shape) != 2:
            raise ValueError("Input `real_y` must be a tabular data with two dimensions.")
        
        # Dtype Check, must be a float not int
        dtype = real_y.dtype
        if str(dtype) in ("int8", "int16", "int32", "int64"):
            raise ValueError("Input `real_y` must be a floatting type data instead of integers. Consider use .astype() to transform.")
        
        # Shape Detection (it is not 3D, then conduct transformation)
        shape_0 = real_y.shape[0]
        shape_1 = real_y.shape[1]
        stacked = real_y.reshape([shape_0, 1, shape_1])
        
        # Use Aggregation to perform the computation of initial values
        # In, fact, it just takes the mean/median/.. or soft_vote/hard vote 
        # to compute the average starting point in the boosting algorithm
        if task == "classification":
            agg = ClassificationAggregation(stacked, method = init_method, floattype = dtype, **kwargs).compute()
        else:
            agg = RegressionAggregation(stacked, method = init_method, floattype = dtype, **kwargs).compute()
        if keepdims == False:
            agg = agg.reshape([-1])
            
        return agg
    
    @staticmethod
    def _compute_pseudo_residuals(result: Tensor | Matrix, target: Tensor | Matrix, task: str = "regression", loss: str = "mse", **kwargs) -> Matrix | Tensor:  
        """
        Computes the pseudo-residuals (negative gradients scaled by the number of samples).
    
        This method calculates the pseudo-residuals based on the predicted values (`result`), true target values (`target`), 
        task type, and loss function. It leverages specialized metrics classes for regression and classification tasks to compute gradients.
    
        Args:
            result (Tensor | Matrix): The predicted values from the model.
            target (Tensor | Matrix): The true target values.
            task (str, optional): The learning task ("classification" or "regression"). Defaults to "regression".
            loss (str, optional): The loss function used during training. Defaults to "mse".
            **kwargs: Additional keyword arguments passed to the metrics classes.
    
        Returns:
            Matrix | Tensor: The computed pseudo-residuals.
    
        Raises:
            ValueError: If an invalid task is specified (other than "regression" or "classification").
        """
        
        # We don't do type checking here and it is handled internally by Metrics implementation.
        # We don't do loss validity checking since it will be handled by external class initializers.
        
        if task == "regression":
            gradients = RegressionMetrics(result = result, target = target, metric_type = loss, **kwargs).deriv_1(**kwargs)
        elif task == "classification":
            if result.shape[1] == 1:
                raise ValueError("We have detected a single column target in doing classification. Do you use a wrong task? Or convert your binary/multiclass into one_hot.")
            else:
                # raw scores F gradients before the softmax
                gradients = result - target
        else:
            raise ValueError("Invalid task. Task can only be either `regression` or `classification`.")
        
        # pseudo-residuals = negative gradients * n
        if task == "classification":
            return -gradients
        else:
            return -gradients * np.array(result.shape).prod()

    @staticmethod
    def _compute_pseudo_residuals_d1_and_d2(result: Tensor | Matrix, target: Tensor | Matrix, task: str = "regression", loss: str = "mse", **kwargs) -> tuple:  
        """
        Computes the pseudo-residuals (negative gradients scaled by the number of samples), and the grads and hessian matrix together.
    
        This method calculates the pseudo-residuals, gradients, and hessian, based on the predicted values (`result`), true target values (`target`), 
        task type, and loss function. It leverages specialized metrics classes for regression and classification tasks to compute gradients.
    
        Args:
            result (Tensor | Matrix): The predicted values from the model.
            target (Tensor | Matrix): The true target values.
            task (str, optional): The learning task ("classification" or "regression"). Defaults to "regression".
            loss (str, optional): The loss function used during training. Defaults to "mse".
            **kwargs: Additional keyword arguments passed to the metrics classes.
    
        Returns:
            Tuple of [Matrix | Tensor]: The computed pseudo-residuals, grads, hessian.
    
        Raises:
            ValueError: If an invalid task is specified (other than "regression" or "classification").
        """
        
        # We don't do type checking here and it is handled internally by Metrics implementation.
        # We don't do loss validity checking since it will be handled by external class initializers.
        
        if task == "regression":
            metric = RegressionMetrics(result = result, target = target, metric_type = loss, **kwargs)
            gradients = metric.deriv_1(**kwargs)
            hessian = metric.deriv_2(**kwargs)
        elif task == "classification":
            if result.shape[1] == 1:
                raise ValueError("We have detected a single column target in doing classification. Do you use a wrong task? Or convert your binary/multiclass into one_hot.")
            else:
                # raw scores F gradients before the softmax
                gradients = result - target
                hessian = result * (1.0 - result)
                # Formula:
                # grad = P - true_onehot
                # hess = P * (1 - P)
        else:
            raise ValueError("Invalid task. Task can only be either `regression` or `classification`.")
        
        # pseudo-residuals = negative gradients * n
        if task == "classification":
            resid = -gradients
        else:
            resid = -gradients * np.array(result.shape).prod()
        return resid, gradients, hessian

    @staticmethod
    def _softmax_and_standardize(raw: Matrix | Tensor, **kwargs):
        """
        Performs a row-wise softmax operation and standardizes the input matrix.
    
        Args:
            raw: Matrix | Tensor, The input matrix or tensor, which is raw scores 
                 in multi-classification before applied to softmax.
    
        Returns:
            Matrix: A new matrix containing the softmaxed and standardized values of the original data,
                    which is a one-hot like probability matrix.
    
        """
        # Row‐wise softmax of F, shape (n_samples, n_classes)
        raw_max = raw.max(axis = 1).reshape([-1, 1])
        raw_exp = (raw - raw_max).exp()
        # Standardize it into probability measure
        return raw_exp / raw_exp.sum(axis=1).reshape([-1, 1])
    
    @staticmethod
    def _log_prior_transform_one_hot(init_val_y: Matrix | Tensor, task: str = "regression", eps: float = 1e-16, **kwargs):
        """
        Transforms one-hot encoded or probability matrix data into log probabilities for classification tasks.
        
        This function is specifically designed to handle input data that represents either one-hot encoding or probability distributions 
        for classification problems. It converts these values to their natural logarithm (log) representation, which can be beneficial 
        for numerical stability and optimization in certain machine learning algorithms.  It only performs the transformation if the task
        is not regression.
        
        Args:
            init_val_y (Matrix | Tensor): The input data, expected to be a Matrix or Tensor representing one-hot encoded values or probabilities.
            task (str, optional): A string indicating the type of machine learning task. Defaults to "regression".  The transformation is only applied if the task is "classification".
            eps (float, optional): A small constant used for clipping the input values before taking the logarithm. This prevents errors due to log(0). Defaults to 1e-16.
            **kwargs: Additional keyword arguments that are not used in this function.
        
        Returns:
            Matrix | Tensor: The transformed data with log probabilities, or the original data if the task is regression.
        """
        
        # Only conduct the transformation if the task is doing regression
        # We only accept one-hot and probability matrix data in classification tasks
        if task != "classification":
            return init_val_y
        else:
            return init_val_y.clip(eps).log()
    
    @staticmethod
    def _softmax_post_transform_one_hot(raw_score_y: Matrix | Tensor, task: str = "regression", **kwargs):
        """
        Applies a post-softmax transformation and standardization to the raw scores if the task is classification.
        
        This method conditionally applies a softmax function followed by standardization to the raw score predictions 
        if the learning task is classification.  For regression tasks, it returns the input unchanged.
        
        Args:
            raw_score_y (Matrix | Tensor): The raw score predictions from the model.
            task (str, optional): The learning task ("classification" or "regression"). Defaults to "regression".
            **kwargs: Additional keyword arguments passed to the softmax and standardization function.
        
        Returns:
            Matrix | Tensor: The transformed raw scores if the task is classification; otherwise, the original input.
        """
        
        # Only conduct the transformation if the task is doing regression
        # We only accept one-hot and probability matrix data in classification tasks
        if task != "classification":
            return raw_score_y
        else:
            return BaseGradientBossting._softmax_and_standardize(raw_score_y, **kwargs)
        
    def __repr__(self):
        return "BaseGradientBossting(Abstract Class)."
    
    
# Implementation for Greadient Boosting Model
class GradientBoostingModel(BaseGradientBoosting):
    
    __attr__ = "MML.GradientBoostingModel"
    
    def __init__(self, task: str = "classification",
                       loss: str | None = None,
                       init_method: str = "mean",
                       *,
                       tree_type: type = CART,
                       n_estimators: int = 10,
                       max_features: str | int | float | None = 0.8,
                       bootstrap_ratio: float = 0.8,
                       replace: bool = True,
                       shuffle: bool = True,
                       eps: float = 1e-16,
                       tol: float = 1e-8,
                       lr: float = 1e-1,      
                       step_size: str = "newton",
                       l2_lambda: float = 1e-6,
                       floattype: type = float,
                       random_state: int | None = None,
                       feature_names: Matrix | Tensor | None = None,
                       tree_kwargs: dict = {},
                       agg_kwargs: dict = {},
                       **kwargs) -> None:
        """
        Initialize a Gradient Boosting Model using boosting methods on weak learners.
        
        Parameters
        ----------
        task: str, one of {"classification", "regression"}, showing the learning task.
        loss: str, the name of the loss function of the Gradient Model, can be a str or left None.
              If left None, we will adjust it to "mse" if regression, "logloss" if classification.
             (not the loss for the tree, if you intend to set the loss for trees, set them in `tree_kwargs`)
        init_method: str, the name of initial method (actually using aggregation method). See aggregation.py.
                    for classification, common ones are: "mean", "hard_vote", "soft_vote", ...
                    for regression, common ones are: "mean", "median", "percentile", "weighted", ...
        Optional:
            tree_type: type, showing the type of trees you intend to use. Default, CART (Classification and Regression Tree). 
                       You may pass in some other tree-compatible classes, like LRTW.
            n_estimators : int, indicating the maximum number of trees.
            max_features : {None, "sqrt", "log2", "over3"} | int | float, number of columns available to each tree.
            bootstrap_ratio : float, fraction of samples drawn per bootstrap sample, [0, 1].
            replace, shuffle : bool, if to shuffle the bootstrapped samples, passed to `_sample_bootstrapping`.
            eps: float, a very small amount to avoid dividing by 0.
            tol: float, tolerence for convergence when optimizing.
            lr: float, learning rate. Default 1e-2.
            step_size: str, name of methods to step_size calculation, can be {
                       "constant" (to use 1), 
                       "newton" (Newton's method to find the step_size), 
                       "xgboost" (XGBoost-like leaf-wise Newton's method (Not Implemented Yet)),
                    or "numerical" (to numerically search (Not Implemented Yet))}
            l2_lambda: float, the l2 regularization parameter to avoid overfitting. Default 1e-6.
            floattype : type, numerical precision stored in internal matrices.
            random_state : int | None, global seed for reproducibility.
            feature_names : Matrix | Tensor | None, optional feature labels, in Object type. May be useful in feature importance.
            tree_kwargs : dict, extra hyperparameters forwarded to every tree model.
            agg_kwargs : dict, extra hyperparameters forwarded to the aggregation instance (in initialization).
            **kwargs: other key word arguments, reserved for compatibility use.
        """
        
        super().__init__(task=task, init_method=init_method, feature_names=feature_names, **kwargs)

        # Loss Function and init method
        if loss is None:
            self.loss = "mse" if task == "regression" else "logloss"
        else:
            self.loss = loss
        self.init_method = init_method
        
        # Gradient Related Variables (lr/tol)
        self.lr = lr     # Learning rate for gradient boosting
        self.tol = tol   # Tolerance for grad and line search
        self.eps = eps   # When used in newton and xgboost mode to prevent g/h = 0
        
        # Gradient Boosting Data ROW/COL settings
        self.max_features = max_features
        if  max_features is None:
            max_features = 1.0
        self.bootstrap_ratio = float(bootstrap_ratio)
        if self.bootstrap_ratio > 1.0 or self.bootstrap_ratio <= 0.0:
            raise ValueError("Extrapulate for bootstrap ratio is NOT allowed. You must set a value in (0, 1]")
            
        # Bootstrap and Feature Selecting arguments
        self.replace = replace
        self.shuffle = shuffle
        
        # Gradient Descent regularization
        if step_size.lower() not in ("constant", "newton", "xgboost", "numerical"):
            raise ValueError(f"Parameter `step_size` must be on of ('constant', 'newton', 'xgboost', 'numerical') but you have f{step_size}.")
        if step_size.lower() in ("xgboost", "numerical"):
            raise NotImplementedError((f"Step size method {step_size.lower()} is not implemented."))
        if step_size.lower() == "xgboost":
            # We should make sure that the tree instance has `apply` attribute
            try:
                _ = tree_type().apply
            except AttributeError as e:
                raise ValueError(f"When `step_size` set to `xgboost`, you have to use a base tree with method `apply` that returns the index of the leaf node a sample belongs to. But your `tree_type` does not have this.")
        self.step_size = step_size.lower()    # The way step size is calculated
        self.l2_lambda = l2_lambda            # L2 lambda when calculating the step size
        
        # Gradient Boosting Key arguments
        self.n_estimators = int(n_estimators) # Specified Number of Estimators
        self.n_estimators_used = 0            # Trainned Number of Estimators
        # Note, early stopping may prevent using all of the estimators
        
        # Global random_state (Use _random_state_next() to move it forward)
        self.random_state = random_state

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
                                     # Value: bootstrapped data tuple (see. ensemble.py)
        
        # Runtime Containers (Gradient Booster Ensemble).
        self._estimators = {}        # Key: tree id, starting from 0
                                     # Value: tree instance
    
        # Runtime Containers (Step Sizes).
        self._step_sizes = {}        # Key: tree id, starting from 0
                                     # Value: step_size, in Matrix or Tensor, a scalar per output or same to y shape

        # Runtime Containers (Incremental Updates by each tree).
        self._inc_updates= {}        # Key: tree id, starting from 0
                                     # Value: prediction by each tree using the training X

    def fit(self, X: Matrix | Tensor, y: Matrix | Tensor,
            *,
            one_hot: bool = True,
            verbosity: int | None = None,
            evalset: Dict[str, Tuple[Matrix | Tensor, Matrix | Tensor]] | None = None,
            evalmetrics: List[str] | str | None = None,
            early_stop: int | None = None,
            early_stop_logic: str = "some",
            continue_to_train: bool | None = None,
            **kwargs):
        """
        Train n_estimators gradient boosting trees sequentially.
        
        Evaluation Remark:
            ----------
            You may want to evaluate datasets while training. If so, please do the following things:
                1. set `verbosity` = 1 to print the evaluation
                2. set the `evalset` to a dict of tuples of your dataset that is going to be evaluated
                3. set the `evalmetrics` either to a string of metrics or a list of strings
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
                    
        # Special logic Check: If classification, then must be one_hot.
        if self.task == "classification" and y.shape[1] == 1:
            raise ValueError("Input target `y` only have 1 column while you are using `classification` mode. Please notice Gradient Boosting only support one-hot coding. Consider transforming your target into a one-hot two column dataset.")
            
        # Stopping Logic Check.
        if early_stop_logic not in ("any", "some", "most", "all"):
            raise ValueError("Stopping logic `early_stop_logic` must be one of ('any', 'some', 'most', 'all')")
            
        # Prepare the bootstrapped sampeles and initial values
        self._fit_prep(X = X, y = y, continue_to_train = continue_to_train)
        n_samples, n_features = X.shape
        
        # Expanded Initial Values
        expanded_init_vals = self.initial_values.repeat(n_samples, axis = 0)
        y_pred = expanded_init_vals.copy()
        # y_pred is the current prediction for ALL samples (not subsamples)
        
        # Special evalmetrics type conversion
        if isinstance(evalmetrics, str) == True:
            evalmetrics = [evalmetrics]
            
        # Verbosity Conversion
        verbosity = verbosity if verbosity is not None else 0
        
        round_ = 0
        tree_id = 0
        undecreased_no = 0
        last_eval_dict = {} # Please use deepcopy() here to avoid being errorly referred
        
        # Continue to train? Restore the last point
        if continue_to_train is not None:
            
            # In boosting, it is NOT simply to train new trees but to 
            # continue to train on the unexplained residuals.
            # We should first restore the prediction by adding predictions of the 
            # previous trees up.
            if continue_to_train == True and self.n_estimators_used > 0:
                # Copy the tree_id
                tree_id = self.n_estimators_used
                
                # Expand the y_pred by adding things up with
                # += lr * step_size * updates
                for i in range(tree_id):
                    # Note, this step can be replaced by calling one predict
                    # on the training data by previous trees.
                    y_pred += self.lr * self._step_sizes[i] * self._inc_updates[i]
        
        # Helper: Print and decide the evaluated results
        def _decide_stop_with_print(batch: int, undecreased_no: int, eval_dict: dict, last_eval_dict: dict, **kwargs):
            """
            Compare the metics and decide if stop or not.

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
            for evalset_name in eval_dict.keys():
                eval_result = eval_dict[evalset_name]
                if verbosity >= 1:
                    print("Evalset: [", evalset_name, " : Metrics {", end = " ", sep = "")
                for metric_name in eval_result.keys():
                    metric_value = eval_result[metric_name]
                    diff_dict[evalset_name][metric_name] = metric_value - last_eval_dict[evalset_name][metric_name]
                    if diff_dict[evalset_name][metric_name].to_list() > 0:
                        undes_count += 1
                    allmetric_count += 1
                    if verbosity >= 1:
                        print(metric_name, ":", round(metric_value.to_list(), 4), ", ", end = " ", sep = "")
                if verbosity >= 1:
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
                    else:
                        return undecreased_no, False
            elif early_stop_logic == "some":
                if undes_count * 3 >= allmetric_count:
                    undecreased_no += 1
                    if undecreased_no >= early_stop:
                        return undecreased_no, True
                    else:
                        return undecreased_no, False
            elif early_stop_logic == "most":
                if undes_count * 2 >= allmetric_count:
                    undecreased_no += 1
                    if undecreased_no >= early_stop:
                        return undecreased_no, True
                    else:
                        return undecreased_no, False
            elif early_stop_logic == "all":
                if undes_count * 1 >= allmetric_count:
                    undecreased_no += 1
                    if undecreased_no >= early_stop:
                        return undecreased_no, True
                    else:
                        return undecreased_no, False
                    
            # If survives here, return 0, False to refresh the undecreased_no
            return 0, False
                
        # Helper: Train one round pipeline
        def _train_one_round_pipeline(X: Matrix | Tensor, y: Matrix | Tensor, y_current_predicted: Matrix | Tensor, tree_id: int, **kwargs):
            """
            Train for one round and return the new prediction and updates.
            
            Returns
                -------
                Tuple of (Matrix | Tensor, Matrix | Tensor):
                    Matrix | Tensor, new y_pred after applied updates
                    Matrix | Tensor, the new updates

            """
            # y_pred create a copy to avoid modification issues
            y_pred = y_current_predicted.copy()
            
            ###################################################################
            #
            # Preparion
            # Collect the data used to train in this round
            X_selected, y_selected, row_indices = self._bootstrapped[tree_id]
            feature_idx = self._feature_sets[tree_id]
            
            # If classification, then perform softmax on the y_pred to calculate residuals
            probabilities = self._softmax_post_transform_one_hot(y_pred, task = self.task, **kwargs)

            # Calculate pseudo residuls and leave gradients, hessian None
            if self.step_size in ("constant", "numerical"):
                residuals = self._compute_pseudo_residuals(result = probabilities, target = y,
                                task = self.task, loss = self.loss, **kwargs)
                gradients, hessian = None, None # A sugar to achieve code simplicity
            
            # Calculate pseudo residuls as -gradient/hessian
            elif self.step_size in ("newton", "xgboost"):
                # First compute the elements
                residuals, gradients, hessian = self._compute_pseudo_residuals_d1_and_d2(result = probabilities, target = y,
                                task = self.task, loss = self.loss, **kwargs)
                # Then, compute the -g/h residuals
                residuals = - gradients / (hessian + self.l2_lambda + self.eps)
            
            ###################################################################
            #
            # Train
            # Sequentially train the trees on the psuedo residuals (only use a subsample)
            self._train_one_gbtree(X = X_selected, 
                                   y_resid = residuals[row_indices.to_numpy_array()], 
                                   weights = None if hessian is None else hessian[row_indices.to_numpy_array()], 
                                   tree_id = tree_id, one_hot = one_hot, feature_idx = feature_idx, **kwargs)
            
            # !!!
            # Note: We should use the entire data when calculating the _backward_step_size and updates
            # Adjust the step_size for this tree and return the tuple (lr, updates, step_size)
            lr, updates, step_size_i = self._backward_step_size(X = X, y_pred = y_pred, tree_id = tree_id, 
                    one_hot = one_hot, gradients = gradients, hessian = hessian, **kwargs)
            
            # Update the y_preds based on formula += lr * step_size_i * updates
            y_pred += lr * step_size_i * updates
            
            return y_pred, updates
            
        #######################################################################
        #        
        # Build up the gradient boosting trees sequentially
        while tree_id < self.n_estimators:
            
            # Verbosity
            if verbosity >= 1:
                print(f"Training on Round: {round_}, starting from tree: {tree_id} trainned on the residuals...")
            
            ###################################################################
            #
            # Train one pipeline
            y_pred, updates = _train_one_round_pipeline(X, y, y_pred, tree_id = tree_id, **kwargs)
            
            # Finished trainning and add the tree_id
            # You must do it here to avoid self-increased before calculating the step size
            tree_id += 1
                    
            ###################################################################
            #
            # Gradient Exition
            # If the total updates are too small, then exit the training process
            if updates.abs().sum().to_list() < self.tol:
                break
            
            ###################################################################
            #
            # Evaluation
            # Evaluate and decide if stop training from now
            eval_dict = self._eval_one_batch(evalset = evalset, evalmetrics = evalmetrics, one_hot = one_hot, **kwargs)
            
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
                
    def _fit_prep(self, X: Matrix | Tensor, y: Matrix | Tensor, continue_to_train: bool = False, **kwargs):
        """
        Prepare datasets and calculate initial values (for regression tasks and classification tasks).

        Parameters:
            ----------
            X: Matrix | Tensor, the feature matrix (each row is a sample).
            y: Matrix | Tensor, the target values (for regression, numerical; for classification, one-hot or multi-label).
            continue_to_train: bool, if False then not refresh the data, else only conduct type checking.
        
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
        
        # If newly added data, do the full initialization
        if continue_to_train == False or continue_to_train is None:
        
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
            self._step_sizes = {}
            self._inc_updates = {}
            self.n_estimators_used = 0
            
            # Calculate the initial value
            self.initial_values = self._compute_initial_values(real_y = y, 
                            task = self.task, init_method = self.init_method, keepdims = True, **kwargs)
            self.initial_values = self._log_prior_transform_one_hot(self.initial_values, task = self.task, **kwargs)
            
        return self

    def _resolve_max_features(self, n_features_all: int, q: int | float | str | None, **kwargs) -> int:
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

    def _train_one_gbtree(self, X: Matrix | Tensor, y_resid: Matrix | Tensor, weights: Matrix | Tensor | None, tree_id: int, one_hot: bool, feature_idx: np.ndarray, *, _mutex : Mutex | None = None, **kwargs) -> None:
        """
        Train one gradient boosting tree based on the split data.

        Parameters
        ----------
            X: Matrix | Tensor, the feature matrix (each row is a sample).
            y_resid: Matrix | Tensor, the target values (peuedo residuals, NOT the actual label).
            weights: Matrix | Tensor | None, the weights used to split the tree or to create the leaf nodes.
            tree_id : int, the tree_id you expect to train.
                      The tree must NOT be initialied, else error.
            one_hot: bool, if classification and using one-hot coding.
            feature_idx: np.ndarray, the indices of used features in training. 

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
            
        # Process the tree kwargs. You should NOT specify the task/loss, if so, then ValueError
        if self.tree_kwargs is not None:
            if "task" in self.tree_kwargs:
                raise ValueError(f"You should never specify any `task` in the tree kwargs since the mode must be automatically decided by the gradient boosting tree. You have {self.tree_kwargs['task']} now.")
            if "loss" in self.tree_kwargs:
                raise ValueError(f"You should never specify any `loss` in the tree kwargs since the mode must be automatically decided by the gradient boosting tree. You have {self.tree_kwargs['loss']} now.")
                
        # Train one gradient boost tree using MSE or WMSE
        if weights is not None:
            tree = self.tree_type(task = "regression", tree_id = tree_id, loss = "wmse",
                                  random_state = self._random_state_next(), floattype = self.floattype, feature_names = self.feature_names, **self.tree_kwargs)
        else:
            tree = self.tree_type(task = "regression", tree_id = tree_id, loss = "mse",
                                  random_state = self._random_state_next(), floattype = self.floattype, feature_names = self.feature_names, **self.tree_kwargs)
        tree.fit(X, y_resid, one_hot = one_hot, use_features_idx = feature_idx, weights = weights, **kwargs)

        # Apply lock to protect the resources
        if _mutex is not None:
            with _mutex:
                self._estimators[tree_id] = tree
                self.n_estimators_used += 1
            return
        else:
            self._estimators[tree_id] = tree
            self.n_estimators_used += 1

        return
    
    def _backward_step_size(self, X: Matrix | Tensor, y_pred: Matrix | Tensor, tree_id: int, one_hot: bool,
                             gradients: Matrix | Tensor | None = None, hessian: Matrix | Tensor | None = None, *, _mutex : Mutex | None = None, **kwargs) -> tuple:
        """
        Calculate the step size and compute the addition values on the backward step.

        Parameters
        ----------
            X: Matrix | Tensor, the feature matrix (each row is a sample).
            y_pred: Matrix | Tensor, the accumulated, unupdated predictions before training this tree.
            tree_id: int, the tree_id you expect to train.
                     The tree must NOT be initialied, else error.
            one_hot: bool, if classification and using one-hot coding.
            gradients: Matrix | Tensor | None, the gradient matrix calculated, can be None in `constant` mode.
            hessian: Matrix | Tensor | None, the hessian matrix (hessian vector without cross terms) calculated, can be None in `constant` mode.

        Returns
            -------
            Tuple[lr, updates, step_size] calculated.
                lr is scalar,
                updates is Matrix | Tensor with y shape,
                step_size is Matrix | Tensor processed into y shape or a scalar.

        """
        # Tree id validity check
        if tree_id > self.n_estimators:
            raise ValueError(f"You are training tree_id {tree_id}, but you only have {self.n_estimators} reserved.")
        
        # If the tree is ont trained
        if self._estimators.get(tree_id, None) is None:
            raise ValueError(f"You are trying to create the updates and step size on tree_id {tree_id}, but the tree has NOT been trained. First train the tree.")
        tree = self._estimators[tree_id]
        
        # If it is updated and calculated
        if self._inc_updates.get(tree_id, None) is not None:
            raise ValueError(f"You are trying to create the updates and step size on tree_id {tree_id}, but they are existed.")
        if self._step_sizes.get(tree_id, None) is not None:
            raise ValueError(f"You are trying to create the updates and step size on tree_id {tree_id}, but they are existed.")
            
        # Check the status of gradients and hessian
        # They should be given in non-constant mode and should be None in constant mode
        if self.step_size in ("constant", "numerical"):
            if gradients is not None or hessian is not None:
                raise ValueError(f"In creating the updates and calculating the step size for tree_id {tree_id}, you have constant step_size but given a gradient/hessian matrix.")
        elif self.step_size in ("newton", "xgboost"):
            if gradients is None or hessian is None:
                raise ValueError(f"In creating the updates and calculating the step size for tree_id {tree_id}, you have {self.step_size} method to estimate the step_size but left gradient/hessian as Nonetype.")
                
        # First, predict the updates on the new train
        updates = tree.predict(X, **kwargs)
        
        # These variables are to be returned
        lr = self.lr
        round_updates = updates
        round_step_size = updates.ones([1], backend = updates._backend, dtype = updates.dtype).to(backend = updates._backend, dtype = updates.dtype, device = updates.device)
        # shape (1) Scalar
        
        # If constant mode, use the default arguments
        if self.step_size == "constant":
            pass
        
        # Numerical is NOT implemented
        elif self.step_size == "numerical":
            raise NotImplementedError("Step size updating method `numerical` is NOT implemented. Consider using `newton` or `xgboost` instead.")
            
        # If Newton's method, implement a output-wise step size and broadcast it to sample wise
        elif self.step_size == "newton":
            # In newton's method, we have already fit the tree on -g/h, so let step_size = 1
            pass
            
        # If XGBoost method, implement a leaf-wise step size and broadcast it to sample wise
        elif self.step_size == "xgboost":
            # In xgboost method, we conduct a leaf-wise step size calculation and let step_size = 1 * step_size
            pass
            # Not Completely implemented yet
            # 1. Needs to store the lwgammas and search and match the leaf zones in predicting
            # 2. set the scaling gamma = 1
            
            # XGBoost method to update the leaf-wise step_size for all samples (n, d)
            def _xgboost_backward(g, h, updates, l2, axis):
                # Calculate the node indices and representations
                # The indices is a 1 column 2 dimensional [-1, 1] Matrix | Tensor
                node_indices, node_repr = tree.apply(X, **kwargs)
                
                # Transform the node indices into a 1d array
                node_indices = node_indices.flatten()
                
                # Convert the indices to a 1 dimensional unique Container
                node_indices_unique = node_indices.unique()
                
                # Leaf-wise Step_size (gamma)
                lw_gammas = {}
                for leaf in node_indices_unique:
                    mask = (node_indices == leaf) 
                    if axis is not None:
                        g_sum = g[mask].sum(axis=axis)       # shape (d,)
                        h_sum = h[mask].sum(axis=axis)       # shape (d,)
                    else:
                        g_sum = g[mask].sum().reshape([1]).repeat(g.shape[0], axis=0).reshape([1, -1])
                        h_sum = h[mask].sum().reshape([1]).repeat(g.shape[0], axis=0).reshape([1, -1])
                    lw_gammas[leaf] = - g_sum / (h_sum + l2) # shape (d,)
                    
                # Vstack to create a full ranged step_size
                gamma_mat = lw_gammas[node_indices[0]]
                gamma_mat = gamma_mat.vstack(*[lw_gammas[l] for l in node_indices[1:]])
                # After this # shape (n, d)
                return lw_gammas, gamma_mat
        
            # Calculate xgboost backward
            if self.task == "classification":
                lw_gammas, round_step_size = _xgboost_backward(gradients, hessian, updates, l2 = self.l2_lambda, axis = None)
            else:
                lw_gammas, round_step_size = _xgboost_backward(gradients, hessian, updates, l2 = self.l2_lambda, axis = 0)

        # Unknown step_size method
        else:
            raise ValueError("Unknown `self.step_size`. Make sure you choose one valid step_size name. Refer to __init__ for more information.")
                    
        # Add the updates (inc) and step_size to class members
        if _mutex is not None:
            with _mutex:
                self._inc_updates[tree_id] = round_updates.copy()
                self._step_sizes[tree_id] = round_step_size.copy()
            return
        else:
            self._inc_updates[tree_id] = round_updates.copy()
            self._step_sizes[tree_id] = round_step_size.copy()
            
        return lr, round_updates, round_step_size
            
    def _eval_one_batch(self, evalset: Dict[str, Tuple[Matrix | Tensor, Matrix | Tensor]] | None = None, evalmetrics: List[str] | str | None = None, one_hot: bool = True, **kwargs):
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
        if evalmetrics is not None and evalset is not None:
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
                            if (y_pred.shape[1] == 2 and one_hot == False) or y_pred.shape[1] == 1:
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

    def predict(self, X: Matrix | Tensor, **kwargs) -> Matrix | Tensor:
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
                
        # Get the initial prediction over samples
        y_pred = self.initial_values.repeat(X.shape[0], axis = 0)

        # Iteratively predict
        for tree_id in self._estimators.keys():
            # Retrieve the estimator (tree)
            estimator = self._estimators[tree_id]
            
            # We here need to use different ways to treat the step_size:
            # If "constant", "newton" type, it is a [1, -1] size, we need to repeat on x axis
            if self.step_size in ("constant", "newton", "numerical"):
                step_size = self._step_sizes[tree_id].copy()
            elif self.step_size in ("xgboost"):
                pass # Need to implement
            
            y_pred += self.lr * step_size * estimator.predict(X)
        
        # If uses classification, apply softmax (automatically handled)
        return self._softmax_post_transform_one_hot(y_pred, task = self.task, **kwargs)
    
    def _is_fitted(self) -> bool:
        """
        Check if the Gradient Boosting Model has been fitted or not.

        Returns
            -------
            bool, if the model is fitted or not.

        """
        return True if len(self._estimators) > 0 else False

    def update_tree_kwargs(self, tree_kwargs : dict | None = None, **kwargs) -> None:
        """
        Update the tree-build key word arguments in whole.
        If you hope to replace or remove, please first get and then set by this.

        Returns
            -------
            None

        """
        if tree_kwargs is not None:
            # Check the new arguments
            if "task" in tree_kwargs:
                raise ValueError(f"You should never specify any `task` in the tree kwargs since the mode must be automatically decided by the gradient boosting tree. You have {tree_kwargs['task']} now.")
            if "loss" in tree_kwargs:
                raise ValueError(f"You should never specify any `loss` in the tree kwargs since the mode must be automatically decided by the gradient boosting tree. You have {tree_kwargs['loss']} now.")
                
            # Assign if passed
            self.tree_kwargs = tree_kwargs
            
    def update_agg_kwargs(self, agg_kwargs: dict | None = None, **kwargs) -> None:
        """
        Update the aggregation related key word arguments in whole.
        If you hope to replace or remove, please first get and then set by this.

        Returns
            -------
            None

        """
        if agg_kwargs is not None:
            self.agg_kwargs = agg_kwargs

    def __repr__(self):
        return f"GradientBoostingModel(task = {self.task}, loss = {self.loss}, tree_type = {str(self.tree_type)}, n_estimators = {self.n_estimators})."
    
               
# Alias for Gradient Boosting Model
GBM = GradientBoostingModel


# Multiple Classification Task
def test_multiclassification():
    
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import accuracy_score
    from metrices import MultiClassificationMetrics as mcm
    import pandas as pd
    
    ###########################################################################
    #
    # Generate the data
    def generate_binary_classification_data(n_samples=10000, n_features=20, n_informative=15, n_redundant=0, random_state=None):
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
    # Use sklearn
    clf = GradientBoostingClassifier(n_estimators = 50, max_depth = 8, random_state = None)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    # Report accuracy
    print(f"Test set accuracy: {accuracy_score(y_test, y_pred):.4f}")
    
    ###########################################################################
    #
    # Construct my Gradient Boosting Model
    frontend = Matrix
    backend = "numpy"
    device = "cpu"
    tree_kwargs = {"max_depth":8, 
                   "min_samples_split":0.0004, 
                   "prune":True, "prune_alpha":0.0002, 
                   "grid_accelerator":20, "grid_use_percentile":False,
                   "grid_point_variant":None}
    
    # Conduct Type Conversion
    X_train_x = frontend(X_train.to_numpy(), backend, device=device)
    re_train_y = frontend(y_train.to_numpy(), backend, device=device).reshape([-1, 1])
    X_test_x = frontend(X_test.to_numpy(), backend, device=device)
    y_actual_ = frontend(y_test.to_numpy(), backend, device=device).reshape([-1, 1])
    
    re_train_y = GBM._to_onehot(re_train_y, 3)
    y_actual_  = GBM._to_onehot(y_actual_ , 3)
    
    # Initialize a Gradient Boosting Model
    gb = GBM(task = "classification", init_method = "soft_vote",
             n_estimators = 100,
             lr = 0.15,
             step_size = "newton",
             random_state = None,
             tree_kwargs = tree_kwargs,
             agg_kwargs = {})
    
    # Fit the tree
    gb.fit(X_train_x, 
           re_train_y, 
           one_hot = True,
           verbosity = 1,
           evalset = {"train": (X_train_x, re_train_y), "test": (X_test_x, y_actual_)},
           evalmetrics = ["accuracy", "logloss"],
           continue_to_train=False)
    
    # Predict the labels
    n = 9999
    y_pred_ = gb.predict(X_test_x[0:n])
    
    # Evaluate the test set
    ct = MultiClassificationMetrics(y_pred_, y_actual_[0:n], metric_type="accuracy", )
    print("Gradient Boosting Accuracy:", ct.compute())
    
    # Plot the n-th tree
    gb.plot_tree_id(1, [35,20])
    
    # Plot the average feature importance
    gb.plot_feature_importances_average(12)


# Single Value Regression Task
def test_singleregression():
    
    import pandas as pd
    import numpy as np
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.metrics import mean_squared_error
    
    ###########################################################################
    #
    # Generate the data
    def generate_regression_data(n_samples=10000, n_features=20, n_informative=15, noise=0.1, random_state=None):
        X, y = make_regression(n_samples=n_samples,
                                n_features=n_features,
                                n_informative=n_informative,
                                noise=noise,
                                random_state=random_state)
        return pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])]), pd.Series(y, name='target')
    
    # Data
    X, y = generate_regression_data()
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
    
    ###########################################################################
    #
    # Reference
    # Use sklearn
    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=8, random_state=1)
    model.fit(X_train, y_train)
    
    # Prediction
    y_pred = model.predict(X_test)
    
    # Evaluation
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    print(f"Mean Squared Error: {rmse:.4f}")
    
    ###########################################################################
    #
    # Construct my Gradient Boosting Model
    frontend = Matrix
    backend = "numpy"
    device = "cpu"
    tree_kwargs = {"max_depth":8, 
                   "min_samples_split":0.0008, 
                   "prune":True, "prune_alpha":0.0002, 
                   "grid_accelerator":10, "grid_use_percentile":False,
                   "grid_point_variant":None}
    
    # Conduct Type Conversion
    X_train_x = frontend(X_train.to_numpy(), backend, device=device)
    re_train_y = frontend(y_train.to_numpy(), backend, device=device).reshape([-1, 1])
    X_test_x = frontend(X_test.to_numpy(), backend, device=device)
    y_actual_ = frontend(y_test.to_numpy(), backend, device=device).reshape([-1, 1])
    
    # Initialize a Gradient Boosting Model
    gb = GBM(task = "regression", init_method = "median",
             n_estimators = 50,
             lr = 0.15,
             step_size = "newton",
             random_state = None,
             tree_kwargs = tree_kwargs,
             agg_kwargs = {})
    
    # Fit the tree
    gb.fit(X_train_x, 
           re_train_y, 
           one_hot = True,
           verbosity = 1,
           evalset = {"train": (X_train_x, re_train_y), "test": (X_test_x, y_actual_)},
           evalmetrics = ["r2", "rmse"],
           continue_to_train=False)
    
    # Predict the labels
    n = 9999
    y_pred_ = gb.predict(X_test_x[0:n])
    
    # Evaluate the test set
    rm = RegressionMetrics(y_pred_, y_actual_[0:n], metric_type="rmse", )
    print("Gradient Boosting RMSE:", rm.compute())
    
    # Plot one tree
    gb.plot_tree_id(0, figsize=(45,20))
    
    # Plot Feature Importance
    gb.plot_feature_importances_average(8)


if __name__ == "__main__":
    
    pass

