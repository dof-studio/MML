# nn_sinterf_evaluator.py
#
# Neural Network Evaluation Pipeline
# From MML Library by Nathmath

import numpy as np
try:
    import torch
except ImportError:
    torch = None

from typing import Any, List, Dict, Tuple, Literal

from copy import deepcopy

from .objtyp import Object
from .tensor import Tensor

from .baseml import Regression, Classification

from .nn_base import nn_Base
from .nn_parameter import nn_Parameter
from .nn_module import nn_Module

from .nn_loss import nn_Loss_BaseLoss
from .nn_optimizer import nn_Optm_BaseOptimizer

from .metrics import RegressionMetrics
from .metrics import BinaryClassificationMetrics
from .metrics import MultiClassificationMetrics


# Neural Network Fast Evaluation Pipeline
class nn_SInterf_Evaluator(nn_Base, Regression, Classification):
    """
    Neural Network Simple Interface - Evaluation pipeline.
    
    This evaluator accepts a neural network module, a ctiterion module (loss function),
    an optimizer instance and conduct controlled automatic training and evaluation job.
    You may use fit(), predict(), or other APIs to experience an easy-to-use and optimized
    neural network training process with full automation loaded.
    """
    
    __attr__ = "MML.nn_SInterf_Evaluator"
    
    def __init__(self, name: str = "Evaluator",
                       task: str = "classification",
                       module: nn_Module | None = None,
                       criterion: nn_Loss_BaseLoss | None = None,
                       optimizer: nn_Optm_BaseOptimizer | None = None,
                       **kwargs) -> None:
        """
        Initialize an easy-interface evaluator pipeline object by passing in modules.
        
        Parameters
        ----------
            task: str, one of {"classification", "regression"}, showing the learning task.
            module: nn_Module | None, you must pass an nn_Module which is the root node of your neural network structure.
            criterion: nn_Loss_BaseLoss, you must pass an instance of a loss function that is the child of the base class.
            optimizer: nn_Optm_BaseOptimizer, you must pass an instance of an optimizer that is the child of the base class.
            Optional:
                **kwargs: other key word arguments, reserved for compatibility use.
            
        Raise
        ----------
            ValueError, if task it not valid.
            TypeError, if any of the parameter is None or does not have the correct type.
        """
        # Task check
        if task not in {"classification", "regression"}:
            raise ValueError(f"In initializing an evaluator, `task` must be either 'classification' or 'regression', but got {task}")

        # Type check (must be the type specified but not None or others)
        if module is not None and not isinstance(module, nn_Module):
            raise TypeError("In initializing an evaluator, `module` must be an instance of nn_Module")
        elif module is None:
            raise TypeError("In initializing an evaluator, `module` must be initialized and cannot be None")
    
        if criterion is not None and not isinstance(criterion, nn_Loss_BaseLoss):
            raise TypeError("In initializing an evaluator, `criterion` must be an instance of nn_Loss_BaseLoss")
        elif criterion is None:
            raise TypeError("In initializing an evaluator, `criterion` must be initialized and cannot be None")
        
        if optimizer is not None and not isinstance(optimizer, nn_Optm_BaseOptimizer):
            raise TypeError("In initializing an evaluator, `optimizer` must be an instance of nn_Optm_BaseOptimizer")
        elif optimizer is None:
            raise TypeError("In initializing an evaluator, `optimizer` must be initialized and cannot be None")
            
        # Call the nn_Base to keep the format consistent
        super().__init__()
        
        # Record name, task, module, criterion, optimizer
        self.name = name
        self.task = task
        self.module = module
        self.criterion = criterion
        self.optimizer = optimizer
        
        # Create a reference of reference_X and reference_y WITHOUT COPY
        self.reference_X = None   # NO COPY
        self.reference_y = None   # ON COPY
        
        # Create a recoder of batch_size
        self.batch_size = None
        
        # Create a counter of how may epoches and steps trainned
        self.n_epoch = 0
        self.n_step = 0
        
        # Create a dictionary to collect training loss and evaluation information (if any)
        self.losses_ = {}    # Stepwise, index: step number
        self.evalhist_ = {}  # Some_epoch-wise, index: epoch number
        
        # Create a record of random state
        self.random_state = None
        
        # Create a record of to_device which means we need to redevice the data before training/testing
        self.to_device = None
        
    def _fit_epoch_prep(self, X: Tensor, y: Tensor,
                        batch_size: int | None = None,
                        shuffle: bool = True, 
                        random_state: int | None = None, 
                        **kwargs) -> Tuple[Tensor, Tensor]:
        """
        Prepare datasets and calculate initial values (for regression tasks and classification tasks).

        Parameters:
            ----------
            X: Tensor, the feature tensor (the 1st dimension is sample).
            y: Tensor, the target values (for regression, numerical; for classification, one-hot or multi-label).
            batch_size: int, the number of samples trained each time. Must be greater than 1. If None, then use all.
            shuffle: bool, whether data will be shuffled for each round (same device same type). By default, True (if batch_size is None then omitted).
            random_seed: int | None, the random seed set to perform shuffle, can be None which means to randomly choose one.

        Returns:
            ----------
            Tuple(X, y) shuffled copy or original reference
        """
        
        # We don't conduct type checks but checks if X or y are None
        if X is None or y is None:
            raise ValueError("In _fit_epoch_prep(), input `X` or `y` is/are None-type.")
        
        # If no need to shuffle, then JUST RETURN without shuffling and copying
        if shuffle == False or batch_size is None:
            return X,y
        elif batch_size >= X.shape[0]:
            return X,y
        
        # Else, we shuffle based on the seed
        if random_state is not None:
            np.random.seed(random_state)
        idx = list(range(X.shape[0]))
        np.random.shuffle(idx)
        
        return X[idx], y[idx]
     
    def _fit_slice_batch(self, X: Tensor, y: Tensor,
                         start: int | None = None,
                         batch_size: int | None = None,
                         to_device: str | None = None,
                         **kwargs) -> Tuple[Tensor, Tensor]:
        """
        Slice the input to create one mini-batch for training/testing.
        
        Parameters:
            ----------
            X: Tensor, the feature tensor (the 1st dimension is sample).
            y: Tensor, the target values (for regression, numerical; for classification, one-hot or multi-label).
            start: int | None, the starting index (begining) to be sliced for this round.
            batch_size: int | None, the number of samples trained each time. Must be greater than 1. If None, then use all.
            to_device: str | None, if non-None, we will perform device transformation after slicing them.
            
        Returns:
            ----------
            Tuple(X, y) sliced copy or original reference.
        """
        
        # We don't conduct type checks but checks if X or y are None
        if X is None or y is None:
            raise ValueError("In _fit_slice_batch(), input `X` or `y` is/are None-type.")
            
        # If no need to slice, then JUST RETURN without shuffling and copying
        if batch_size is None and start is None:
            if to_device is None:
                return X,y
            else:
                return X.to(backend=X._backend, dtype=X.dtype, device=to_device), y.to(backend=y._backend, dtype=y.dtype, device=to_device)
        elif batch_size >= X.shape[0] and start is None:
            if to_device is None:
                return X,y
            else:
                return X.to(backend=X._backend, dtype=X.dtype, device=to_device), y.to(backend=y._backend, dtype=y.dtype, device=to_device)
        
        # Then, we need to slice.
        if start is None:
            raise ValueError("In _fit_slice_batch(), input `start` is None while a small batch_size is specified")
        if batch_size is None:
            raise ValueError("In _fit_slice_batch(), input `batch_size` is None while a start is specified")
        if start >= X.shape[0]:
            raise ValueError(f"In _fit_slice_batch(), input `start` {start} is greater than the number of samples in `X`")
        
        # Slice and return a copy.
        if start + batch_size <= X.shape[0]:
            idx = list(range(start, start + batch_size))
        else:
            idx = list(range(start, X.shape[0]))
        if to_device is None:
            return X[idx],y[idx]
        else:
            return X[idx].to(backend=X._backend, dtype=X.dtype, device=to_device), y[idx].to(backend=y._backend, dtype=y.dtype, device=to_device)
    
    def _fit_slice_batch_X(self, X: Tensor,
                         start: int | None = None,
                         batch_size: int | None = None,
                         to_device: str | None = None,
                         **kwargs) -> Tensor:
        """
        Slice the input to create one mini-batch for training/testing.
        
        Parameters:
            ----------
            X: Tensor, the feature tensor (the 1st dimension is sample).
            y: Tensor, the target values (for regression, numerical; for classification, one-hot or multi-label).
            start: int | None, the starting index (begining) to be sliced for this round.
            batch_size: int | None, the number of samples trained each time. Must be greater than 1. If None, then use all.
            to_device: str | None, if non-None, we will perform device transformation after slicing them.
            
        Returns:
            ----------
            X sliced copy or original reference.
        """
        
        # We don't conduct type checks but checks if X or y are None
        if X is None:
            raise ValueError("In _fit_slice_batch_X(), input `X` is/are None-type.")
            
        # If no need to slice, then JUST RETURN without shuffling and copying
        if batch_size is None and start is None:
            if to_device is None:
                return X
            else:
                return X.to(backend=X._backend, dtype=X.dtype, device=to_device)
        elif batch_size >= X.shape[0] and start is None:
            if to_device is None:
                return X
            else:
                return X.to(backend=X._backend, dtype=X.dtype, device=to_device)
        
        # Then, we need to slice.
        if start is None:
            raise ValueError("In _fit_slice_batch(), input `start` is None while a small batch_size is specified")
        if batch_size is None:
            raise ValueError("In _fit_slice_batch(), input `batch_size` is None while a start is specified")
        if start >= X.shape[0]:
            raise ValueError(f"In _fit_slice_batch(), input `start` {start} is greater than the number of samples in `X`")
        
        # Slice and return a copy.
        if start + batch_size <= X.shape[0]:
            idx = list(range(start, start + batch_size))
        else:
            idx = list(range(start, X.shape[0]))
        if to_device is None:
            return X[idx]
        else:
            return X[idx]    
    
    def _fit_train_one_step(self, X: Tensor, y: Tensor, **kwargs) -> Tuple[int, int, float]:
        """
        Train the model for 1 complete step without switching to evaluation mode.
        
        Parameters:
            ----------
            X: Tensor, the mini-batch feature tensor (the 1st dimension is sample).
            y: Tensor, the mini-batch target values (for regression, numerical; for classification, one-hot or multi-label).
            
        Returns:
            ----------
            Tuple[int, int, float]: (epoch, step, train_loss)

        """
        # We don't conduct type checks but checks if X or y are None
        if X is None or y is None:
            raise ValueError("In _fit_slice_batch(), input `X` or `y` is/are None-type.")
        
        # Module must be in training mode
        if self.module.training == False:
            raise RuntimeError("Called _fit_train_one_step() to perform one step training but the module is in non-training mode.")
        
        # Module must have the same dtype, device with X
        if self.module.dtype != X.dtype or self.module.device != X.device:
            raise RuntimeError("Called _fit_train_one_step() to perform one step training but the module and your data have different dtype/device.")
        
        # Perform a forward pass on the inputs
        out = self.module.forward(X)
        
        # Calculate loss of this step
        loss = self.criterion(out, y)
        
        # Perform the backward propagation of the loss function
        lossgrad = self.criterion.backward()
    
        # Perform the backward propagation of the neural network module
        self.module.backward(lossgrad)
        
        # Apply one step on optimizer to update the parameters
        self.optimizer.step()
        
        # Apply zero grad to clear the gradients
        self.module.zero_grad()
        
        # Increment the step += 1
        self.n_step += 1
        return self.n_epoch, self.n_step, loss.to_list()
        
    def _fit_switch_to_mode(self, mode: Literal["train", "eval"] = "train", **kwargs) -> None:
        """
        Switch the module to train mode or evaluation mode.
        """
        if mode not in {"train", "eval"}:
            raise ValueError(f"In _fit_switch_to_mode(), you gave a mode {mode} which is neither `train` nor `eval`.")
    
        if mode == "train":
            self.module.train()
        elif mode == "eval":
            self.module.eval()
        return
    
    def _eval_one_batch(self, evalset: Dict[str, Tuple[Tensor, Tensor]] | None = None, evalmetrics: List[str] | str | None = None, one_hot: bool = True, **kwargs):
        """
        Evaluate the `evalset` after training for one batch.    

        Returns
            -------
            result_dict : dict  # Key: evalset name
                                # Value dict {metric_name: metric_value}
            or 
            {} if failed or did not evaluate
        """
        # First switch to evaluation mode
        self._fit_switch_to_mode(mode = "eval")
        
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

    def fit(self, 
            X: Tensor, 
            y: Tensor,
            epoches: int = 100,
            batch_size: int | None = None,
            shuffle: bool = True,
            random_state: int | None = None,
            to_device: str | None = None,
            *,
            one_hot: bool = True,
            verbosity: int | None = None,
            evalper: int = 1,
            evalset: Dict[str, Tuple[Tensor, Tensor]] | None = None,
            evalmetrics: List[str] | str | None = None,
            early_stop: int | None = None,
            early_stop_logic: str = "some",
            continue_to_train: bool | None = True,
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
            X: Tensor, the feature tensor (the 1st dimension is sample).
            y: Tensor, the target values (for regression, numerical; for classification, one-hot or multi-label).
            epoches: int, the number of rounds (maximum rounds) to be trainned. Default is 100.
            batch_size: int, the number of samples trained each time. Must be greater than 1. If None, then use all.
            shuffle: bool, whether data will be shuffled for each round (same device same type). By default, True (if batch_size is None then omitted).
            random_seed: int | None, the random seed set to perform shuffle, can be None which means to randomly choose one.
            Optional:
                one_hot : bool, if y is one-hot encoded for classification tasks.
                verbosity: int | None, if >= 1 and having `evalset`, then will report metrics each batch.
                evalper: int, the number of rounds to perform before evaluation conducted again.
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

        # Type Check (must be an Tensor type).
        if isinstance(X, Tensor) == False or isinstance(y, Tensor) == False:
            raise ValueError("Input dataset must be Tensor for neural networks. Use Tensor(data) or Tensor(data) to convert.")
        if type(X) != type(y):
            raise ValueError("Input feature `X` and target `y` must have the same type, use Tensor instead.")
        
        # Dimension Check.
        if len(X.shape) < 2:
            raise ValueError("Input feature `X` must be at least 2 dimensional (the first is for samples).")
        if len(y.shape) == 1:
            raise ValueError("Input target `y` must also be a 2d data. If only one label or value, use data.reshape([-1, 1])")
                    
        # Batch size Check.
        if batch_size is not None:
            if int(batch_size) < 1:
                raise ValueError("Input `batch_size` must be an interger which is greater or equal to 1.")
                        
        # Stopping Logic Check.
        if early_stop_logic not in ("any", "some", "most", "all"):
            raise ValueError("Stopping logic `early_stop_logic` must be one of ('any', 'some', 'most', 'all')")
    
        # Record the original data, random seeds, and to_device
        self.reference_X = X
        self.reference_y = y
        self.random_state = random_state
        self.to_device = to_device
        
        # Record batch size
        self.batch_size = int(batch_size) if batch_size is not None else None
        
        # Special evalmetrics type conversion
        if isinstance(evalmetrics, str) == True:
            evalmetrics = [evalmetrics]
            
        # Verbosity Conversion
        verbosity = verbosity if verbosity is not None else 0
        
        # Create Evaluation Related Objects
        undecreased_no = 0
        last_eval_dict = {} # Please use deepcopy() here to avoid being errorly referred
        
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
                
        #######################################################################
        #        
        # Iteratively train the neural network
        rounds = 0
        while rounds < epoches:
            
            # Verbosity
            if verbosity >= 1:
                print(f"Training on Total Epoch: {self.n_epoch}, Round: {rounds}")

            ###################################################################
            #
            # If needs to shuffle the data, then shuffle it
            epo_X, epo_y = self._fit_epoch_prep(X, y, 
                    batch_size=batch_size, shuffle=shuffle, random_state=self._random_state_next(), **kwargs)
            
            # Calculate number of steps in this round
            if batch_size is None:
                this_steps = 1
            elif batch_size >= epo_X.shape[0]:
                this_steps = 1
            else:
                this_steps = int(np.ceil(epo_X.shape[0] / batch_size))
            
            ###################################################################
            #
            # Formally strat to train this epoch, we first transfer to train mode
            self._fit_switch_to_mode(mode = "train")
            
            # For steps in one epoch, train iteratively
            if this_steps == 1:
                # Just train and get the results
                stp_X, stp_y = self._fit_slice_batch(epo_X, epo_y, start = None, batch_size = None, to_device = to_device, **kwargs)
                _epoch, _step, _loss = self._fit_train_one_step(stp_X, stp_y, **kwargs)
                # Record the step loss
                self.losses_[_step] = _loss
                
            else:
                for step in range(this_steps):
                    # Prepare step-sliced data
                    stp_X, stp_y = self._fit_slice_batch(epo_X, epo_y, start = step * batch_size, 
                         batch_size = batch_size, to_device = to_device, **kwargs)
                    # Train one step
                    _epoch, _step, _loss = self._fit_train_one_step(stp_X, stp_y, **kwargs)
                    # Record the step loss
                    self.losses_[_step] = _loss
                                        
            # Self-increment epoch
            self.n_epoch += 1
            
            ###################################################################
            #
            # Evaluation if needed
            if rounds % evalper != 0 or rounds == 0:
                # Count self increasing
                rounds += 1
                continue
            
            # Evaluate and decide if stop training from now
            eval_dict = self._eval_one_batch(evalset = evalset, evalmetrics = evalmetrics, one_hot = one_hot, **kwargs)
            
            # Try stop maker and receive the advice
            undecreased_no, decision = _decide_stop_with_print(rounds, undecreased_no = undecreased_no, eval_dict = eval_dict, last_eval_dict = last_eval_dict)
            
            # Record the evaluation result
            self.evalhist_[self.n_epoch] = deepcopy(eval_dict)
            
            # Copy last evaluated dict
            last_eval_dict = deepcopy(eval_dict)
            
            # Count self increasing
            rounds += 1
            
            # Make decision to terminate or not
            if decision == True:
                break
            
        return self
            
    def eval(self) -> None:
        """
        Switch the module to evaluation mode.
        """
        self.module.eval()
        return
    
    def predict(self, X: Tensor, **kwargs) -> Tensor:
        """
        Predict target values for samples in X in batches.
        
        Returns:
            Tensor, output of predictions.
            
        Raises:
            RuntimeError: if you did NOT switched to evalation mode.
        """
        # Check the module training status
        if self.module.training == True:
            raise RuntimeError("In predict(), you called this in training mode. Please call .eval() to make the model safe in evaluation mode.")
        
        # Type Check (must be an Tensor type).
        if isinstance(X, Tensor) == False:
            raise ValueError("Input dataset must be a Tensor. Use Tensor(data) to convert.")
        
        # If does not need mini-batch, directly go with X to deviced
        if self.batch_size is None:
            epo_X = X if self.to_device is None else X.to(X._backend, dtype=X.dtype, device=self.to_device)
            return self.module.forward(epo_X)
        elif self.batch_size >= X.shape[0]:
            epo_X = X if self.to_device is None else X.to(X._backend, dtype=X.dtype, device=self.to_device)
            return self.module.forward(epo_X)
        
        # We have to batchly predict to avoid exceeding the limit of memory
        else:
            pred = None
            start = 0
            while start < X.shape[0]:
                stp_X = self._fit_slice_batch_X(X, start = start, batch_size = self.batch_size, to_device = self.to_device, **kwargs)
                stp_pred = self.module.forward(stp_X)
                if pred is None:
                    pred = stp_pred
                else:
                    pred = pred.vstack(stp_pred)
                start += self.batch_size
            return pred

    def predict_loss(self, X: Tensor, y: Tensor, **kwargs) -> Tuple[Tensor, Tensor]:
        """
        Predict target values for samples in X and calculate the loss by a given target y.
        
        Returns:
            Tuple[Tensor, Tensor]: output of predictions, loss.
            
        Raises:
            RuntimeError: if you did NOT switched to evalation mode.
        """
        # Check the module training status
        if self.module.training == True:
            raise RuntimeError("In predict(), you called this in training mode. Please call .eval() to make the model safe in evaluation mode.")
        
        # Type Check (must be an Tensor type).
        if isinstance(X, Tensor) == False:
            raise ValueError("Input dataset must be either Matrix and Tensor. Use Matrix(data) or Tensor(data) to convert.")
        
        # Conduct a forward pass and return result
        pred = self.predict(X, **kwargs)    
        
        # Compute loss function
        loss = self.criterion.forward(pred, y if self.to_device is None else y.to(backend=y._backend, dtype=y.dtype, device=self.to_device))
        return pred, loss
    
    def __repr__(self):
        return f"NN Simple Interf Evaluator(name = {self.name}, task = {self.task}, has trained n_epoch = {self.n_epoch}, n_step = {self.n_step})."
    
    
# Alias for nn_SInterf_Evaluator
Evaluator = nn_SInterf_Evaluator


# Test case of nn_SInterf_Evaluator
if __name__ == "__main__":

    from nn import Tensor
    from nn import Dense
    from nn import Sigmoid, ReLU
    from nn import Softmax
    from nn import Module, nn_Module
    from nn import RMSE, MSE, MultiCrossEntropy
    from nn import Adam
    from nn import Evaluator

    ##############################################
    #
    # Regression Test
    class reg_test(Module):

        def __init__(self, **kwargs):

            super().__init__(module_name="reg_test", **kwargs)
            self.dense = Dense(4, 16, True, **kwargs)
            self.actv = Sigmoid(**kwargs)
            self.sumover = Dense(16, 1, True, **kwargs)

        def forward(self, inputs):
            out = self.dense.forward(inputs)
            out = self.actv.forward(out)
            out = self.sumover.forward(out)
            return out

    class reg_test2(Module):

        def __init__(self, **kwargs):

            super().__init__(module_name="reg_test", **kwargs)
            self.dense = Dense(4, 96, True, **kwargs)
            self.actv = Sigmoid(**kwargs)
            self.dense2 = Dense(96, 256, True, **kwargs)
            self.actv2 = Sigmoid(**kwargs)
            self.sumover = Dense(256, 1, True, **kwargs)

        def forward(self, inputs):
            out = self.dense.forward(inputs)
            out = self.actv.forward(out)
            out = self.dense2.forward(out)
            out = self.actv2.forward(out)
            out = self.sumover.forward(out)
            return out

    # Sample Data 1
    inputs = Tensor([[0, 0.1, 0.5, 0.9], [1, 0.2, -0.1, -0.9]], backend="torch")
    label = Tensor([[0.002312], [0.991215]], backend="torch")
    # Sample Data 2
    inputs = Tensor.rand([84240, 4], backend="torch", device="cuda").logistic_inv()
    label = (inputs.logistic() + Tensor.rand([84240, 4], backend="torch", device="cuda") * 0.1).sum(axis = 1, keepdims=True).tanh()

    # Create a model
    x = reg_test2(backend="torch", device="cuda")
    
    # Create a crit and optm
    crit = RMSE(backend="torch", device="cuda")
    optm = Adam(x.parameters(), lr=1E-4)
    
    # Train using evaluator
    ev = Evaluator("My Evaluator", "regression",
                   module = x,
                   criterion = crit,
                   optimizer = optm)
    # Fit
    ev.fit(inputs, label, 
           epoches=100,
           batch_size=7680,
           verbosity=1,
           evalset={"Train": (inputs, label)},
           evalmetrics=["MSE", "RMSE", "R2"],
           evalper=1)
    
    # Try predicting
    pred = ev.predict(inputs)
    
    # Save the model
    ev.save(ev, "torch.bin")
    