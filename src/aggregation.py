# aggregation.py
#
# Aggregation Methods in Bagging Learners
# From MML Library by Nathmath

import numpy as np
try:
    import torch
except ImportError:
    torch = None

from objtyp import Object
from tensor import Tensor
from matrix import Matrix

from baseml import MLBase, Classification, Regression

# Base Class for Aggregation
class Aggregation(Classification, Regression):
    
    __attr__ = "MML.Aggregation"
    
    def __init__(self, *, predictions: Matrix | Tensor, method: str = 'mean', floattype: type = float):
        
        # Basic Type Check
        if isinstance(predictions, Object) == False:
            raise ValueError("Input predictions must be either Matrix and Tensor. Use Matrix(data) or Tensor(data) to convert.")

        if len(predictions.shape) < 3:
            raise ValueError("Input predictions must also be at least 3D (n_estimators, n_samples, n_outputs, ...).")
        
        super().__init__()
        
        # Record predictions
        self.floattype = floattype
        self.predictions = predictions if predictions.dtype == floattype else predictions.astype(floattype)
        
        # Record method
        self.method = method

    def compute(self):
        raise NotImplementedError("Compute method is NOT implemented in the base class.")   
    
    def __repr__(self):
        return "Aggregation(Abstract Class)."
    

# Aggregation for Regression
class RegressionAggregation(Aggregation):
    
    __attr__ = "MML.RegressionAggregation"
    
    def __init__(self, predictions: Matrix | Tensor, method: str = "mean", floattype: type = float, **kwargs):
        """
        Initialize a Regression Aggregation instance.

        Parameters:
            predictions: Matrix | Tensor of shape (n_estimators, n_samples, n_outputs, ...),
                         containing predictions from each weak learner.
            method: str, one of {"mean", "median", "trimmed_mean", "trimmed_mean_align", 
                            "winsorized_mean", "weighted", "quantile"}.
            floattype: type, float type used to cast input, otherwise methods may yield errors.
            **kwargs: Additional parameters per method:
                - `mean`: NO additional arg.
                - `median`: NO additional arg.
                - `trimmed_mean`: `trim_frac` : float in [0,0.5)
                - `trimmed_mean_align`: `trim_frac` : float in [0,0.5)
                                        `along_col` : int of the index of column to be sorted
                - `winsorized_mean`: `limits` : tuple(lower_frac, upper_frac)
                - `weighted`: `weights` : weights per sample, 1D Matrix | Tensor | list with length n_estimators
                - `quantile`: `q` : float in decimal for quantile as the aggregated value, like 0.50
        """
        # Type check: must be Matrix or Tensor
        if not isinstance(predictions, Object):
            raise ValueError("`preds` must be a Matrix or Tensor. Use Matrix(data) or Tensor(data) to convert.")
        
        # Aggregation Dimensional Check
        if len(predictions.shape) < 3:
            raise ValueError("Input predictions must also be at least 3D (n_estimators, n_samples, n_outputs, ...).")
        
        # Method check.
        if method.lower() not in {"mean", "median", "trimmed_mean", "trimmed_mean_align", "winsorized_mean", "weighted", "quantile"}:
            raise ValueError("Unknown method. Choose a valid method then.")
        
        super().__init__(predictions = predictions, method = method, floattype = floattype)
        
        # Record the kwargs into a self.kwargs.
        self.kwargs = kwargs
        
        # Record the typeclass of the input matrix | tensor
        self.typeclass = type(predictions)

    def compute(self) -> Matrix | Tensor:
        """
        Aggregate predictions according to the specified method initialized in __init__.

        Parameters:
            None

        Returns:
            Aggregated predictions as a Matrix or Tensor.
        """

        if self.method == "mean":
            return self._mean()
        elif self.method == "median":
            return self._median()
        elif self.method == "trimmed_mean":
            return self._trimmed_mean(self.kwargs.get("trim_frac", 0.01))
        elif self.method == "trimmed_mean_align":
            return self._trimmed_mean_align(self.kwargs.get("trim_frac", 0.01), self.kwargs.get("along_col", 0))
        elif self.method == "winsorized_mean":
            return self._winsorized_mean(self.kwargs.get("limits", (0.01, 0.01)))
        elif self.method == "weighted":
            return self._weighted_mean(self.kwargs["weights"])
        elif self.method == "quantile":
            return self._quantile(self.kwargs["q"])
        else:
            raise ValueError(f"Unknown method '{self.method}'")

    def _mean(self) -> Matrix | Tensor:
        """
        Compute the arithmetic mean across the estimator axis (axis=0).

        Returns:
            Matrix | Tensor of shape (n_samples, ...).
        """
        return self.predictions.mean(axis = 0)

    def _median(self) -> Matrix | Tensor:
        """
        Compute the median across the estimator axis (axis=0).

        Returns:
            Matrix | Tensor of shape (n_samples, ...).
        """
        return self.predictions.median(axis = 0)

    def _trimmed_mean(self, trim_frac: float) -> Matrix | Tensor:
        """
        Compute the trimmed mean by removing the smallest and largest
        trim_frac fraction of predictions before averaging.

        Parameters:
            trim_frac: Fraction to trim on each end (0 <= trim_frac < 0.5).
    
        Returns:
            Matrix | Tensor of trimmed-mean predictions, shape (n_samples, ...).
        """

        # trim_frac Check
        if trim_frac < 0 or trim_frac >= 0.5:
            raise ValueError(f"Invalid Optional Arg `trim_frac` = {trim_frac}. It should be in [0, 0.5)")
        
        n_classes = self.predictions.shape[0]
        k = int(round(trim_frac * n_classes))
        # If no trimming, fallback to mean
        if k == 0:
            return self._mean()
        
        # Ensure trimming is valid
        if 2 * k >= n_classes:
            raise ValueError(f"`trim_frac` = {trim_frac} too large for number of estimators.")
            
        # Sort along estimator axis and slice
        sorted_data = self.predictions.sort(axis = 0)
        trimmed = sorted_data[k:-k]
        return trimmed.mean(axis = 0)

    def _trimmed_mean_align(self, trim_frac: float, along_col: int) -> Matrix | Tensor:
        """
        Compute the trimmed mean by removing the smallest and largest
        trim_frac fraction of predictions before averaging.
        Use column aligned sorting to make each row appears all together.
        Note: Align Mode only support 2D Predictions.

        Parameters:
            trim_frac: Fraction to trim on each end (0 <= trim_frac < 0.5).
            along_col: The column index to sort the data.
    
        Returns:
            Matrix | Tensor of trimmed-mean predictions, shape (n_samples, ...).
        """
        # 2D Output Only, check
        if len(self.predictions.shape) != 3:
            raise ValueError(f"Method `trimmed_mean` can only be applied to n_estimators * a 2D output, it should be a 3D array but you have {len(self.predictions.shape)} axis(es).")
        
        # Arg along_col outrange test
        if self.along_col < 0 or self.along_col >= self.predictions.shape[2]:
            raise ValueError(f"along_col must be within the range [0, {self.predictions.shape[1]}], but you provided {self.along_col}.")
            
        # trim_frac Check
        if trim_frac < 0 or trim_frac >= 0.5:
            raise ValueError(f"Invalid Optional Arg `trim_frac` = {trim_frac}. It should be in [0, 0.5)")
        
        n_classes = self.predictions.shape[0]
        k = int(round(trim_frac * n_classes))
        # If no trimming, fallback to mean
        if k == 0:
            return self._mean()
        
        # Ensure trimming is valid
        if 2 * k >= n_classes:
            raise ValueError(f"`trim_frac` = {trim_frac} too large for number of estimators.")
            
        # Sort along estimator axis and slice
        sorted_data = self.predictions.sort_along(axis = (None, None, along_col))
        trimmed = sorted_data[k:-k]
        return trimmed.mean(axis = 0)

    def _winsorized_mean(self, limits: tuple) -> Matrix | Tensor:
        """
        Compute the winsorized mean by clipping predictions to the
        [lower_frac, 1-upper_frac] quantiles before averaging.

        Parameters:
            limits: (lower_frac, upper_frac) tuple of two.

        Returns:
            Matrix | Tensor of winsorized-mean predictions, shape (n_samples, ...).
        """
        
        # Check if limits is indeed a tuple and has exactly two elements
        if not isinstance(limits, tuple) or len(limits) != 2:
            raise TypeError("limits must be a tuple of length 2.")
        
        lower_frac, upper_frac = limits
        lower = self.predictions.quantile(lower_frac, axis = 0)
        upper = self.predictions.quantile(upper_frac if upper_frac > 0.5 else 1 - upper_frac, axis = 0)
        clipped = self.predictions.clip(lower, upper)
        return clipped.mean(axis = 0)

    def _weighted_mean(self, weights: list | np.ndarray | Matrix | Tensor) -> Matrix | Tensor:
        """
        Compute a weighted average of predictions.

        Parameters:
            weights: 1D Matrix | Tensor of each estimator

        Returns:
            Matrix | Tensor of weighted-mean predictions, shape (n_samples, ...).
        """
        # If weights not Object, convert
        if isinstance(weights, Object) == False:
            weights = self.typeclass(weights, backend = self.predictions._backend, dtype = self.floattype, device = self.predictions.device)
        
        # Weights Dim check
        if len(weights.shape) != 1:
            raise ValueError(f"weights should have length 1, but is {len(weights.shape)}")
        
        # High dimensional data 1 column reshape factor
        reshape_factor = np.repeat(1, len(self.predictions.shape)).tolist()
        reshape_factor[0] = -1
        
        normalized_weights = weights.reshape(reshape_factor) / weights.sum()
        weighted = self.predictions * normalized_weights
        return weighted.sum(axis = 0)

    def _quantile(self, q: float) -> Matrix | Tensor:
        """
        Compute specified quantile(s) across estimators for each sample.

        Parameters:
            q: quantile float.

        Returns:
            Matrix | Tensor of q quantile result, shape (n_samples, ...).
        """
        return self.predictions.quantile(q, axis = 0)


# Aggregation for Classification
class ClassificationAggregation(Aggregation):
    
    __attr__ = "MML.ClassificationAggregation"
    
    def __init__(self, predictions: Matrix | Tensor, method: str = "hard_vote", n_classes: int | None = None, floattype: type = float, **kwargs):
        """
        Initialize a classification aggregator for bagged learners.
        For multi-classification, we only support one-hot output. Even though you may input a non-one-hot data.

        Parameters:
            predictions: Matrix | Tensor of shape (n_estimators, n_samples, n_outputs, ...),
                         containing predictions from each weak learner.
            method: str, one of { 
                "mean", "weighted",           # Raw methods, return probabilities 
                "threshold", "hard_vote",     # One-hot methods, return 0-1 matrix
                "soft_vote", "weighted_vote", # One-hot methods, return 0-1 matrix
                }.
            n_classes: int | None, number of classes, if not given, then inferred.
            floattype: type, float type for probabilities.
            **kwargs: method-specific arguments:
                - `mean`: NO additional arg. Raw probability output if one-hot.
                - `weighted`: `weights` : weights per sample, 1D Matrix | Tensor | list with length n_samples
                - `threshold`: `threshold`, float, to determine whether to classify one column to 1 or 0
                - `hard_vote`: NO additional arg.
                - `soft_vote`: `prob`: bool, if returns a probabilities matrix or not (one-hot matrix)
                - `weighted_vote`: `weights` : weights per sample, 1D Matrix | Tensor | list with length n_samples
                                   `prob`: bool, if returns a probabilities matrix or not (one-hot matrix)
        """
        
        # Type check: must be Matrix or Tensor
        if not isinstance(predictions, Object):
            raise ValueError("`predictions` must be a Matrix or Tensor. Use Matrix(data) or Tensor(data) to convert.")
        
        # Aggregation Dimensional Check
        if len(predictions.shape) < 3:
            raise ValueError("Input predictions must also be at least 3D (n_estimators, n_samples, n_outputs, ...).")
        
        # Method check.
        if method.lower() not in {"mean", "weighted", "threshold", "hard_vote", "soft_vote", "weighted_vote"}:
            raise ValueError("Unknown method. Choose a valid method then.")
        
        super().__init__(predictions = predictions, method = method, floattype = floattype)
        
        # Record the kwargs into a self.kwargs.
        self.kwargs = kwargs
        
        # Record the typeclass of the input matrix | tensor.
        self.typeclass = type(predictions)
        
        # Infer number of classes
        if n_classes is not None:
            self.n_classes = n_classes
        else:
            if self.predictions.shape[2] > 1:
                self.n_classes = self.predictions.shape[2]
            else:
                self.n_classes = max(len(self.predictions.flatten().bincount()), 2)
        
        # Convert ANY input into a true probability tensor
        self.pred_onehot = self.predictions.copy() if self.predictions.shape[2] > 1 else None
        if self.pred_onehot is None:
            # Not one-hot, need to encode     
            dimensions = list(self.predictions.shape)
            dimensions[-1] = self.n_classes
            self.pred_onehot = self.typeclass.zeros(dimensions, backend=self.predictions._backend, dtype=floattype).to(backend=self.predictions._backend, dtype=floattype, device=self.predictions.device)
            for i in range(self.predictions.shape[0]):
                self.pred_onehot[i] = self._to_onehot(self.predictions[i], self.n_classes).astype(floattype)
    
    def compute(self) -> Matrix | Tensor:
        """
        Aggregate predictions according to the specified method initialized in __init__.

        Parameters:
            None

        Returns:
            Aggregated predictions as a Matrix or Tensor.
        """
        if self.method == "mean":
            return self._mean()
        elif self.method == "weighted":
            return self._weighted(self.kwargs["weights"])
        elif self.method == "threshold":
            return self._threshold(self.kwargs.get("threshold", 0.5))
        elif self.method == "hard_vote":
            return self._hard_vote()
        elif self.method == "soft_vote":
            return self._soft_vote(self.kwargs.get("prob", True))
        elif self.method == "weighted_vote":
            return self._weighted_vote(self.kwargs["weights"], self.kwargs.get("prob", True))
        else:
            raise ValueError(f"Unknown method '{self.method}'")
          
    def _mean(self) -> Matrix | Tensor:
        """
        Using the sample mean of one_hot results to determine the class distribution.
        Return the raw axis-0-wise-mean without doing anything special.

        Returns:
            Matrix | Tensor, (n_samples, n_classes).
        """
        return self.pred_onehot.mean(axis=0)
        
    def _weighted(self, weights: list | np.ndarray | Matrix | Tensor) -> Matrix | Tensor:
        """
        Using the weighted mean of one_hot results to determine the class distribution.
        Return the raw axis-0-wise-mean without doing anything special.
        
        Parameters:
            weights: 1D Matrix | Tensor of each estimator

        Returns:
            Matrix | Tensor, (n_samples, n_classes).
        """
        # If weights not Object, convert
        if isinstance(weights, Object) == False:
            weights = self.typeclass(weights, backend = self.pred_onehot._backend, dtype = self.floattype, device = self.pred_onehot.device)
        
        # Weights Dim check
        if len(weights.shape) != 1:
            raise ValueError(f"weights should have length 1, but is {len(weights.shape)}")
        
        # High dimensional data 1 column reshape factor
        reshape_factor = np.repeat(1, len(self.pred_onehot.shape)).tolist()
        reshape_factor[0] = -1
        
        normalized_weights = weights.reshape(reshape_factor) / weights.sum()
        weighted = self.pred_onehot * normalized_weights
        return weighted.sum(axis = 0)
        
    def _threshold(self, threshold: float) -> Matrix | Tensor:
        """
        Using a threshold to determine the result of binary classification.

        Returns:
            Matrix | Tensor, (n_samples, n_classes).
        """
        # If it is not a binary case (self.n_classes), then raise error
        if self.n_classes != 2:
            raise ValueError("This method should only be used with a binary classifier.")
        
        # Check if the threshold is out of bounds.
        if not 0 <= threshold <= 1.0:
            raise ValueError("Threshold must be between 0 and 1.")
        
        avg = self.pred_onehot.mean(axis=0)
        return self.typeclass(avg.data >= threshold, backend = self.pred_onehot._backend, dtype = self.floattype, device = self.pred_onehot.device)

    def _hard_vote(self) -> Matrix | Tensor:
        """
        Sum one-hot counts over estimators, then argmax -> discrete label.

        Returns:
            Matrix | Tensor, (n_samples, n_classes).
        """
        # vote_counts: (n_samples, n_classes)
        vote_counts = self.pred_onehot.sum(axis=0)
        # labels: (n_samples,)
        labels = vote_counts.argmax(axis=1)
        # make it 2D: (n_samples,1)
        labels = labels.reshape([-1, 1])
        # Convert to one-hot
        return self._to_onehot(labels, self.n_classes)
    
    def _soft_vote(self, prob: bool = False) -> Matrix | Tensor:
        """
        Average class‑probabilities across estimators and then argmax.

        Returns:
            Matrix | Tensor, (n_samples, n_classes).
        """
        if prob == True:
            return self.pred_onehot.mean(axis=0)
        else:
            # labels: (n_samples, n_classes)
            probabilities = self.pred_onehot.mean(axis=0)
            # labels: (n_samples,)
            labels = probabilities.argmax(axis=1)
            # make it 2D: (n_samples,1)
            labels = labels.reshape([-1, 1])
            # Convert to one-hot
            return self._to_onehot(labels, self.n_classes)
    
    def _weighted_vote(self, weights: list | np.ndarray | Matrix | Tensor, prob: bool = False) -> Matrix | Tensor:
        """
        Weighted average of probabilities.

        Parameters:
            weights: 1D Matrix | Tensor of each estimator
            
        Returns:
            Matrix | Tensor, (n_samples, n_classes).
        """
        weighted = self._weighted(weights)
        if prob == True:
            return weighted
        else:
            # labels: (n_samples,)
            labels = weighted.argmax(axis=1)
            # make it 2D: (n_samples,1)
            labels = labels.reshape([-1, 1])
            # Convert to one-hot
            return self._to_onehot(labels, self.n_classes)
        

if __name__ == "__main__":
    
    # Test batch
    
    _preds = np.array([
        [[1,2],[3,4]],
        [[2,3],[4,5]],
        [[0,1],[2,3]]
    ])
    backend = "torch"
    m_preds = Matrix(_preds, backend)
    
    
    # Regression: mean, median, trimmed, weighted, quantile
    reg = RegressionAggregation(m_preds, "mean")
    assert np.allclose(reg.compute().data, np.array([[1,2],[3,4]]))
    
    reg = RegressionAggregation(m_preds, "median")
    assert np.allclose(reg.compute().data, np.array([[1,2],[3,4]]))
    
    reg = RegressionAggregation(m_preds, "trimmed_mean", trim_frac=1/3)
    assert np.allclose(
        reg.compute().data,
        np.array([[1,2],[3,4]])
    )
    
    reg = RegressionAggregation(m_preds, "weighted", weights = [0.5,0.5,0])
    assert np.allclose(
        reg.compute().data,
        np.array([[1.5,2.5],[3.5,4.5]])
    )
    
    reg = RegressionAggregation(m_preds, "quantile", q=0.75)
    assert np.allclose(
        reg.compute().data,
        np.quantile(_preds,0.75,axis=0)
    )

    # Binary‑classification tests  (n_classes = 2)
    _preds = np.array([
        [[0.6],[0.2]],
        [[0.9],[0.8]],
        [[0.1],[0.6]],
        [[0.7],[0.9]]
    ])
    backend = "torch"
    m_preds = Matrix(_preds, backend)
    
    clf = ClassificationAggregation(m_preds, method="mean", n_classes=2)
    expected_mean = np.mean(_preds, axis=0)
    assert np.allclose(clf.compute().data[:,1], expected_mean)
    
    clf = ClassificationAggregation(m_preds, method="hard_vote", n_classes=2)
    expected_labels = np.array([[0,1], [0,1]])
    assert np.array_equal(clf.compute().data, expected_labels)
    
    clf = ClassificationAggregation(
        m_preds, method="threshold", n_classes=2, threshold=0.5
    )
    expected_thresh = (np.mean(_preds, axis=0) >= 0.5).astype(int)
    assert np.array_equal(clf.compute().data[:,1], expected_thresh.flatten())
    
    clf = ClassificationAggregation(
        m_preds, method="soft_vote", n_classes=2, prob=False
    )
    avg_probs = np.mean(_preds, axis=0)
    expected_onehot = np.zeros_like(avg_probs, dtype=int)
    expected_onehot[np.arange(2), avg_probs.argmax(axis=1)] = 1
    assert np.array_equal(clf.compute().data[:,1], expected_onehot.flatten())
    
    # Multi‑class tests  (n_classes = 3)
    _preds_multi = np.array([
    [[0.3, 0.6, 0.1], [0.1, 0.3, 0.6]],  # estimator‑0
    [[0.6, 0.3, 0.1], [0.2, 0.2, 0.6]],  # estimator‑1
    [[0.3, 0.5, 0.2], [0.4, 0.3, 0.3]]   # estimator‑2
    ])
    m_preds_multi = Matrix(_preds_multi, backend)
    
    clf = ClassificationAggregation(m_preds_multi, method="mean")
    assert np.allclose(
        clf.compute().data,
        np.mean(_preds_multi, axis=0)
    )
    
    clf = ClassificationAggregation(m_preds_multi, method="hard_vote")
    labels_multi = np.argmax(_preds_multi, axis=2)
    maj_multi = np.apply_along_axis(
        lambda col: np.eye(3, dtype=int)[np.bincount(col, minlength=3).argmax()],
        0, labels_multi).T
    assert np.array_equal(clf.compute().data, maj_multi)
