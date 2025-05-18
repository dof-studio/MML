# metrices.py
#
# Evaluation metirces for regressions and classifications
# From MML Library by Nathmath

import numpy as np
try:
    import torch
except ImportError:
    torch = None
    
from objtyp import Object
from tensor import Tensor
from matrix import Matrix

# Metrics Base Class
class BaseMetrics:
    
    __attr__ = "MML.BaseMetrics"

    def __init__(self):
        pass
    
    def compute(self):
        """
        Compute the specified metric for the predictions given true data.
        """
        raise NotImplementedError("Compute is NOT implemented in the base class.")
        
    def deriv_1(self):
        """
        Compute the sample-wise 1st order derivatives of metric for the predictions given true data.
        """
        raise NotImplementedError("Deriv_1 is NOT implemented in the base class.")
        
    def deriv_2(self):
        """
        Compute the sample-wise 2nd order derivatives of metric for the predictions given true data.
        """
        raise NotImplementedError("Deriv_2 is NOT implemented in the base class.")
    
    def __repr__(self):
        return "BaseMetrics(Abstract Class)."


# Metrics for regression
class RegressionMetrics(BaseMetrics):
    """
    A class to compute common regression metrics between predicted results and target values.
    
    Supported metrics:
        - MSE (Mean Squared Error)
        - RMSE (Root Mean Squared Error) 
        - MAE (Mean Absolute Error)
        - MAPE (Mean Absolute Percentage Error)
        - Huber Loss
        - Quantile Loss
        - WMSE (Weighted Mean Squared Error)
        - WRMSE (Weighted Root Mean Squared Error)
        - R^2 (R Square)
        - Adjusted R^2 (Adjusted R Square)
        
    Special metrics:
        - Negative R^2 (R Square)
        - Negative Adjusted R^2 (Adjusted R Square)
        
    The computations are performed using the underlying tensor operations, maintaining
    compatibility with both numpy and torch backends.
    
    Attributes:
        result: Predicted results tensor
        target: Target values tensor
        metric_type: String specifying which metric to compute ('mse', 'rmse', 'mae', 'mape', 'r2', 'adjusted r2')
    """
    
    __attr__ = "MML.RegressionMetrics"
    
    def __init__(self, result: Tensor | Matrix, target: Tensor | Matrix, metric_type: str, k: int | None = None, **kwargs):
        """
        Initializes the RegressionMetrics instance with result and target tensors.
        
        Args:
            result (Tensor | Matrix): Predicted results tensor
            target (Tensor | Matrix): Target values tensor
            metric_type (str): Metric type to compute ('mse', 'rmse', 'mae', 'mape', 'huber_loss', 'quantile_loss', 
                              'wmse', 'wrmse', 'r2', 'adjusted r2', 'nr2', 'nadjusted r2')
            k (int): Number of predictors (parameters) in the model, only used in Adjusted R2.
        """
        super().__init__()
        
        # Different instances or different backends.
        if isinstance(result,  Object) == False or isinstance(target, Object) == False:
            raise ValueError("Predicted `result` and real `target` should be either `Matrix` or `Tensor` type!")
        if type(result) != type(target):
            raise ValueError("Predicted `result` and real `target` should have the same type, either Tensor or Matrix!")
        if result._backend != target._backend:
            raise ValueError("Predicted `result` and real `target` should have the same backend, either numpy or torch!")
        
        # Member variables.
        self.k = k
        self.result = result
        self.target = target
        self.typeclass = type(result)
        self.metric_type = metric_type.lower()
        
        if not self.result.shape == self.target.shape:
            raise ValueError("Result and target tensors must have the same shape.")
            
    def compute(self, **kwargs) -> Tensor | Matrix:
        """
        Computes the specified regression metric between result and target.
        
        Args:
            **kwargs: Other arguments supported by metrics.
        
        Returns:
            Tensor | Matrix: The computed metric value as a tensor
        """
        if self.metric_type == 'mse':
            return self._compute_mse(**kwargs)
        elif self.metric_type == 'rmse':
            return self._compute_rmse(**kwargs)
        elif self.metric_type == 'mae':
            return self._compute_mae(**kwargs)
        elif self.metric_type == 'mape':
            return self._compute_mape(**kwargs)
        elif self.metric_type == 'huber_loss':
            return self._compute_huber_loss(**kwargs)
        elif self.metric_type == 'quantile_loss':
            return self._compute_quantile_loss(**kwargs)
        elif self.metric_type == 'wmse':
            return self._compute_wmse(**kwargs)
        elif self.metric_type == 'wrmse':
            return self._compute_wrmse(**kwargs)
        elif self.metric_type == 'r2':
            return self._compute_r2(**kwargs)
        elif self.metric_type == 'adjusted r2':
            return self._compute_adjusted_r2(**kwargs)
        # Special Metrics
        elif self.metric_type == 'nr2':
            return - self._compute_r2(**kwargs)
        elif self.metric_type == 'nadjusted r2':
            return - self._compute_adjusted_r2(**kwargs)
        else:
            raise ValueError(f"Unsupported metric type: {self.metric_type}")
    
    def deriv_1(self, **kwargs) -> Tensor | Matrix:
        """
        Computes the sample-wise 1st order derivative of the specified regression metric between result and target.
        
        Args:
            **kwargs: Other arguments supported by metrics.
        
        Returns:
            Tensor | Matrix: The computed gradient vector as a matrix or a tensor
        """
        if self.metric_type == 'mse':
            return self._deriv_1_mse(**kwargs)
        elif self.metric_type == 'rmse':
            return self._deriv_1_rmse(**kwargs)
        elif self.metric_type == 'mae':
            return self._deriv_1_mae(**kwargs)
        elif self.metric_type == 'mape':
            return self._deriv_1_mape(**kwargs)
        elif self.metric_type == 'huber_loss':
            return self._deriv_1_huber_loss(**kwargs)
        elif self.metric_type == 'quantile_loss':
            return self._deriv_1_quantile_loss(**kwargs)
        elif self.metric_type == 'wmse':
            return self._deriv_1_wmse(**kwargs)
        elif self.metric_type == 'wrmse':
            return self._deriv_1_wrmse(**kwargs)
        elif self.metric_type == 'r2':
            raise  ValueError(f"Metric type: {self.metric_type} cannot compute derivatives.")
        elif self.metric_type == 'adjusted r2':
            raise  ValueError(f"Metric type: {self.metric_type} cannot compute derivatives.")
        # Special Metrics
        elif self.metric_type == 'nr2':
            raise  ValueError(f"Metric type: {self.metric_type} cannot compute derivatives.")
        elif self.metric_type == 'nadjusted r2':
            raise  ValueError(f"Metric type: {self.metric_type} cannot compute derivatives.")
        else:
            raise  ValueError(f"Unsupported metric type: {self.metric_type}")
    
    def deriv_2(self, **kwargs) -> Tensor | Matrix:
        """
        Computes the sample-wise 2nd order derivative of the specified regression metric between result and target.
        
        Args:
            **kwargs: Other arguments supported by metrics.
        
        Returns:
            Tensor | Matrix: The computed hessian matrix (without cross terms) as a matrix or a tensor
        """
        if self.metric_type == 'mse':
            return self._deriv_2_mse(**kwargs)
        elif self.metric_type == 'rmse':
            return self._deriv_2_rmse(**kwargs)
        elif self.metric_type == 'mae':
            return self._deriv_2_mae(**kwargs)
        elif self.metric_type == 'mape':
            return self._deriv_2_mape(**kwargs)
        elif self.metric_type == 'huber_loss':
            return self._deriv_2_huber_loss(**kwargs)
        elif self.metric_type == 'quantile_loss':
            return self._deriv_2_quantile_loss(**kwargs)
        elif self.metric_type == 'wmse':
            return self._deriv_2_wmse(**kwargs)
        elif self.metric_type == 'wrmse':
            return self._deriv_2_wrmse(**kwargs)
        elif self.metric_type == 'r2':
            raise  ValueError(f"Metric type: {self.metric_type} cannot compute derivatives.")
        elif self.metric_type == 'adjusted r2':
            raise  ValueError(f"Metric type: {self.metric_type} cannot compute derivatives.")
        # Special Metrics
        elif self.metric_type == 'nr2':
            raise  ValueError(f"Metric type: {self.metric_type} cannot compute derivatives.")
        elif self.metric_type == 'nadjusted r2':
            raise  ValueError(f"Metric type: {self.metric_type} cannot compute derivatives.")
        else:
            raise  ValueError(f"Unsupported metric type: {self.metric_type}")
        
    def _compute_mse(self, axis: int | None = None, **kwargs) -> Tensor | Matrix:
        """
        Computes the Mean Squared Error between result and target.
        
        Args:
            axis: None or int, if you intend to get the per-output metrics/derivs, set axis = 0. Else None.
        
        Returns:
            Tensor | Matrix: MSE tensor or matrix
        """
        error = (self.result - self.target)
        squared_error = error ** 2
        if axis is None:
            mean_squared_error = squared_error.sum(axis = axis) / np.array(squared_error.shape).prod()
        else:
            mean_squared_error = squared_error.sum(axis = axis) / squared_error.shape[axis]
        return mean_squared_error
    
    def _deriv_1_mse(self, axis: int | None = None, **kwargs) -> Tensor | Matrix:
        """
        Computes the first-order derivative of the Mean Squared Error between result and target.
        
        Args:
            axis: None or int, if you intend to get the per-output metrics/derivs, set axis = 0. Else None.
        
        Returns:
            Tensor | Matrix: Derivative of MSE tensor or matrix with respect to result.
        """
        error = self.result - self.target
        grad = 2 * error / error.shape[0]
        if axis is None:
            grad = 2 * error / np.array(error.shape).prod()
        else:
            grad = 2 * error / error.shape[axis]
        return grad
    
    def _deriv_2_mse(self, axis: int | None = None, **kwargs) -> Tensor | Matrix:
        """
        Computes the second-order derivative (Hessian diagonal) of the Mean Squared Error between result and target.
    
        Args:
            only_diag: bool, if True, only calculate the diagonal vector and return,
                             else, return the full hessian matrix.
    
        Returns:
            Tensor | Matrix: Constant Hessian of MSE (2/N) with the same shape as result.
        """
        ones = self.result.copy(); ones[...] = 1;
        if axis is None:
            return ones * (2.0 / np.array(self.result.shape).prod())
        else:
            return ones * (2.0 / self.result.shape[axis])
    
    def _compute_rmse(self, axis: int | None = None, **kwargs) -> Tensor | Matrix:
        """
        Computes the Root Mean Squared Error between result and target.
        
        Args:
            axis: None or int, if you intend to get the per-output metrics/derivs, set axis = 0. Else None.
                
        Returns:
            Tensor | Matrix: RMSE tensor or matrix
        """
        error = (self.result - self.target)
        squared_error = error ** 2
        if axis is None:
            mean_squared_error = squared_error.sum(axis = axis) / np.array(squared_error.shape).prod()
        else:
            mean_squared_error = squared_error.sum(axis = axis) / squared_error.shape[axis]
        rmse = mean_squared_error ** 0.5
        return rmse
    
    def _deriv_1_rmse(self, axis: int | None = None, **kwargs) -> Tensor | Matrix:
        """
        Computes the first-order derivative of the Root Mean Squared Error between result and target.
        
        Args:
            axis: None or int, if you intend to get the per-output metrics/derivs, set axis = 0. Else None.
        
        Returns:
            Tensor | Matrix: Derivative of RMSE tensor or matrix with respect to result.
        """
        error = self.result - self.target
        squared_error = error ** 2
        if axis is None:
            mean_squared_error = squared_error.sum(axis = axis) / np.array(squared_error.shape).prod()
            rmse = mean_squared_error ** 0.5
            grad = error / (np.array(squared_error.shape).prod() * rmse)
        else:
            mean_squared_error = squared_error.sum(axis = axis) / squared_error.shape[axis]
            rmse = mean_squared_error ** 0.5
            grad = error / (error.shape[axis] * rmse)

        return grad
    
    def _deriv_2_rmse(self, axis: int | None = None, **kwargs) -> Tensor | Matrix:
        """
        Computes the second-order derivative of the Root Mean Squared Error between result and target.
        
        Args:
            axis: None or int, if you intend to get the per-output metrics/derivs, set axis = 0. Else None.
        
        Returns:
            Tensor | Matrix: Second-order derivative (Hessian diagonal) of RMSE with respect to result.
        """
        error = self.result - self.target
        squared_error = error ** 2
        if axis is None:
            sum_squared = squared_error.sum(axis = axis)
            mean_squared_error = sum_squared / np.array(squared_error.shape).prod()
            rmse = mean_squared_error ** 0.5
            hessian = (sum_squared - squared_error) / ((np.array(squared_error.shape).prod() ** 2) * (rmse ** 3))
        else:
            sum_squared = squared_error.sum(axis = axis)
            mean_squared_error = sum_squared / squared_error.shape[axis]
            rmse = mean_squared_error ** 0.5
            hessian = (sum_squared - squared_error) / ((error.shape[axis] ** 2) * (rmse ** 3))
        
        return hessian
    
    def _compute_mae(self, axis: int | None = None, **kwargs) -> Tensor | Matrix:
        """
        Computes the Mean Absolute Error between result and target.
        
        Args:
            axis: None or int, if you intend to get the per-output metrics/derivs, set axis = 0. Else None.
        
        Returns:
            Tensor | Matrix: MAE tensor or matrix
        """
        error = (self.result - self.target)
        absolute_error = error.abs()
        if axis is None:
            mean_absolute_error = absolute_error.sum(axis = axis) / np.array(absolute_error.shape).prod()
        else:
            mean_absolute_error = absolute_error.sum(axis = axis) / absolute_error.shape[axis]
        return mean_absolute_error

    def _deriv_1_mae(self, axis: int | None = None, **kwargs) -> Tensor | Matrix:
        """
        Computes the first-order derivative of the Mean Absolute Error between result and target.
        
        Args:
            axis: None or int, if you intend to get the per-output metrics/derivs, set axis = 0. Else None.
        
        Returns:
            Tensor | Matrix: Derivative of MAE tensor or matrix with respect to result.
        """
        error = self.result - self.target
        grad = error.sign() / error.shape[0]
        if axis is None:
            grad = error.sign() / np.array(error.shape).prod()
        else:
            grad = error.sign() / error.shape[axis]
        return grad
    
    def _deriv_2_mae(self, axis: int | None = None, **kwargs) -> Tensor | Matrix:
        """
        Computes the second-order derivative of per-output MAE between result and target.
        
        Args:
            axis: None or int, if you intend to get the per-output metrics/derivs, set axis = 0. Else None.
                
        Returns:
            Tensor | Matrix: Hessian diagonal of MAE (zero), shape (N, D).
        """
        zeros = self.result.copy(); zeros[...] = 0;
        return zeros
    
    def _compute_mape(self, axis: int | None = None, **kwargs) -> Tensor | Matrix:
        """
        Computes the Mean Absolute Percentage Error between result and target.
        
        Note: Division by zero occurs if target contains zeros. This is handled
        gracefully by the underlying tensor operations, but users should ensure 
        target values are non-zero when using MAPE.
        
        Args:
            axis: None or int, if you intend to get the per-output metrics/derivs, set axis = 0. Else None.
        
        Returns:
            Tensor | Matrix: MAPE tensor or matrix
        """
        error = (self.result - self.target) / self.target
        absolute_percentage_error = error.abs()
        if axis is None:
            mean_absolute_percentage_error = absolute_percentage_error.sum(axis = axis) / np.array(absolute_percentage_error.shape).prod()
        else:
            mean_absolute_percentage_error = absolute_percentage_error.sum(axis = axis) / absolute_percentage_error.shape[axis]
        return mean_absolute_percentage_error

    def _deriv_1_mape(self, axis: int | None = None, **kwargs) -> Tensor | Matrix:
        """
        Computes the first-order derivative of the Mean Absolute Percentage Error between result and target.
        
        Args:
            axis: None or int, if you intend to get the per-output metrics/derivs, set axis = 0. Else None.
        
        Returns:
            Tensor | Matrix: Derivative of MAPE tensor or matrix with respect to result.
        """
        # Derivative: sign(ratio) * (1/target) / N
        ratio = (self.result - self.target) / self.target
        if axis is None:
            grad = ratio.sign() / (self.target * np.array(ratio.shape).prod())
        else:
            grad = ratio.sign() / (self.target * ratio.shape[axis])
        return grad
    
    def _deriv_2_mape(self, axis: int | None = None, **kwargs) -> Tensor | Matrix:
        """
        Computes the second-order derivative of per-output MAPE between result and target.
        
        Args:
            axis: None or int, if you intend to get the per-output metrics/derivs, set axis = 0. Else None.
                
        Returns:
            Tensor | Matrix: Hessian diagonal of MAPE (zero), shape (N, D).
        """
        zeros = self.result.copy(); zeros[...] = 0;
        return zeros
    
    def _compute_huber_loss(self, axis: int | None = None, delta: float = 1.0, **kwargs) -> Tensor | Matrix:
        """
        Computes the Huber Loss between result and target.
        
        Args:
            axis: None or int, if you intend to get the per-output metrics/derivs, set axis = 0. Else None.
        
        Returns:
            Tensor | Matrix: Huber Loss tensor or matrix
        """
        error = self.result - self.target
        abs_error = error.abs()
        # The mask is in an INTERNAL format (np/torch)
        small_mask = abs_error.data <= delta
        
        # The squared region: 0.5 * e^2
        sq_loss = 0.5 * (error ** 2)
        # The linear region: delta * (|e| - 0.5 * delta)
        lin_loss = delta * (abs_error - 0.5 * delta)
        
        # Huber loss is a combinition of mse and mae
        if self.result._is_numpy:
            huber = np.where(small_mask, sq_loss.data, lin_loss.data)
        else:
            huber = torch.where(small_mask, sq_loss.data, lin_loss.data)
        
        if axis is None:
            return type(self.result)(huber, backend=self.result._backend, dtype=self.result.dtype, device=self.result.device).sum(axis = axis) / np.array(error.shape).prod()
        else:
            return type(self.result)(huber, backend=self.result._backend, dtype=self.result.dtype, device=self.result.device).sum(axis = axis) / error.shape[axis]
    
    def _deriv_1_huber_loss(self, axis: int | None = None, delta: float = 1.0, **kwargs) -> Tensor | Matrix:
        """
        Computes the first-order derivative of the Huber Loss between result and target.
        
        Args:
            axis: None or int, if you intend to get the per-output metrics/derivs, set axis = 0. Else None.
            delta: float, the Huber threshold parameter.
        
        Returns:
            Tensor | Matrix: Derivative of Huber Loss tensor or matrix with respect to result.
        """
        error = self.result - self.target
        abs_error = error.abs()
        # The mask is in an INTERNAL format (np/torch)
        small_mask = abs_error.data <= delta
        
        if self.result._is_numpy:
            grad_elt = np.where(small_mask, error.data, delta * error.sign().data)
        else:
            grad_elt = torch.where(small_mask, error.data, delta * error.sign().data)
        
        if axis is None:
            return type(self.result)(grad_elt, backend=self.result._backend, dtype=self.result.dtype, device=self.result.device) / np.array(error.shape).prod()
        else:
            return type(self.result)(grad_elt, backend=self.result._backend, dtype=self.result.dtype, device=self.result.device) / error.shape[axis]

    def _deriv_2_huber_loss(self, axis: int | None = None, delta: float = 1.0, **kwargs) -> Tensor | Matrix:
        """
        Computes the second-order derivative (Hessian diagonal) of the Huber Loss between result and target.

        Args:
            axis: None or int, if you intend to get per-output metrics/derivs, set axis = 0. Else None.
            delta: float, the Huber threshold parameter.
            
        Returns:
            Tensor | Matrix: Second-order derivative of Huber Loss wrt result, shape like result.
        """
        error = self.result - self.target
        abs_error = error.abs()
        # The mask is in an INTERNAL format (np/torch)
        small_mask = abs_error.data <= delta

        if self.result._is_numpy:
            hess_elt = small_mask.astype(float)
        else:
            # torch.where on a boolean mask: 1.0 where small, else 0.0
            one = error.ones(error.shape, backend=self.result._backend).to(backend=self.result._backend, dtype=self.result.dtype, device=self.result.device)
            zero = error.zeros(error.shape, backend=self.result._backend).to(backend=self.result._backend, dtype=self.result.dtype, device=self.result.device)
            hess_elt = torch.where(small_mask, one.data, zero.data)

        if axis is None:
            return type(self.result)(hess_elt, backend=self.result._backend, dtype=self.result.dtype, device=self.result.device) / np.array(error.shape).prod()
        else:
            return type(self.result)(hess_elt, backend=self.result._backend, dtype=self.result.dtype, device=self.result.device) / error.shape[axis]

    def _compute_quantile_loss(self, axis: int | None = None, q: float = 0.5, **kwargs) -> Tensor | Matrix:
        """
        Computes the Quantile Loss between result and target.
        
        Args:
            axis: None or int, if you intend to get the per-output metrics/derivs, set axis = 0. Else None.
        
        Returns:
            Tensor | Matrix: Quantile Loss tensor or matrix
        """
        error = self.result - self.target
        if self.result._is_numpy:
            loss = np.where(error.data >= 0, q * error.data, (q - 1) * error.data)
        else:
            loss = torch.where(error.data >= 0, q * error.data, (q - 1) * error.data)
        
        if axis is None:
            return type(self.result)(loss, backend=self.result._backend, dtype=self.result.dtype, device=self.result.device).sum(axis = axis) / np.array(error.shape).prod()
        else:
            return type(self.result)(loss, backend=self.result._backend, dtype=self.result.dtype, device=self.result.device).sum(axis = axis) / error.shape[axis]

    def _deriv_1_quantile_loss(self, axis: int | None = None, q: float = 0.5, **kwargs) -> Tensor | Matrix:
        """
        Computes the first-order derivative of the Quantile Loss between result and target.
        
        Args:
            axis: None or int, if you intend to get per-output metrics/derivs, set axis = 0. Else None.
            q: float in (0,1), the quantile level.
            
        Returns:
            Tensor | Matrix: Derivative of Quantile Loss tensor or matrix with respect to result.
        """
        error = self.result - self.target
        if self.result._is_numpy:
            grad_elt = np.where(error.data >= 0, q, q - 1)
        else:
            grad_elt = torch.where(error.data >= 0, q, q - 1)

        if axis is None:
            return type(self.result)(grad_elt, backend=self.result._backend, dtype=self.result.dtype, device=self.result.device) / np.array(error.shape).prod()
        else:
            return type(self.result)(grad_elt, backend=self.result._backend, dtype=self.result.dtype, device=self.result.device) / error.shape[axis]

    def _deriv_2_quantile_loss(self, axis: int | None = None, q: float = 0.5, **kwargs) -> Tensor | Matrix:
        """
        Computes the second-order derivative (Hessian diagonal) of the Quantile Loss between result and target.

        Args:
            axis: None or int, if you intend to get per-output metrics/derivs, set axis = 0. Else None.
            q: float in (0,1), the quantile level.
            
        Returns:
            Tensor | Matrix: Second-order derivative of Quantile Loss wrt result, shape like result (all zeros).
        """
        error = self.result - self.target
        
        if axis is None:
            return error.zeros_like(error).to(backend=self.result._backend, dtype=self.result.dtype, device=self.result.device) / np.array(error.shape).prod()
        else:
            return error.zeros_like(error).to(backend=self.result._backend, dtype=self.result.dtype, device=self.result.device) / error.shape[axis]

    def _compute_wmse(self, axis: int | None = None, weights: Matrix | Tensor | None = None, **kwargs) -> Tensor | Matrix:
        """
        Computes the Weighted Mean Squared Error between result and target.
        
        Args:
            axis: None or int, if you intend to get the per-output metrics/derivs, set axis = 0. Else None.
            weights: Matrix or Tensor or None, if None, fail to normal mse.
        
        Returns:
            Tensor | Matrix: MSE tensor or matrix
        """
        if weights is None:
            return self._compute_mse(axis = axis, **kwargs)
        
        error = (self.result - self.target)
        squared_error = error ** 2
        if axis is None:
            mean_squared_error = (weights * squared_error).sum(axis = axis) / np.array(squared_error.shape).prod()
        else:
            mean_squared_error = (weights * squared_error).sum(axis = axis) / squared_error.shape[axis]
        return mean_squared_error
    
    def _deriv_1_wmse(self, axis: int | None = None, weights: Matrix | Tensor | None = None, **kwargs) -> Tensor | Matrix:
        """
        Computes the first-order derivative of the Weighted Mean Squared Error between result and target.
        
        Args:
            axis: None or int, if you intend to get the per-output metrics/derivs, set axis = 0. Else None.
            weights: Matrix or Tensor or None, if None, fail to normal mse.
        
        Returns:
            Tensor | Matrix: Derivative of MSE tensor or matrix with respect to result.
        """
        if weights is None:
            return self._deriv_1_mse(axis = axis, **kwargs)
        
        error = self.result - self.target
        grad = 2 * error / error.shape[0]
        if axis is None:
            grad = 2 * weights * error / np.array(error.shape).prod()
        else:
            grad = 2 * weights * error / error.shape[axis]
        return grad
    
    def _deriv_2_wmse(self, axis: int | None = None, weights: Matrix | Tensor | None = None, **kwargs) -> Tensor | Matrix:
        """
        Computes the second-order derivative (Hessian diagonal) of the Weighted Mean Squared Error between result and target.
    
        Args:
            axis: None or int, if you intend to get the per-output metrics/derivs, set axis = 0. Else None.
            weights: Matrix or Tensor or None, if None, fail to normal mse.
            
        Returns:
            Tensor | Matrix: Constant Hessian of MSE (2/N) with the same shape as result.
        """
        if weights is None:
            return self._deriv_2_mse(axis = axis, **kwargs)
        
        ones = self.result.copy(); ones[...] = 1;
        if axis is None:
            return ones * (2.0 * weights / np.array(self.result.shape).prod())
        else:
            return ones * (2.0 * weights / self.result.shape[axis])
    
    def _compute_wrmse(self, axis: int | None = None, weights: Matrix | Tensor | None = None, **kwargs) -> Tensor | Matrix:
        """
        Computes the Weighted Root Mean Squared Error between result and target.
        
        Args:
            axis: None or int, if you intend to get the per-output metrics/derivs, set axis = 0. Else None.
            weights: Matrix or Tensor or None, if None, fail to normal rmse.
            
        Returns:
            Tensor | Matrix: RMSE tensor or matrix
        """
        if weights is None:
            return self._compute_rmse(axis = axis, **kwargs)
        
        error = (self.result - self.target)
        squared_error = weights * error ** 2
        if axis is None:
            mean_squared_error = squared_error.sum(axis = axis) / np.array(squared_error.shape).prod()
        else:
            mean_squared_error = squared_error.sum(axis = axis) / squared_error.shape[axis]
        rmse = mean_squared_error ** 0.5
        return rmse
    
    def _deriv_1_wrmse(self, axis: int | None = None, weights: Matrix | Tensor | None = None, **kwargs) -> Tensor | Matrix:
        """
        Computes the first-order derivative of the Weighted Root Mean Squared Error between result and target.
        
        Args:
            axis: None or int, if you intend to get the per-output metrics/derivs, set axis = 0. Else None.
            weights: Matrix or Tensor or None, if None, fail to normal rmse.
            
        Returns:
            Tensor | Matrix: Derivative of RMSE tensor or matrix with respect to result.
        """
        if weights is None:
            return self._deriv_1_rmse(axis = axis, **kwargs)
        
        error = self.result - self.target
        squared_error = weights * error ** 2
        if axis is None:
            mean_squared_error = squared_error.sum(axis = axis) / np.array(squared_error.shape).prod()
            rmse = mean_squared_error ** 0.5
            grad = weights * error / (np.array(squared_error.shape).prod() * rmse)
        else:
            mean_squared_error = squared_error.sum(axis = axis) / squared_error.shape[axis]
            rmse = mean_squared_error ** 0.5
            grad = weights * error / (error.shape[axis] * rmse)

        return grad
    
    def _deriv_2_wrmse(self, axis: int | None = None, weights: Matrix | Tensor | None = None, **kwargs) -> Tensor | Matrix:
        """
        Computes the second-order derivative of the Weighted Root Mean Squared Error between result and target.
        
        Args:
            axis: None or int, if you intend to get the per-output metrics/derivs, set axis = 0. Else None.
            weights: Matrix or Tensor or None, if None, fail to normal rmse.
            
        Returns:
            Tensor | Matrix: Second-order derivative (Hessian diagonal) of RMSE with respect to result.
        """
        if weights is None:
            return self._deriv_2_rmse(axis = axis, **kwargs)
        
        error = self.result - self.target
        squared_error = weights * error ** 2
        if axis is None:
            sum_squared = squared_error.sum(axis = axis)
            mean_squared_error = sum_squared / np.array(squared_error.shape).prod()
            rmse = mean_squared_error ** 0.5
            N = np.array(squared_error.shape).prod()
            hessian = weights / (N * rmse) - (weights **2 * error ** 2) / ((N ** 2) * (rmse ** 3))
        else:
            sum_squared = squared_error.sum(axis = axis)
            mean_squared_error = sum_squared / squared_error.shape[axis]
            rmse = mean_squared_error ** 0.5
            N = error.shape[axis] 
            hessian = weights / (N * rmse) - (weights **2 * error ** 2) / ((N ** 2) * (rmse ** 3))
        
        return hessian
    
    def _compute_r2(self, **kwargs) -> Tensor | Matrix:
        """
        Computes the coefficient of determination R^2 between result and target.
        
        Args:
            axis: None or int, if you intend to get the per-output metrics/derivs, set axis = 0. Else None.
        
        Returns:
            Tensor | Matrix: R^2 value.
        """
        # Compute the residual sum of squares (SS_res)
        error = self.result - self.target
        ss_res = (error ** 2.0).sum()
        
        # Compute the total sum of squares (SS_tot)
        target_mean = self.target.sum() / self.target.shape[0]
        total_error = self.target - target_mean
        ss_tot = (total_error ** 2.0).sum()
        
        # Calculate R^2 = 1 - (SS_res / SS_tot)
        r2 = 1 - (ss_res / ss_tot)
        return r2
    
    def _compute_adjusted_r2(self, **kwargs) -> Tensor | Matrix:
        """
        Computes the adjusted R^2 value.
        
        Args:
            axis: None or int, if you intend to get the per-output metrics/derivs, set axis = 0. Else None.
        
        Returns:
            Tensor | Matrix: Adjusted R^2 value.
        """
        # If self.k is None, badly initialized.
        if self.k is None or isinstance(self.k, int) == False:
            raise ValueError("You must specify a valid `k` as the number of parameters in the model before calculating Adjusted R^2.")
        
        # Compute R^2 using the previously defined method.
        r2 = self._compute_r2()
        
        # Determine the number of observations (be the size along the first dimension)
        n = self.target.shape[0]
        
        # Calculate adjusted R^2 using: 1 - (1-R^2)*((n-1)/(n-p-1))
        adjusted_r2 = 1 - ((1 - r2) * ((n - 1) /  (n - self.k - 1)))
        return adjusted_r2
    
    def __repr__(self):
        """
        String representation of the RegressionMetrics instance.
        """
        return f"RegressionMetrics(metric_type={self.metric_type}, shape={self.result.shape})"



# Metrics for classfication (base)
class ClassificationMetrics(BaseMetrics):

    __attr__ = "MML.ClassificationMetrics"    

    def __init__(self):
        super().__init__()
    
    def __repr__(self):
        return "ClassificationMetrics(Abstract Class)."


# Metrics for binary classification
class BinaryClassificationMetrics(ClassificationMetrics):
    """
    A class to compute common binary classification metrics between predicted results and target values.
    
    Supported metrics include:
        - accuracy
        - precision
        - recall (sensitivity) [TPR]
        - f1 score
        - specificity [TNR]
        - auc_roc
        - confusion_matrix
        - tpr (True Positive Rate)
        - tnr (True Negative Rate)
        - fpr (False Positive Rate)
        - fnr (False Negative Rate)
        - logloss (continuous)
    
    The computations are performed using the underlying tensor operations. It is assumed that both 
    the result and target are of the same type (Tensor or Matrix) and support similar operations.
    
    Attributes:
        result: Predicted results tensor or matrix (can be continuous scores or binary labels).
        target: Target binary values tensor or matrix.
        metric_type: A string specifying which metric to compute ('accuracy', 'precision', 'recall',
                     'f1', 'specificity', 'auc_roc', 'confusion_matrix').
        threshold: A float value used to convert continuous scores into binary predictions (default 0.5).
    """
    
    __attr__ = "MML.BinaryClassificationMetrics"   
    
    def __init__(self, result: Tensor | Matrix, target: Tensor | Matrix, metric_type: str = "accuracy", threshold: float = 0.5, **kwargs):
        """
        Initializes the BinaryClassificationMetrics instance with result and target tensors.
        
        Args:
            result (Tensor | Matrix): Predicted results tensor
            target (Tensor | Matrix): Target values tensor
            metric_type (str): Metric type to compute ('accuracy', 'precision', 'recall', 'f1', 'specificity',
                               'auc_roc', 'confusion_matrix', 'tpr', 'tnr', 'fpr', 'fnr', 'logloss')
            threshold (float): a threshold for considering which one to be the positive samples and negative samples.
                               In normal tasks, it is recommended to be 0.5. But adjusting this may change the metrics.
        """
        super().__init__()
        
        # Different instances or different backends.
        if isinstance(result, Object) == False or isinstance(target, Object) == False:
            raise ValueError("Predicted `result` and real `target` should be either `Matrix` or `Tensor` type!")
        if type(result) != type(target):
            raise ValueError("Predicted `result` and real `target` should have the same type, either Tensor or Matrix!")
        if result._backend != target._backend:
            raise ValueError("Predicted `result` and real `target` should have the same backend, either numpy or torch!")
        
        # Data Members.
        self.result = result
        self.target = target
        self.metric_type = metric_type.lower()
        self.threshold = threshold
        
        # Use the type of result as the typeclass.
        self.typeclass = type(result)
        
        if not self.result.shape == self.target.shape:
            raise ValueError("Result and target tensors must have the same shape.")

    def compute(self, **kwargs) -> Matrix | Tensor:
        """
        Computes the specified metric for a given model or data.
    
        Args:
            **kwargs: Other arguments supported by metrics.
            
        Returns:
            Matrix | Tensor: The computed metric value. The result is always returned as a Matrix or Tensor object,
                                      even if the computation yields a scalar.                            
        
        Raises:
            ValueError: If an unsupported metric type is provided.
            
        """
        # Note, all results are stored in a Matrix | Tensor even it is a scalar.
        if self.metric_type == 'accuracy':
            return self._compute_accuracy(**kwargs)
        elif self.metric_type == 'precision':
            return self._compute_precision(**kwargs)
        elif self.metric_type in ('recall', 'sensitivity', 'tpr'):
            return self._compute_recall(**kwargs)
        elif self.metric_type in ('f1', 'f1 score'):
            return self._compute_f1(**kwargs)
        elif self.metric_type in ('specificity', 'tnr'):
            return self._compute_specificity(**kwargs)
        elif self.metric_type == 'fpr':
            return self._compute_fpr(**kwargs)
        elif self.metric_type == 'fnr':
            return self._compute_fnr(**kwargs)
        elif self.metric_type == 'auc_roc':
            return self._compute_auc_roc(**kwargs)
        elif self.metric_type == 'confusion_matrix':
            return self._compute_confusion_matrix(**kwargs)
        elif self.metric_type in ('logloss', 'log-loss', 'entropy', 'cross-entropy'):
            return self._compute_logloss(**kwargs)
        # Implemented by Nathmath Huang.
        else:
            raise ValueError(f"Unsupported metric type: {self.metric_type}")
    
    def deriv_1(self, **kwargs) -> Tensor | Matrix:
        """
        Computes the sample-wise 1st order derivative for a given model or data.
        
        Args:
            **kwargs: Other arguments supported by metrics.
        
        Returns:
            Tensor | Matrix: The computed metric value as a tensor
        """
        # Note, all results are stored in a Matrix | Tensor even it is a scalar.
        if self.metric_type in ('logloss', 'log-loss', 'entropy', 'cross-entropy'):
            return self._deriv_1_logloss(**kwargs)
        elif self.metric_type == 'accuracy':
            raise  ValueError(f"Metric type: {self.metric_type} cannot compute derivatives.")
        elif self.metric_type == 'precision':
            raise  ValueError(f"Metric type: {self.metric_type} cannot compute derivatives.")
        elif self.metric_type in ('recall', 'sensitivity', 'tpr'):
            raise  ValueError(f"Metric type: {self.metric_type} cannot compute derivatives.")
        elif self.metric_type in ('f1', 'f1 score'):
            raise  ValueError(f"Metric type: {self.metric_type} cannot compute derivatives.")
        elif self.metric_type in ('specificity', 'tnr'):
            raise  ValueError(f"Metric type: {self.metric_type} cannot compute derivatives.")
        elif self.metric_type == 'fpr':
            raise  ValueError(f"Metric type: {self.metric_type} cannot compute derivatives.")
        elif self.metric_type == 'fnr':
            raise  ValueError(f"Metric type: {self.metric_type} cannot compute derivatives.")
        elif self.metric_type == 'auc_roc':
            raise  ValueError(f"Metric type: {self.metric_type} cannot compute derivatives.")
        elif self.metric_type == 'confusion_matrix':
            raise  ValueError(f"Metric type: {self.metric_type} cannot compute derivatives.")
        else:
            raise ValueError(f"Unsupported metric type: {self.metric_type}")
    
    def deriv_2(self, **kwargs) -> Tensor | Matrix:
        """
        Computes the sample-wise 2nd order derivative for a given model or data.
        
        Args:
            **kwargs: Other arguments supported by metrics.
        
        Returns:
            Tensor | Matrix: The computed metric value as a tensor
        """
        # Note, all results are stored in a Matrix | Tensor even it is a scalar.
        if self.metric_type in ('logloss', 'log-loss', 'entropy', 'cross-entropy'):
            return self._deriv_2_logloss(**kwargs)
        elif self.metric_type == 'accuracy':
            raise  ValueError(f"Metric type: {self.metric_type} cannot compute derivatives.")
        elif self.metric_type == 'precision':
            raise  ValueError(f"Metric type: {self.metric_type} cannot compute derivatives.")
        elif self.metric_type in ('recall', 'sensitivity', 'tpr'):
            raise  ValueError(f"Metric type: {self.metric_type} cannot compute derivatives.")
        elif self.metric_type in ('f1', 'f1 score'):
            raise  ValueError(f"Metric type: {self.metric_type} cannot compute derivatives.")
        elif self.metric_type in ('specificity', 'tnr'):
            raise  ValueError(f"Metric type: {self.metric_type} cannot compute derivatives.")
        elif self.metric_type == 'fpr':
            raise  ValueError(f"Metric type: {self.metric_type} cannot compute derivatives.")
        elif self.metric_type == 'fnr':
            raise  ValueError(f"Metric type: {self.metric_type} cannot compute derivatives.")
        elif self.metric_type == 'auc_roc':
            raise  ValueError(f"Metric type: {self.metric_type} cannot compute derivatives.")
        elif self.metric_type == 'confusion_matrix':
            raise  ValueError(f"Metric type: {self.metric_type} cannot compute derivatives.")
        else:
            raise ValueError(f"Unsupported metric type: {self.metric_type}")
            
    def _binarize(self, y_real_or_pred: Matrix | Tensor) -> Matrix | Tensor:
        """
        Binarizes continuous prediction scores by applying a threshold.
    
        Args:
            y_real_or_pred (Matrix | Tensor): the y values to be binarized.
    
        Returns:
            Matrix | Tensor: A matrix or tensor containing binary predictions (True/False values).
    
        """
        # Return the results in a Matrix or Tensor of Booleans
        return self.typeclass(y_real_or_pred.data >= self.threshold, backend = y_real_or_pred._backend, device = y_real_or_pred.device)
    
    def _compute_confusion_counts(self, **kwargs):
        """
        Computes the counts of true positives (TP), true negatives (TN), 
                     false positives (FP) and false negatives (FN) using binarized predictions.
    
        Args:
            None
    
        Returns:
            tuple: A tuple containing four elements, each representing TP, TN, FP, and FN respectively.
                  Each element is a matrix or tensor of the same type as self.target.
    
        """
        pred = self._binarize(self.result)  # Full of Booleans.
        real = self._binarize(self.target)  # Full of Booleans.

        TP = ((pred.data == True) & (real.data == True)).sum()
        TN = ((pred.data == False) & (real.data == False)).sum()
        FP = ((pred.data == True) & (real.data == False)).sum()
        FN = ((pred.data == False) & (real.data == True)).sum()
        return (self.typeclass(TP, backend=self.target._backend, dtype=self.target.dtype, device=self.target.device), 
                self.typeclass(TN, backend=self.target._backend, dtype=self.target.dtype, device=self.target.device),
                self.typeclass(FP, backend=self.target._backend, dtype=self.target.dtype, device=self.target.device),
                self.typeclass(FN, backend=self.target._backend, dtype=self.target.dtype, device=self.target.device)
                )

    def _compute_accuracy(self, **kwargs):
        """
        Computes accuracy = (TP + TN) / total.
        
        Args:
            None
    
        Returns:
            Matrix | Tensor: The computed accuracy value.
        
        """
        # Always return a Matrix | Tensor as the class input.
        TP, TN, FP, FN = self._compute_confusion_counts()
        total = TP + TN + FP + FN
        return (TP + TN) / total

    def _compute_precision(self, **kwargs):
        """
        Computes precision = TP / (TP + FP).
    
        Args:
            None
    
        Returns:
            Matrix | Tensor: The computed precision value.
        
        """
        # Always return a Matrix | Tensor as the class input.
        TP, _, FP, _ = self._compute_confusion_counts()
        denom = TP + FP
        if bool(denom.data == 0) == True:
            return self.typeclass(0, backend=self.target._backend, dtype=self.target.dtype, device=self.target.device)
        return TP / denom

    def _compute_recall(self, **kwargs):
        """
        Computes recall (sensitivity) = TP / (TP + FN).
        
        Args:
            None
    
        Returns:
            Matrix | Tensor: The computed recall value.
        
        """
        # Always return a Matrix | Tensor as the class input.
        TP, _, _, FN = self._compute_confusion_counts()
        denom = TP + FN
        if bool(denom.data == 0) == True:
            return self.typeclass(0, backend=self.target._backend, dtype=self.target.dtype, device=self.target.device)
        return TP / denom

    def _compute_f1(self, **kwargs):
        """
        Computes the F1 score as the harmonic mean of precision and recall.
        
        Args:
            None
    
        Returns:
            Matrix | Tensor: The computed f1 score value.
        
        """
        # Always return a Matrix | Tensor as the class input.
        TP, TN, FP, FN = self._compute_confusion_counts()
        denom = 2 * TP + FP + FN
        if bool(denom.data == 0) == True:
            return self.typeclass(0, backend=self.target._backend, dtype=self.target.dtype, device=self.target.device)
        return 2 * TP / denom

    def _compute_specificity(self, **kwargs):
        """
        Computes specificity = TN / (TN + FP).
        
        Args:
            None
    
        Returns:
            Matrix | Tensor: The computed specificity value.
        """
        # Always return a Matrix | Tensor as the class input.
        _, TN, FP, _ = self._compute_confusion_counts()
        denom = TN + FP
        if bool(denom.data == 0) == True:
            return self.typeclass(0, backend=self.target._backend, dtype=self.target.dtype, device=self.target.device)
        return TN / denom

    def _compute_tpr(self, **kwargs):
        """
        Computes recall (TPR) = TP / (TP + FN).
        
        Args:
            None
    
        Returns:
            Matrix | Tensor: The computed TPR value.
        
        """
        # Always return a Matrix | Tensor as the class input.
        TP, _, _, FN = self._compute_confusion_counts()
        denom = TP + FN
        if bool(denom.data == 0) == True:
            return self.typeclass(0, backend=self.target._backend, dtype=self.target.dtype, device=self.target.device)
        return TP / denom

    def _compute_tnr(self, **kwargs):
        """
        Computes specificity (TNR) = TN / (TN + FP).
        
        Args:
            None
    
        Returns:
            Matrix | Tensor: The computed TNR value.
        """
        # Always return a Matrix | Tensor as the class input.
        _, TN, FP, _ = self._compute_confusion_counts()
        denom = TN + FP
        if bool(denom.data == 0) == True:
            return self.typeclass(0, backend=self.target._backend, dtype=self.target.dtype, device=self.target.device)
        return TN / denom

    def _compute_fpr(self, **kwargs):
        """
        Computes FPR = FP / (FP + TN).
        
        Args:
            None
    
        Returns:
            Matrix | Tensor: The computed TPR value.
        
        """
        # Always return a Matrix | Tensor as the class input.
        _, TN, FP, _ = self._compute_confusion_counts()
        denom = FP + TN
        if bool(denom.data == 0) == True:
            return self.typeclass(0, backend=self.target._backend, dtype=self.target.dtype, device=self.target.device)
        return FP / denom

    def _compute_fnr(self, **kwargs):
        """
        Computes FNR = FN / (TP + FN).
        
        Args:
            None
    
        Returns:
            Matrix | Tensor: The computed FNR value.
        
        """
        TP, _, _, FN = self._compute_confusion_counts()
        denom = TP + FN
        if bool(denom.data == 0) == True:
            return self.typeclass(0, backend=self.target._backend, dtype=self.target.dtype, device=self.target.device)
        return FN / denom

    def _compute_auc_roc(self, **kwargs):
        """
        Computes the area under the ROC curve (AUC-ROC) using the trapezoidal rule.
        This method assumes that self.result contains continuous prediction scores.
        
        Args:
            None
    
        Returns:
            Matrix | Tensor: The computed auc_roc area.
        """
        # Always return a Matrix | Tensor as the class input.
        scores = self.result.data
        labels = self.target.data
        
        # Sort indices based on scores in descending order.
        sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        sorted_labels = [labels[i] for i in sorted_indices]
        P = sum(labels)
        N = len(labels) - P
        if P == 0 or N == 0:
            return self.typeclass(0.0, backend=self.target._backend, dtype=self.target.dtype, device=self.target.device)

        tpr, fpr = [], []
        tp = 0
        fp = 0
        for label in sorted_labels:
            if label == 1:
                tp += 1
            else:
                fp += 1
            tpr.append(tp / P)
            fpr.append(fp / N)
        
        auc = 0.0
        prev_fpr = 0.0
        prev_tpr = 0.0
        for current_fpr, current_tpr in zip(fpr, tpr):
            auc += (current_fpr - prev_fpr) * (current_tpr + prev_tpr) / 2.0
            prev_fpr = current_fpr
            prev_tpr = current_tpr
        return self.typeclass(auc, backend=self.target._backend, dtype=self.target.dtype, device=self.target.device)

    def _compute_logloss(self, **kwargs):
        """
        Computes the binary classification log loss between predicted and actual values.
    
        Args: 
            None
            
        Returns:
            Matrix | Tensor: The computed logloss using this formula:
                logloss = - (y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred))
        """
        epsilon = 1e-15
        preds = self.result.to(self.result._backend, dtype=float)
        labels = self.target.to(self.result._backend, dtype=float)
        clipped_preds = preds.clip(epsilon, 1 - epsilon)
        losses = -(labels * clipped_preds.log() + (1 - labels) * (1 - clipped_preds).log())
        return losses.mean()

    def _deriv_1_logloss(self, **kwargs) -> Tensor | Matrix:
        """
        Computes the first-order derivative of the binary classification log loss between predicted and actual values.

        Args: 
            None
            
        Returns:
            Matrix | Tensor: Derivative of logloss with respect to the predictions.
        """
        epsilon = 1e-15
        preds = self.result.to(self.result._backend, dtype=float)
        labels = self.target.to(self.result._backend, dtype=float)
        clipped = preds.clip(epsilon, 1 - epsilon)
        grad = ((1 - labels) / (1 - clipped) - labels / clipped) / clipped.shape[0]
        return grad
    
    def _deriv_2_logloss(self, **kwargs) -> Tensor | Matrix:
        """
        Computes the second-order derivative of the binary classification log loss between predicted and actual values.

        Args: 
            None
            
        Returns:
            Matrix | Tensor: Second-order derivative (Hessian diagonal) of logloss with respect to the predictions.
        """
        epsilon = 1e-15
        preds = self.result.to(self.result._backend, dtype=float)
        labels = self.target.to(self.result._backend, dtype=float)
        clipped = preds.clip(epsilon, 1 - epsilon)
        hess = ((1 - labels) / (1 - clipped) ** 2 + labels / clipped ** 2) / clipped.shape[0]
        return hess

    def _compute_confusion_matrix(self, **kwargs):
        """
        Computes the confusion matrix as a 2x2 tensor or matrix with the format:
          [[TP, FP],
           [FN, TN]]
            
        Args:
            None
    
        Returns:
            Matrix | Tensor: The computed confusion matrix, with shape 2,2.
        """
        TP, TN, FP, FN = self._compute_confusion_counts()
        return self.typeclass(
            [TP.data, FP.data,
             FN.data, TN.data], 
            backend=self.target._backend, dtype=self.target.dtype, device=self.target.device).reshape([2,2])

    def __repr__(self):
        return f"BinaryClassificationMetrics(metric_type={self.metric_type}, shape={self.result.shape})"


# Metrics for multi-class classification
class MultiClassificationMetrics(ClassificationMetrics):
    """
    A class to compute common multi-class classification metrics between predicted results and target values.
    
    Supported metrics include:
        - accuracy
        - precision        (macro-average computed either in one-vs-rest (OVR) or one-vs-one (OVO) mode)
        - recall           (macro-average computed either in OVR or OVO mode)
        - f1 score         (macro-average computed either in OVR or OVO mode)
        - logloss          (cross-entropy loss, continuous)
        - confusion_matrix (of shape [n_classes, n_classes])
        
    The class is designed to support two scenarios:
        1. Multi-target: where predictions are provided as a 1D vector of labels
           (e.g. 0, 1, 2, 3, ...) and the target is also a vector.
        2. One-hot: where the target (and optionally predictions) are provided as a
           one-hot encoded matrix of shape [n_samples, n_classes].
    
    When computing precision, recall, and f1-score, the user can specify
    whether the aggregation should be based on one-vs-rest (default) or one-vs-one.
    
    Attributes:
        result (Tensor | Matrix): Predicted results. Can be either a 1D vector (labels) 
                                   or a 2D matrix (probabilities / one-hot scores). 
        target (Tensor | Matrix): True labels. Must be in a format compatible with result
                                   (either both 1D or both 2D, or convertible between them).
        metric_type (str): Which metric to compute ("accuracy", "precision", "recall",
                           "f1", "logloss", "confusion_matrix").
        mode (str): For metrics that require binary decomposition ("precision",
                    "recall", "f1"), the aggregation mode: either "ovr" (one-vs-rest) or "ovo" (one-vs-one).
    """
    
    __attr__ = "MML.MultiClassificationMetrics"   
    
    def __init__(self, result: Tensor | Matrix, target: Tensor | Matrix, metric_type: str = "accuracy", n_classes: int = None, mode: str = "ovr", **kwargs):
        """
        Initializes the MultiClassificationMetrics instance with result and target tensors.
        
        Args:
            result (Tensor | Matrix): Predicted results tensor
            target (Tensor | Matrix): Target values tensor
            metric_type (str): Metric type to compute ('accuracy', 'precision', 'recall', 'f1', 'confusion_matrix', 'logloss')
            n_classes (int): Number of Classes
            mode (str): `ovr` or `ovo`, one versus remaining or one versus one.
        """
        super().__init__()
        
        # Check type compatibility
        if isinstance(result, Object) == False or isinstance(target,  Object) == False:
            raise ValueError("Predicted `result` and real `target` should be either `Matrix` or `Tensor` type!")
        if type(result) != type(target):
            raise ValueError("Predicted `result` and real `target` should have the same type, either Tensor or Matrix!")
        if result._backend != target._backend:
            raise ValueError("Predicted `result` and real `target` should have the same backend, either numpy or torch!")

        # Data Members.        
        self.result = result
        self.target = target
        self.metric_type = metric_type.lower()
        self.mode = mode.lower()
        self.typeclass = type(result)
        
        # Determine classification format and number of classes.
        # If given, then okay, or infer.
        # If one of the inputs is two-dimensional, we assume the second dimension is the number of classes.
        if n_classes is not None:
            self.n_classes = n_classes
        else:
            if len(result.shape) == 2:
                if result.shape[1] == 1:
                    # Check if it is indeed a binary problem
                    if len(result.flatten().bincount().unique()) == 2:
                        self.n_classes = 2
                    else:
                        self.n_classes = len(result.flatten().bincount().unique())
                        # Not safe, but okay.
                else:
                    self.n_classes = result.shape[1]
            elif len(target.shape) == 2:
                if target.shape[1] == 1:
                    # Check if it is indeed a binary problem
                    if len(target.flatten().unique()) == 2:
                        self.n_classes = 2
                    else:
                        self.n_classes = len(target.flatten().unique())
                        # Not safe, but okay.
                else:
                    self.n_classes = target.shape[1]
            else:
                # Error. The result dimension is not 2?!!
                raise ValueError("The input `result` and `target` do not have a 2-dimension shape. Make sure it is a multi-classification problem. Set n_classes or resize the Matrix | Tensor if you only have one row.")
        
    def compute(self, eps: float = 1e-15, floattype: type = float, **kwargs) -> Matrix | Tensor:
        """
        Computes the specified multi-class metric.
        
        Supported metric_type values (case insensitive):
            - 'accuracy'
            - 'precision'
            - 'recall'
            - 'f1'
            - 'logloss'
            - 'confusion_matrix'
        
        For precision, recall and f1, the results are computed according to the specified mode (ovr or ovo).
        
        Args:
            eps: float, clip value to ensure 0/0 cases.
            floattype: type, the internal precision of calculation.
            **kwargs: Other arguments supported by metrics.
    
        Returns:
            Matrix | Tensor: The computed metric value. The result is always returned as a Matrix or Tensor object,
                                      even if the computation yields a scalar.
        
        Raises:
            ValueError: If an unsupported metric type or mode type is provided.
        """
        
        # Note, all results are stored in a Matrix | Tensor even it is a scalar.
        # Accuracy
        if self.metric_type == 'accuracy':
            return self._compute_accuracy(floattype=floattype, **kwargs)
        # Precision
        elif self.metric_type == 'precision':
            if self.mode == 'ovr':
                return self._compute_precision_ovr(eps=eps, floattype=floattype, **kwargs)
            elif self.mode == 'ovo':
                return self._compute_precision_ovo(eps=eps, floattype=floattype, **kwargs)
            else:
                raise ValueError(f"Unsupported mode for precision: {self.mode}. Use `ovo` or `ovr`.")
        # Recall
        elif self.metric_type in ('recall', 'sensitivity', 'tpr'):
            if self.mode == 'ovr':
                return self._compute_recall_ovr(eps=eps, floattype=floattype, **kwargs)
            elif self.mode == 'ovo':
                return self._compute_recall_ovo(eps=eps, floattype=floattype, **kwargs)
            else:
                raise ValueError(f"Unsupported mode for recall: {self.mode}. Use `ovo` or `ovr`.")
        # F1 Score
        elif self.metric_type in ('f1', 'f1 score'):
            if self.mode == 'ovr':
                return self._compute_f1_ovr(eps=eps, floattype=floattype, **kwargs)
            elif self.mode == 'ovo':
                return self._compute_f1_ovo(eps=eps, floattype=floattype, **kwargs)
            else:
                raise ValueError(f"Unsupported mode for f1: {self.mode}. Use `ovo` or `ovr`.")
        # Cross entropy/logloss
        elif self.metric_type in ('logloss', 'log-loss', 'entropy', 'cross-entropy'):
            return self._compute_logloss(eps=eps, floattype=floattype, **kwargs)
        # Confusion matrix
        elif self.metric_type == 'confusion_matrix':
            return self._compute_confusion_matrix(**kwargs)
        # Implemented by Nathmath Huang.
        else:
            raise ValueError(f"Unsupported metric type: {self.metric_type}")
    
    def deriv_1(self, eps: float = 1e-15, floattype: type = float, **kwargs) -> Matrix | Tensor:
        """
        Computes the specified multi-class element-wise 1st order derivative.
        
        Supported metric_type values (case insensitive):
            - 'logloss'
        
        Args:
            eps: float, clip value to ensure 0/0 cases.
            floattype: type, the internal precision of calculation.
            **kwargs: Other arguments supported by metrics.
    
        Returns:
            Matrix | Tensor: The computed metric value. The result is always returned as a Matrix or Tensor object,
                                      even if the computation yields a scalar.
        
        Raises:
            ValueError: If an unsupported metric type or mode type is provided.
        """
        
        # Note, only "logloss" supports derivatives.
        # Cross entropy/logloss
        if self.metric_type in ('logloss', 'log-loss', 'entropy', 'cross-entropy'):
            return self._deriv_1_logloss(eps=eps, floattype=floattype, **kwargs)
        # Accuracy
        elif self.metric_type == 'accuracy':
            raise  ValueError(f"Metric type: {self.metric_type} cannot compute derivatives.")
        # Precision
        elif self.metric_type == 'precision':
            raise  ValueError(f"Metric type: {self.metric_type} cannot compute derivatives.")
        # Recall
        elif self.metric_type in ('recall', 'sensitivity', 'tpr'):
            raise  ValueError(f"Metric type: {self.metric_type} cannot compute derivatives.")
        # F1 Score
        elif self.metric_type in ('f1', 'f1 score'):
            raise  ValueError(f"Metric type: {self.metric_type} cannot compute derivatives.")

        # Confusion matrix
        elif self.metric_type == 'confusion_matrix':
            raise  ValueError(f"Metric type: {self.metric_type} cannot compute derivatives.")
        
        # Implemented by Nathmath Huang.
        else:
            raise ValueError(f"Unsupported metric type: {self.metric_type}")
    
    def deriv_2(self, eps: float = 1e-15, floattype: type = float, **kwargs) -> Tensor | Matrix:
        """
        Computes the sample-wise 2nd order derivative for a given model or data.
        
        Returns:
            Tensor | Matrix: The computed metric value as a tensor
        """
        # Note, only "logloss" supports derivatives.
        # Cross entropy/logloss
        if self.metric_type in ('logloss', 'log-loss', 'entropy', 'cross-entropy'):
            return self._deriv_2_logloss(eps=eps, floattype=floattype, **kwargs)
        # Accuracy
        elif self.metric_type == 'accuracy':
            raise  ValueError(f"Metric type: {self.metric_type} cannot compute derivatives.")
        # Precision
        elif self.metric_type == 'precision':
            raise  ValueError(f"Metric type: {self.metric_type} cannot compute derivatives.")
        # Recall
        elif self.metric_type in ('recall', 'sensitivity', 'tpr'):
            raise  ValueError(f"Metric type: {self.metric_type} cannot compute derivatives.")
        # F1 Score
        elif self.metric_type in ('f1', 'f1 score'):
            raise  ValueError(f"Metric type: {self.metric_type} cannot compute derivatives.")

        # Confusion matrix
        elif self.metric_type == 'confusion_matrix':
            raise  ValueError(f"Metric type: {self.metric_type} cannot compute derivatives.")
        
        # Implemented by Nathmath Huang.
        else:
            raise ValueError(f"Unsupported metric type: {self.metric_type}")
            
    def _to_labels(self, x: Tensor | Matrix, *, apply_softmax:bool = False) -> Tensor | Matrix:
        """
        Converts predictions or targets into label vectors.
        
        If x has more than one column (i.e. one-hot or probability matrix), it returns
        the index of the maximum value along axis 1. Otherwise, x is assumed already to be a vector.
             
        Args:
            x: Matrix | Tensor: The one-hot or probability matrix.
            apply_softmax: bool, whether to apply softmax before calculating argmax or not.
    
        Returns:
            Matrix | Tensor: The converted Tensor or Matrix in (n_samples, 1) shape.
        """
        # Wide-table: prob or one-hot
        if len(x.shape) > 1 and x.shape[1] > 1:
            # Always keep the dim.
            return x.argmax(axis=1).reshape([-1, 1]) if apply_softmax == False else x.softmax(axis=1).argmax(axis=1).reshape([-1, 1])
        # Narrow table
        else:    
            return x

    def _to_onehot(self, x: Tensor | Matrix, *, binarize = False) -> Tensor | Matrix:
        """
        Converts a label vector into a one-hot encoded matrix of shape [n_samples, n_classes].
        If x is already a matrix with the correct number of columns, it is returned unaltered.
        If x is binary probability input and binarize is False, then will return the probablistic one-hot.
                     
        Args:
            x: Matrix | Tensor: The label-encoded matrix.
    
        Returns:
            Matrix | Tensor: The converted one-hot Tensor or Matrix in (n_samples, n_classes) shape.
        """
        if len(x.shape) == 2 and x.shape[1] == self.n_classes:
            return x
        
        # If binary case, then create a probabilistic one-hot to reduce information loss
        if self.n_classes == 2 and binarize == False:
            onehot_data = type(x).zeros([x.shape[0], 2], backend=x._backend)
            onehot_data[:, 1] = x.flatten()
            onehot_data[:, 0] = 1.0 - onehot_data[:, 1]
            return onehot_data.to(backend=x._backend, device=x.device, dtype=x.dtype)
        
        # Else, do the round
        else:
            # Create one-hot by comparing each element with a range vector.
            range_vec = self.typeclass(np.arange(self.n_classes), backend=x._backend, device=x.device)
            # Reshape x to [n_samples, 1] if necessary
            x_reshaped = x.reshape([x.shape[0], 1])
        
            # Broadcast the comparison: each entry becomes True if equal to the class index.
            onehot_data = x_reshaped.astype(self.result.dtype).round() == range_vec
            # The above one produces a boolean array -> like True, False, True, ...
            #                                                False, True, False, ...
            return onehot_data.to(backend=x._backend, device=x.device, dtype=float)

    def _compute_accuracy(self, *, floattype: type = float, **kwargs):
        """
        Computes accuracy = (# correct predictions) / (# total samples).
                             
        Args:
            floattype: type, the internal precision of calculation
    
        Returns:
            Matrix | Tensor: The computed accuracy value.
        """
        pred_labels = self._to_labels(self.result)
        true_labels = self._to_labels(self.target)
        correct = pred_labels == true_labels
        total = true_labels.shape[0]
        return correct.sum().to(correct._backend, dtype=floattype, device=correct.device) / total
    
    def _compute_confusion_matrix(self, **kwargs):
        """
        Computes the multi-class confusion matrix of shape [n_classes, n_classes],
        where rows correspond to true labels and columns to predicted labels.
        
        Args:
            None
    
        Returns:
            Matrix | Tensor: The computed confusion matrix with shape [n_classes, n_classes].
        """
        # Convert both predictions and targets to label vectors
        pred_labels = self._to_labels(self.result)
        true_labels = self._to_labels(self.target)
        
        # Convert them into one-hot matrices of shape [n_samples, n_classes]
        pred_onehot = self._to_onehot(pred_labels)
        true_onehot = self._to_onehot(true_labels)
        
        # Compute the confusion matrix as: (true_onehot)^T dot (pred_onehot)
        conf_matrix = true_onehot.transpose().dot(pred_onehot)
        return conf_matrix

    def _compute_logloss(self, *, eps: float = 1e-15, floattype: type = float, **kwargs):
        """
        Computes the cross-entropy loss (logloss) for multi-class classification.
        
        Assumes that `result` is a probability matrix of shape [n_samples, n_classes].
        The loss is computed as:
            logloss = - 1/N * sum_over_samples [ sum_over_classes (y_true * log(y_pred)) ]
            
        Args:
            eps: float, clip value to ensure predictions are not going to be log(0)
            floattype: type, the internal precision of calculation
    
        Returns:
            Matrix | Tensor: The computed log loss value.
        """
        if len(self.result.shape) != 2:
            raise ValueError("Logloss metric requires probability predictions with shape [n_samples, n_classes].")
        
        # Ensure predictions are floating point and clip values to avoid log(0)
        preds = self.result.to(self.result._backend, dtype=floattype, device=self.result.device).clip(eps, 1 - eps)

        # Compute elementwise: y_true * log(y_pred), then sum over classes (axis=1) then average over samples.
        true_onehot = self._to_onehot(self._to_labels(self.target))
        losses = -(true_onehot * preds.log()).sum(axis=1)
        return losses.mean()
    
    def _deriv_1_logloss(self, *, eps: float = 1e-15, floattype: type = float, **kwargs) -> Tensor | Matrix:
        """
        Computes the first-order derivative of the cross-entropy loss (logloss) for multi-class classification.
        
        Assumes that `result` is a probability matrix of shape [n_samples, n_classes].
        
        Args:
            eps: float, clip value to ensure predictions are not going to be log(0)
            floattype: type, the internal precision of calculation
    
        Returns:
            Matrix | Tensor: Derivative of the logloss with respect to the predictions.
        """
        if len(self.result.shape) != 2:
            raise ValueError("Logloss metric requires probability predictions with shape [n_samples, n_classes].")
        
        # Ensure predictions are floating point and clip values to avoid log(0)
        preds = self.result.to(self.result._backend, dtype=floattype, device=self.result.device).clip(eps, 1 - eps)

        # Convert to one-hot labels
        true_onehot = self._to_onehot(self._to_labels(self.target))

        # ∂L/∂p = -y/p 
        grad = -(true_onehot / preds) / preds.shape[0]
        return grad
    
    def _deriv_2_logloss(self, *, eps: float = 1e-15, floattype: type = float, **kwargs) -> Tensor | Matrix:
        """
        Computes the second-order derivative of the cross-entropy loss (logloss) for multi-class classification.
        
        Assumes that `result` is a probability matrix of shape [n_samples, n_classes].
        
        Args:
            eps: float, clip value to ensure predictions are not going to be log(0)
            floattype: type, the internal precision of calculation
    
        Returns:
            Matrix | Tensor: Second-order derivative (Hessian diagonal) of the logloss with respect to the predictions.
        """
        if len(self.result.shape) != 2:
            raise ValueError("Logloss metric requires probability predictions with shape [n_samples, n_classes].")
        
        # Ensure predictions are floating point and clip values to avoid log(0)
        preds = self.result.to(self.result._backend, dtype=floattype, device=self.result.device).clip(eps, 1 - eps)

        # Convert to one-hot labels
        true_onehot = self._to_onehot(self._to_labels(self.target))
        
        hess = (true_onehot / preds ** 2) / preds.shape[0]
        return hess

    # OVR (One-Vs-Remaining) implementations for precision, recall and f1

    def _compute_precision_ovr(self, *, eps: float = 1e-15, floattype: type = float, **kwargs):
        """
        Computes macro-average precision using a one-vs-rest approach.
        
        For each class c:
            precision[c] = TP[c] / (TP[c] + FP[c])
        and the final metric is the mean over classes.
        
        Args:
            eps: float, clip value to ensure 0/0 cases.
            floattype: type, the internal precision of calculation.
    
        Returns:
            Matrix | Tensor: The computed precision value.
        """
        pred_labels = self._to_labels(self.result)
        true_labels = self._to_labels(self.target)
        pred_onehot = self._to_onehot(pred_labels.astype(self.result.dtype))
        true_onehot = self._to_onehot(true_labels.astype(self.target.dtype))
        
        # True positives: elementwise multiplication then sum over samples (axis=0)
        TP = (true_onehot * pred_onehot).sum(axis=0)
       
        # False positives: predicted positive but not truly positive.
        FP = ((self.typeclass.ones_like(true_onehot, backend=true_onehot._backend) - true_onehot) * pred_onehot).sum(axis=0)
        precision_per_class = TP / (TP + FP + floattype(eps))
        return precision_per_class.mean()

    def _compute_recall_ovr(self, *, eps: float = 1e-15, floattype: type = float, **kwargs):
        """
        Computes macro-average recall (sensitivity) using a one-vs-rest approach.
        
        For each class c:
            recall[c] = TP[c] / (TP[c] + FN[c])
        and the final metric is the mean over classes.
        
        Args:
            eps: float, clip value to ensure 0/0 cases.
            floattype: type, the internal precision of calculation.
    
        Returns:
            Matrix | Tensor: The computed recall value.
        """
        pred_labels = self._to_labels(self.result)
        true_labels = self._to_labels(self.target)
        pred_onehot = self._to_onehot(pred_labels.astype(self.result.dtype))
        true_onehot = self._to_onehot(true_labels.astype(self.target.dtype))
        
        # True positives: elementwise multiplication then sum over samples (axis=0)
        TP = (true_onehot * pred_onehot).sum(axis=0)
        
        # False negatives: predicted negative but not trully negative.
        FN = (true_onehot * (self.typeclass.ones_like(pred_onehot, backend=pred_onehot._backend) - pred_onehot)).sum(axis=0)
        recall_per_class = TP / (TP + FN + floattype(eps))
        return recall_per_class.mean()

    def _compute_f1_ovr(self, *, eps: float = 1e-15, floattype: type = float, **kwargs):
        """
        Computes macro-average F1-score in one-vs-rest mode.
        F1 per class is computed as:
            F1[c] = 2 * precision[c] * recall[c] / (precision[c] + recall[c])
        The final score is the average over classes.
        
        Args:
            eps: float, clip value to ensure 0/0 cases.
            floattype: type, the internal precision of calculation.
    
        Returns:
            Matrix | Tensor: The computed recall value.
        """
        # Compute per-class precision and recall in OVR mode.
        pred_labels = self._to_labels(self.result)
        true_labels = self._to_labels(self.target)
        pred_onehot = self._to_onehot(pred_labels.astype(self.result.dtype))
        true_onehot = self._to_onehot(true_labels.astype(self.target.dtype))
        
        TP = (true_onehot * pred_onehot).sum(axis=0)
        FP = ((self.typeclass.ones_like(true_onehot, backend=true_onehot._backend) - true_onehot) * pred_onehot).sum(axis=0)
        FN = (true_onehot * (self.typeclass.ones_like(pred_onehot, backend=pred_onehot._backend) - pred_onehot)).sum(axis=0)
        
        precision_per_class = TP / (TP + FP + floattype(eps))
        recall_per_class = TP / (TP + FN + floattype(eps))
        f1_per_class = (2 * precision_per_class * recall_per_class) / (precision_per_class + recall_per_class + floattype(eps))
        return f1_per_class.mean()

    # OVO (One-Vs-One) implementations for precision, recall and f1
    #
    # These computations use the full confusion matrix. For every pair of different classes
    # (i, j), we define binary precision and recall:
    #   For class i in pair (i,j):
    #       precision_i = M[i,i] / (M[i,i] + M[j,i] + eps)
    #       recall_i = M[i,i] / (M[i,i] + M[i,j] + eps)
    #   Similarly for class j.
    # The final OVO metric is computed as the average over all the binary evaluations.
    
    def _compute_precision_ovo(self, *, eps: float = 1e-15, floattype: type = float, **kwargs):
        """
        Computes macro-average precision using a ovo approach.
        
        Args:
            eps: float, clip value to ensure 0/0 cases.
            floattype: type, the internal precision of calculation.
    
        Returns:
            Matrix | Tensor: The computed precision value.
        """
        conf_matrix = self._compute_confusion_matrix()  # shape: [n_classes, n_classes]
        
        # Create index matrices using broadcasting.
        idx = self.typeclass(np.arange(self.n_classes), backend=conf_matrix._backend, device=conf_matrix.device)
        I = idx.reshape([self.n_classes, 1]).repeat(self.n_classes, axis=1)
        J = idx.reshape([1, self.n_classes]).repeat(self.n_classes, axis=0)
        mask = I.data < J.data  # boolean mask selecting one instance per unordered pair
                                # Internal type
        
        # Extract diagonal elements as a vector.
        diag = conf_matrix.diag()  # shape [n_classes]
        
        # Expand diagonals for broadcasting.
        diag_i = diag.reshape([self.n_classes, 1]).repeat(self.n_classes, axis=1)  # each row: diag[i]
        diag_j = diag.reshape([1, self.n_classes]).repeat(self.n_classes, axis=0)  # each column: diag[j]
        
        # For a given pair (i, j):
        # precision for class i:
        p_i_matrix = diag_i / (diag_i + conf_matrix.transpose())  
        
        # We need M[j, i] for p_i. In our matrix, conf_matrix[j,i] is given by
        # conf_matrix.transpose()[i,j]. Thus, we use:
        p_i_matrix = diag_i / (diag_i + conf_matrix.transpose() + floattype(eps))
        
        # And precision for class j:
        p_j_matrix = diag_j / (diag_j + conf_matrix + floattype(eps))
        
        # Now select only entries for pairs where I < J.
        p_i_vals = p_i_matrix[mask]
        p_j_vals = p_j_matrix[mask]
        
        # Concatenate and compute the mean.
        all_precisions = p_i_vals.append(p_j_vals, axis=0)
        return all_precisions.mean()

    def _compute_recall_ovo(self, *, eps: float = 1e-15, floattype: type = float, **kwargs):
        """
        Computes macro-average recall using a ovo approach.
        
        Args:
            eps: float, clip value to ensure 0/0 cases.
            floattype: type, the internal precision of calculation.
    
        Returns:
            Matrix | Tensor: The computed recall value.
        """
        conf_matrix = self._compute_confusion_matrix()  # shape: [n_classes, n_classes]
        
        # Create index matrices using broadcasting.
        idx = self.typeclass(np.arange(self.n_classes), backend=conf_matrix._backend, device=conf_matrix.device)
        I = idx.reshape([self.n_classes, 1]).repeat(self.n_classes, axis=1)
        J = idx.reshape([1, self.n_classes]).repeat(self.n_classes, axis=0)
        mask = I.data < J.data  # boolean mask selecting one instance per unordered pair
                                # Internal type
        
        # Extract diagonal elements as a vector.
        diag = conf_matrix.diag()
        diag_i = diag.reshape([self.n_classes, 1]).repeat(self.n_classes, axis=1)
        diag_j = diag.reshape([1, self.n_classes]).repeat(self.n_classes, axis=0)

        # For recall in a pair (i, j):
        # recall for class i:
        r_i_matrix = diag_i / (diag_i + conf_matrix +  floattype(eps))
        # and recall for class j:
        r_j_matrix = diag_j / (diag_j + conf_matrix.transpose() + floattype(eps))
        
        # Now select only entries for pairs where I < J.
        r_i_vals = r_i_matrix[mask]
        r_j_vals = r_j_matrix[mask]
        
        # Concatenate and compute the mean.
        all_recalls = r_i_vals.append(r_j_vals, axis=0)
        return all_recalls.mean()

    def _compute_f1_ovo(self, *, eps: float = 1e-15, floattype: type = float, **kwargs):
        """
        Computes macro-average f1-score using a ovo approach.
        
        Args:
            eps: float, clip value to ensure 0/0 cases.
            floattype: type, the internal precision of calculation.
    
        Returns:
            Matrix | Tensor: The computed f1-score.
        """
        
        # First compute the binary precisions and recalls from OVO.
        precision_ovo = self._compute_precision_ovo(eps=eps, floattype=floattype)
        recall_ovo = self._compute_recall_ovo(eps=eps, floattype=floattype)
        f1_ovo = (2 * precision_ovo * recall_ovo) / (precision_ovo + recall_ovo + floattype(eps))
        return f1_ovo

    def __repr__(self):
        return (f"MultiClassificationMetrics(metric_type={self.metric_type}, mode={self.mode}, "
                f"n_classes={self.n_classes}, result_shape={self.result.shape})")


# Some notes heere.
# Since all kinds of variants of metrics are child classes of BaseMetrics,
# in some applications like Decision Trees or Neural Network that needs metircs
# to do optimizations, you may require and check whether it belongs to a child class
# of the base class or not.

# Test suites
def regression_metrics_test():
    
    import numpy as np
    
    # Create example tensors
    result = Matrix(np.array([1.5, 2.77, 5.98, 7.0, 9.0, 110.0, 114.0]), backend="torch")
    target = Matrix(np.array([2.0, 4.0, 6.0, 8.0, 11.0, 114.0, 114.514]), backend="torch")
    weights= Matrix([0.1,0.1,0.1,0.1,0.1,0.1,0.3], backend="torch")
    
    # A new set of results
    result = Matrix(np.array([[252, 237.0, 17.2, 84.4],
                              [127, 123.6, 13.28, 1.1],
                              [251, 224.5, 96.26, 2.9]]), backend="torch")
    target = Matrix(np.array([[247, 236.9, 17.2, 95.0],
                              [128, 124.1, 14.4, 1.22],
                              [256, 192.9, 98.09,3.11]]), backend="torch")
    weights= Matrix([[0.1],[0.3],[0.6]], backend="torch")
    
    
    # Compute metrics
    mtype = 'mse'
    metrics = RegressionMetrics(result, target, metric_type=mtype)
    print(mtype + " Metric : ", metrics.compute())
    print(mtype + " Grads  : ", metrics.deriv_1())
    print(mtype + " Hessian: ", metrics.deriv_2())
    
    mtype = 'wmse'
    metrics = RegressionMetrics(result, target, metric_type=mtype)
    print(mtype + " Metric : ", metrics.compute(weights=weights))
    print(mtype + " Grads  : ", metrics.deriv_1(weights=weights))
    print(mtype + " Hessian: ", metrics.deriv_2(weights=weights))
    
    mtype = 'rmse'
    metrics = RegressionMetrics(result, target, metric_type=mtype)
    print(mtype + " Metric : ", metrics.compute())
    print(mtype + " Grads  : ", metrics.deriv_1())
    print(mtype + " Hessian: ", metrics.deriv_2())
    
    mtype = 'wrmse'
    metrics = RegressionMetrics(result, target, metric_type=mtype)
    print(mtype + " Metric : ", metrics.compute(weights=weights))
    print(mtype + " Grads  : ", metrics.deriv_1(weights=weights))
    print(mtype + " Hessian: ", metrics.deriv_2(weights=weights))
    
    mtype = 'mae'
    metrics = RegressionMetrics(result, target, metric_type=mtype)
    print(mtype + " Metric : ", metrics.compute())
    print(mtype + " Grads  : ", metrics.deriv_1())
    print(mtype + " Hessian: ", metrics.deriv_2())
    
    mtype = 'mape'
    metrics = RegressionMetrics(result, target, metric_type=mtype)
    print(mtype + " Metric : ", metrics.compute())
    print(mtype + " Grads  : ", metrics.deriv_1())
    print(mtype + " Hessian: ", metrics.deriv_2())
    
    mtype = 'huber_loss'
    metrics = RegressionMetrics(result, target, metric_type=mtype)
    print(mtype + " Metric : ", metrics.compute())
    print(mtype + " Grads  : ", metrics.deriv_1())
    print(mtype + " Hessian: ", metrics.deriv_2())
    

def binary_classification_test():
    import numpy as np
    
    # Define predicted scores and binary target values.
    scores = Matrix(np.array([0.9, 0.3, 0.8, 0.4, 0.6, 0.3]), "torch", dtype=float)
    targets = Matrix(np.array([ 1,   0,   1,   0,   1,   0]), "torch", dtype=float)
    
    # Accuracy test (using threshold 0.5).
    cm_accuracy = BinaryClassificationMetrics(scores, targets, 'accuracy', threshold=0.5)
    print("Accuracy:", cm_accuracy.compute())

    # Precision test.
    cm_precision = BinaryClassificationMetrics(scores, targets, 'precision', threshold=0.5)
    print("Precision:", cm_precision.compute())
    
    # Recall test.
    cm_recall = BinaryClassificationMetrics(scores, targets, 'recall', threshold=0.5)
    print("Recall:", cm_recall.compute())
    
    # F1 score test.
    cm_f1 = BinaryClassificationMetrics(scores, targets, 'f1', threshold=0.5)
    print("F1 Score:", cm_f1.compute())
    
    # Specificity test.
    cm_specificity = BinaryClassificationMetrics(scores, targets, 'specificity', threshold=0.5)
    print("Specificity:", cm_specificity.compute())
    
    # AUC-ROC test (note: scores are used as continuous values).
    cm_auc = BinaryClassificationMetrics(scores, targets, 'auc_roc')
    print("AUC-ROC:", cm_auc.compute())
    
    # Confusion matrix test.
    cm_confusion = BinaryClassificationMetrics(scores, targets, 'confusion_matrix', threshold=0.5)
    print("Confusion Matrix:", cm_confusion.compute())
    
    # Compute FPR
    cm_fpr = BinaryClassificationMetrics(scores, targets, 'fpr', threshold=0.5)
    print("FPR:", cm_fpr.compute())
    
    # Compute FNR
    cm_fnr = BinaryClassificationMetrics(scores, targets, 'fnr', threshold=0.5)
    print("FNR:", cm_fnr.compute())
    
    # Compute Logloss
    cm_logloss = BinaryClassificationMetrics(scores, targets, 'logloss')
    print("Logloss:", cm_logloss.compute())


def multi_classification_test():
    import numpy as np
    
    # Predicted Scores Matrix for 8 samples and 3 classes.
    scores = Matrix(np.array([
        [0.7, 0.2, 0.1],  # Sample 1
        [0.4, 0.5, 0.1],  # Sample 2
        [0.2, 0.3, 0.5],  # Sample 3
        [0.3, 0.4, 0.3],  # Sample 4 (error: true class is 0 but predicted 1)
        [0.2, 0.2, 0.6],  # Sample 5 (error: true class is 1 but predicted 2)
        [0.5, 0.3, 0.2],  # Sample 6 (error: true class is 2 but predicted 0)
        [0.3, 0.6, 0.1],  # Sample 7
        [0.1, 0.3, 0.6]   # Sample 8
    ]), "torch", dtype=float)
    
    # True Targets Matrix in one-hot encoding for 8 samples and 3 classes.
    targets = Matrix(np.array([
        [1, 0, 0],  # Sample 1: True class 0
        [0, 1, 0],  # Sample 2: True class 1
        [0, 0, 1],  # Sample 3: True class 2
        [1, 0, 0],  # Sample 4: True class 0 (mismatch with predicted)
        [0, 1, 0],  # Sample 5: True class 1 (mismatch with predicted)
        [0, 0, 1],  # Sample 6: True class 2 (mismatch with predicted)
        [0, 1, 0],  # Sample 7: True class 1
        [0, 0, 1]   # Sample 8: True class 2
    ]), "torch", dtype=float)
    
    # -------------------------
    # Accuracy Test
    # -------------------------
    cm_accuracy = MultiClassificationMetrics(scores, targets, metric_type='accuracy')
    print("Accuracy:", cm_accuracy.compute())
    
    # -------------------------
    # Precision Tests
    # -------------------------
    # Precision using one-vs-rest (OVR) aggregation.
    cm_precision_ovr = MultiClassificationMetrics(scores, targets, metric_type='precision', mode='ovr')
    print("Precision (OVR):", cm_precision_ovr.compute())
    
    # Precision using one-vs-one (OVO) aggregation.
    cm_precision_ovo = MultiClassificationMetrics(scores, targets, metric_type='precision', mode='ovo')
    print("Precision (OVO):", cm_precision_ovo.compute())
    
    # -------------------------
    # Recall Tests
    # -------------------------
    # Recall using one-vs-rest (OVR) aggregation.
    cm_recall_ovr = MultiClassificationMetrics(scores, targets, metric_type='recall', mode='ovr')
    print("Recall (OVR):", cm_recall_ovr.compute())
    
    # Recall using one-vs-one (OVO) aggregation.
    cm_recall_ovo = MultiClassificationMetrics(scores, targets, metric_type='recall', mode='ovo')
    print("Recall (OVO):", cm_recall_ovo.compute())
    
    # -------------------------
    # F1 Score Tests
    # -------------------------
    # F1 Score using one-vs-rest (OVR) aggregation.
    cm_f1_ovr = MultiClassificationMetrics(scores, targets, metric_type='f1', mode='ovr')
    print("F1 Score (OVR):", cm_f1_ovr.compute())
    
    # F1 Score using one-vs-one (OVO) aggregation.
    cm_f1_ovo = MultiClassificationMetrics(scores, targets, metric_type='f1', mode='ovo')
    print("F1 Score (OVO):", cm_f1_ovo.compute())
    
    # -------------------------
    # Logloss (Cross-entropy) Test
    # -------------------------
    cm_logloss = MultiClassificationMetrics(scores, targets, metric_type='logloss')
    print("Logloss:", cm_logloss.compute())
    print("Logloss Gradiant:", cm_logloss.deriv_1())
    print("Logloss Hessian :", cm_logloss.deriv_2())
    
    # -------------------------
    # Confusion Matrix Test
    # -------------------------
    cm_confusion = MultiClassificationMetrics(scores, targets, metric_type='confusion_matrix')
    print("Confusion Matrix:\n", cm_confusion.compute())


# Test cases
if __name__ == "__main__":
    
    regression_metrics_test()
    binary_classification_test()
    multi_classification_test()
    