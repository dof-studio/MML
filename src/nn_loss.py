# nn_loss.py
#
# Neural Network Loss Function Collection
# From MML Library by Nathmath

import numpy as np
try:
    import torch
except ImportError:
    torch = None

from typing import Any, Literal

from objtyp import Object
from tensor import Tensor

from nn_parameter import nn_Parameter
from nn_module import nn_Module


# Implementation of Base Lose Class
class nn_Loss_BaseLoss(nn_Module):

    __attr__ = "MML.nn_Loss_BaseLoss"
    
    def __init__(self,
                 *,
                 module_name: str = "nn_Loss_Base",
                 backend: Literal["torch", "numpy"] = "torch",
                 dtype: type | str = None,
                 device: str | None = None,
                 autograd: bool = False,
                 **kwargs):
        """
        An abstract Loss Implemetation.

        Parameters:
            module_name: str, The name of the module instance.
            backend: Literal["torch", "numpy"], The computational backend to use. Defaults to "torch".
            dtype: type, The data type for the tensor values. Defaults to None (auto detection). 
                    For PyTorch, this corresponds to torch.dtype; for NumPy, it corresponds to np.dtype.
            device: str | None, The target device (e.g., "cpu", "cuda") where the layer's parameters will be placed. 
                    If None, uses the default device. Defaults to None (auto detection).
            autograd: bool, A flag indicating whether to use PyTorch's autograd for gradient computation. 
                    If True, manual gradient tracking is disabled. Defaults to False.

        Attributes:
            self.dtype: type, The data type of the tensor values.
            self.device: str | None, The device where the parameters are placed.
            self.autograd: bool, Whether PyTorch's autograd is enabled for this layer.
            self._parameters.weight: nn_Parameter, The weight parameter matrix (shape: [in_features, out_features]).
            self._parameters.bias: nn_Parameter | optional, The optional bias vector (shape: [out_features]) if has_bias is True.
        """

        super().__init__(module_name=module_name, backend=backend, dtype=dtype, device=device, autograd=autograd)

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """
        Forward pass, to calculate the loss.
        """
        raise NotImplementedError("forward() is not implemented in the base loss class")
        
    def backward(self, grad_output: Tensor | None = None) -> Tensor | None:
        """
        Backward pass, to calculate the chained gradient with respect to parameters and return 
        the gradients with respect to inputs.
        """
        raise NotImplementedError("backward() is not implemented in the base loss class")
        
    def __repr__(self):
        return "nn_Loss_BaseLoss(Abstract Loss Class)."   


# Implementation of Mean Square Error Loss
class nn_Loss_MSE(nn_Loss_BaseLoss):
    """
    Mean Squared Error Loss.

    The Mean Squared Error (MSE) loss function quantifies the average squared difference 
    between predicted values and true values. It is widely used in regression tasks 
    and is defined as: 

    Formula: L = \frac{1}{n} \sum_{i=1}^{n} (y_{true,i} - y_{pred,i})^2

    Where:
        - $ n $ is the number of samples
        - $ y_{true,i} $ is the true value for sample i
        - $ y_{pred,i} $ is the predicted value for sample i

    MSE penalizes larger errors more heavily due to squaring, making it sensitive 
    to outliers. It is differentiable and computationally efficient, but not suitable 
    for classification tasks where probabilistic outputs are required.

    Formula: L = \frac{1}{n} \sum_{i=1}^{n} (y_{true,i} - y_{pred,i})^2

    """

    __attr__ = "MML.nn_Loss_MSE"

    def __init__(self,
                 *,
                 module_name: str = "nn_Loss_MSE",
                 backend: Literal["torch", "numpy"] = "torch",
                 dtype: type | str = None,
                 device: str | None = None,
                 autograd: bool = False,
                 **kwargs):
        """
        A Mean Squared Error Loss Function.

        Parameters:
            module_name: str, The name of the module instance.
            backend: Literal["torch", "numpy"], The computational backend to use. Defaults to "torch".
            dtype: type, The data type for the tensor values. Defaults to None (auto detection). 
                    For PyTorch, this corresponds to torch.dtype; for NumPy, it corresponds to np.dtype.
            device: str | None, The target device (e.g., "cpu", "cuda") where the layer's parameters will be placed. 
                    If None, uses the default device. Defaults to None (auto detection).
            autograd: bool, A flag indicating whether to use PyTorch's autograd for gradient computation. 
                    If True, manual gradient tracking is disabled. Defaults to False.

        Attributes:
            self.dtype: type, The data type of the tensor values.
            self.device: str | None, The device where the parameters are placed.
            self.autograd: bool, Whether PyTorch's autograd is enabled for this layer.
            self._parameters.weight: nn_Parameter, The weight parameter matrix (shape: [in_features, out_features]).
            self._parameters.bias: nn_Parameter | optional, The optional bias vector (shape: [out_features]) if has_bias is True.
        """

        super().__init__(module_name=module_name, backend=backend, dtype=dtype, device=device, autograd=autograd)

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """
        Apply the Mean Squared Error (MSE) Loss function to evaluate the values predicted by the network.

        This method computes the average squared difference between predicted values (`pred`) and actual values (`target`), 
        which is a common loss function for regression tasks. The output is a scalar value representing the loss, 
        averaged over all elements in the input tensors.

        Args:
            pred (Tensor): Predicted tensor containing model outputs.
            target (Tensor): Target tensor containing ground truth values.

        Returns:
            Tensor: Scalar tensor representing the computed MSE loss.

        Raises:
            ValueError: If `pred` or `target` is not a valid MML.Tensor object, 
                        or if `pred` and `target` do not have the same shape.

        Attributes:
            self.numel (scalar): Total number of elements in the input tensors (product of tensor shapes).
            self.mse (Tensor): Saved MSE Computed fpr backward computation.
            self.pred (Tensor): Saved predicted tensor for backward computation.
            self.target (Tensor): Saved target tensor for backward computation.
            self.loss ([Tensor]): Save the computed loss for backpropagation uses.
        """

        # Type check, pred and target must be an instance of Tensor
        if isinstance(pred, Tensor) == False or isinstance(target, Tensor) == False:
            raise ValueError(f"In performing forward(), input `pred` or `target` must be in a MML `Tensor` format but you have {type(pred)} and {type(target)}")

        # Shape check, pred and target must have the same shape
        if pred.shape != target.shape:
            raise ValueError(f"In performing forward(), input `pred` and `target` must have the same shape, but you have {pred.shape} and {target.shape}")

        # Compute the MSE loss
        mse = ((pred - target) ** 2).mean()

        # Save the pred, input, mse, and total number of elements for backward
        self.__setattr__("numel", np.array(pred.shape).prod())
        self.__setattr__("mse", mse)
        self.__setattr__("pred", pred)
        self.__setattr__("target", target)
        self.__setattr__("loss", [mse])

        return mse

    def backward(self, grad_output: Tensor | None = None) -> Tensor | None:
        """
        Backward pass: compute gradients for weight, bias, and input.

        This method performs the gradient computation for a module during backpropagation. 
        It calculates the gradients of the loss with respect to the weights, biases, and input tensor,
        based on the provided `grad_output` (gradient from the next layer). The implementation
        supports both PyTorch autograd and manual gradient calculation modes.

        Args:
            None: Since it is the first in calculating backward.

        Returns:
            Tensor: Gradient with respect to the input tensor, for recursive backward calculations in previous layers.

        """

        # If use autograd, pass; if manual mode, then calculate
        if self.autograd == True:
            self.loss[0].data.backward()
            return None

        # If grad_output is None (by default), assign it to 1
        if grad_output is None:
            grad_output = Tensor(1.0, backend=self.backend, dtype=self.dtype, device=self.device)
        
        # Else, it must be a scalar.
        else:
            if isinstance(grad_output, Tensor) == False:
                raise ValueError(f"In performing backward(), input `grad_output` must be in a MML `Tensor` format but you have {type(grad_output)}")
            if len(grad_output.shape) != 0:
                raise ValueError("In performing backward(), input `grad_output` must be in a MML `Tensor` with a scalar stored in")

        # dL/dpred = 2*(pred - target) / N (N = number of elements)
        grad_input = 2 * (self.pred - self.target) / self.numel
        return grad_input * grad_output

    def __repr__(self):
        return "nn_Loss_MSE(Mean Square Error Loss)."


# Alias for nn_Loss_MSE
MSE = nn_Loss_MSE


# Implementation of Root Mean Square Error Loss
class nn_Loss_RMSE(nn_Loss_BaseLoss):
    """
    Root Mean Squared Error Loss.

    The Root Mean Squared Error (RMSE) is the square root of the Mean Squared Error (MSE), 
    providing a measure of the magnitude of errors in the same units as the target variable. 
    It is widely used for evaluating regression models and is defined as:

    Formula: L = \sqrt{ \frac{1}{n} \sum_{i=1}^{n} (y_{true,i} - y_{pred,i})^2 }

    Where:
        - $ n $ is the number of samples
        - $ y_{true,i} $ is the true value for sample i
        - $ y_{pred,i} $ is the predicted value for sample i

    RMSE addresses the interpretability limitation of MSE by scaling the error metric to the same units 
    as the target variable. It retains the sensitivity to outliers from squaring but offers a more intuitive 
    interpretation compared to MSE. Like MSE, it is differentiable and computationally efficient.

    Formula: L = \sqrt{ \frac{1}{n} \sum_{i=1}^{n} (y_{true,i} - y_{pred,i})^2 }

    """

    __attr__ = "MML.nn_Loss_RMSE"

    def __init__(self,
                 *,
                 module_name: str = "nn_Loss_RMSE",
                 backend: Literal["torch", "numpy"] = "torch",
                 dtype: type | str = None,
                 device: str | None = None,
                 autograd: bool = False,
                 **kwargs):
        """
        A Root Mean Squared Error Loss Function.

        Parameters:
            module_name: str, The name of the module instance. Defaults to "nn_Loss_RMSE".
            backend: Literal["torch", "numpy"], The computational backend to use. Defaults to "torch".
            dtype: type, The data type for the tensor values. Defaults to None (auto detection). 
                    For PyTorch, this corresponds to torch.dtype; for NumPy, it corresponds to np.dtype.
            device: str | None, The target device (e.g., "cpu", "cuda") where the layer's parameters will be placed. 
                    If None, uses the default device. Defaults to None (auto detection).
            autograd: bool, A flag indicating whether to use PyTorch's autograd for gradient computation. 
                    If True, manual gradient tracking is disabled. Defaults to False.

        Attributes:
            self.dtype: type, The data type of the tensor values.
            self.device: str | None, The device where the parameters are placed.
            self.autograd: bool, Whether PyTorch's autograd is enabled for this layer.
            self._parameters.weight: nn_Parameter, The weight parameter matrix (shape: [in_features, out_features]).
            self._parameters.bias: nn_Parameter | optional, The optional bias vector (shape: [out_features]) if has_bias is True.
        """

        super().__init__(module_name=module_name, backend=backend, dtype=dtype, device=device, autograd=autograd)

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """
        Compute the Root Mean Squared Error (RMSE) between predictions and targets.

        This method evaluates the square root of the average squared difference 
        between predicted values (`pred`) and actual values (`target`), providing 
        a loss measure in the same units as the original data.

        Args:
            pred (Tensor): Predicted tensor containing model outputs.
            target (Tensor): Target tensor containing ground truth values.

        Returns:
            Tensor: Scalar tensor representing the computed MSE loss.

        Raises:
            ValueError: If `pred` or `target` is not a valid MML.Tensor object, 
                        or if `pred` and `target` do not have the same shape.

        Attributes:
            self.numel (scalar): Total number of elements in the input tensors (product of tensor shapes).
            self.mse (Tensor): Saved MSE Computed fpr backward computation.
            self.rmse (Tensor): Saved RMSE value for use in gradient calculation.
            self.pred (Tensor): Saved predicted tensor for backward computation.
            self.target (Tensor): Saved target tensor for backward computation.
            self.loss (Tensor): Save the computed loss for backpropagation uses.
        """

        # Type check, pred and target must be an instance of Tensor
        if isinstance(pred, Tensor) == False or isinstance(target, Tensor) == False:
            raise ValueError(f"In performing forward(), input `pred` or `target` must be in a MML `Tensor` format but you have {type(pred)} and {type(target)}")

        # Shape check, pred and target must have the same shape
        if pred.shape != target.shape:
            raise ValueError(f"In performing forward(), input `pred` and `target` must have the same shape, but you have {pred.shape} and {target.shape}")

        # Compute MSE and RMSE in Tensor
        mse = ((pred - target) ** 2).mean()
        rmse = mse ** 0.5

        # Save the pred, input, mse, rmse, and total number of elements for backward
        self.__setattr__("numel", np.array(pred.shape).prod())
        self.__setattr__("mse", mse)
        self.__setattr__("rmse", rmse)
        self.__setattr__("pred", pred)
        self.__setattr__("target", target)
        self.__setattr__("loss", [rmse])

        return rmse

    def backward(self, grad_output: Tensor | None = None) -> Tensor | None:
        """
        Backward pass: compute gradients for weight, bias, and input.

        This method performs the gradient computation for a module during backpropagation. 
        It calculates the gradients of the loss with respect to the weights, biases, and input tensor,
        based on the provided `grad_output` (gradient from the next layer). The implementation
        supports both PyTorch autograd and manual gradient calculation modes.

        Args:
            None: Since it is the first in calculating backward.

        Returns:
            Tensor: Gradient with respect to the input tensor, for recursive backward calculations in previous layers.

        """

        # If use autograd, pass; if manual mode, then calculate
        if self.autograd == True:
            self.loss[0].data.backward()
            return None
        
        # If grad_output is None (by default), assign it to 1
        if grad_output is None:
            grad_output = Tensor(1.0, backend=self.backend, dtype=self.dtype, device=self.device)
        
        # Else, it must be a scalar.
        else:
            if isinstance(grad_output, Tensor) == False:
                raise ValueError(f"In performing backward(), input `grad_output` must be in a MML `Tensor` format but you have {type(grad_output)}")
            if len(grad_output.shape) != 0:
                raise ValueError("In performing backward(), input `grad_output` must be in a MML `Tensor` with a scalar stored in")

        # dL/dpred = (pred - target) / (N * RMSE)
        grad_input = (self.pred - self.target) / (self.rmse * self.numel)
        return grad_input * grad_output

    def __repr__(self):
        return "nn_Loss_RMSE(Root Mean Square Error Loss)."


# Alias for nn_Loss_RMSE
RMSE = nn_Loss_RMSE


# Implementation of Mean Absolute Error Loss
class nn_Loss_MAE(nn_Loss_BaseLoss):
    """
    Mean Absolute Error Loss.

    The Mean Absolute Error (MAE) loss function quantifies the average absolute 
    difference between predicted values and true values. It is widely used in 
    regression tasks and is defined as: 

    Formula: L = \frac{1}{n} \sum_{i=1}^{n} |y_{true,i} - y_{pred,i}|

    Where:
        - $ n $ is the number of samples
        - $ y_{true,i} $ is the true value for sample i
        - $ y_{pred,i} $ is the predicted value for sample i

    MAE is less sensitive to outliers compared to Mean Squared Error (MSE), as it 
    uses absolute differences rather than squared differences. However, it is not 
    differentiable at zero, which can affect gradient-based optimization methods. 
    It provides an interpretable measure of error in the same units as the target 
    variable.

    Formula: L = \frac{1}{n} \sum_{i=1}^{n} |y_{true,i} - y_{pred,i}|

    """

    __attr__ = "MML.nn_Loss_MAE"

    def __init__(self,
                 *,
                 module_name: str = "nn_Loss_MAE",
                 backend: Literal["torch", "numpy"] = "torch",
                 dtype: type | str = None,
                 device: str | None = None,
                 autograd: bool = False,
                 **kwargs):
        """
        A Mean Absolute Error Loss Function.

        Parameters:
            module_name: str, The name of the module instance. Defaults to "nn_Loss_MAE".
            backend: Literal["torch", "numpy"], The computational backend to use. Defaults to "torch".
            dtype: type, The data type for the tensor values. Defaults to None (auto detection). 
                    For PyTorch, this corresponds to torch.dtype; for NumPy, it corresponds to np.dtype.
            device: str | None, The target device (e.g., "cpu", "cuda") where the layer's parameters will be placed. 
                    If None, uses the default device. Defaults to None (auto detection).
            autograd: bool, A flag indicating whether to use PyTorch's autograd for gradient computation. 
                    If True, manual gradient tracking is disabled. Defaults to False.

        Attributes:
            self.dtype: type, The data type of the tensor values.
            self.device: str | None, The device where the parameters are placed.
            self.autograd: bool, Whether PyTorch's autograd is enabled for this layer.
            self._parameters.weight: nn_Parameter, The weight parameter matrix (shape: [in_features, out_features]).
            self._parameters.bias: nn_Parameter | optional, The optional bias vector (shape: [out_features]) if has_bias is True.
        """

        super().__init__(module_name=module_name, backend=backend, dtype=dtype, device=device, autograd=autograd)

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """
        Compute the Mean Absolute Error (MAE) between predictions and targets.

        This method evaluates the average absolute difference 
        between predicted values (`pred`) and actual values (`target`), 
        offering a robust loss measure less sensitive to outliers.

        Args:
            pred (Tensor): Predicted tensor containing model outputs.
            target (Tensor): Target tensor containing ground truth values.

        Returns:
            Tensor: Scalar tensor representing the computed MSE loss.

        Raises:
            ValueError: If `pred` or `target` is not a valid MML.Tensor object, 
                        or if `pred` and `target` do not have the same shape.

        Attributes:
            self.numel (scalar): Total number of elements in the input tensors (product of tensor shapes).
            self.mae (Tensor): Saved Mean Absolute Error Tensor value for reference.
            self.pred (Tensor): Saved predicted tensor for backward computation.
            self.target (Tensor): Saved target tensor for backward computation.
            self.loss (Tensor): Save the computed loss for backpropagation uses.
        """

        # Type check, pred and target must be an instance of Tensor
        if isinstance(pred, Tensor) == False or isinstance(target, Tensor) == False:
            raise ValueError(f"In performing forward(), input `pred` or `target` must be in a MML `Tensor` format but you have {type(pred)} and {type(target)}")

        # Shape check, pred and target must have the same shape
        if pred.shape != target.shape:
            raise ValueError(f"In performing forward(), input `pred` and `target` must have the same shape, but you have {pred.shape} and {target.shape}")

        # Compute MAE in Tensor
        abs_diff = (pred - target).abs()
        mae = abs_diff.mean()

        # Save the pred, input, mse, rmse, and total number of elements for backward
        self.__setattr__("numel", np.array(pred.shape).prod())
        self.__setattr__("mae", mae)
        self.__setattr__("pred", pred)
        self.__setattr__("target", target)
        self.__setattr__("loss", [mae])

        return mae

    def backward(self, grad_output: Tensor | None = None) -> Tensor | None:
        """
        Backward pass: compute gradients for weight, bias, and input.

        This method performs the gradient computation for a module during backpropagation. 
        It calculates the gradients of the loss with respect to the weights, biases, and input tensor,
        based on the provided `grad_output` (gradient from the next layer). The implementation
        supports both PyTorch autograd and manual gradient calculation modes.

        Args:
            None: Since it is the first in calculating backward.

        Returns:
            Tensor: Gradient with respect to the input tensor, for recursive backward calculations in previous layers.

        """

        # If use autograd, pass; if manual mode, then calculate
        if self.autograd == True:
            self.loss[0].data.backward()
            return None

        # If grad_output is None (by default), assign it to 1
        if grad_output is None:
            grad_output = Tensor(1.0, backend=self.backend, dtype=self.dtype, device=self.device)
        
        # Else, it must be a scalar.
        else:
            if isinstance(grad_output, Tensor) == False:
                raise ValueError(f"In performing backward(), input `grad_output` must be in a MML `Tensor` format but you have {type(grad_output)}")
            if len(grad_output.shape) != 0:
                raise ValueError("In performing backward(), input `grad_output` must be in a MML `Tensor` with a scalar stored in")

        # dL/dpred = (pred - target).sign() / N
        grad_input = (self.pred - self.target).sign() / self.numel
        return grad_input * grad_output

    def __repr__(self):
        return "nn_Loss_MAE(Mean Absolute Error Loss)."


# Alias for nn_Loss_MAE
MAE = nn_Loss_MAE


# Implementation of Binary Cross Entropy Loss
class nn_Loss_BinaryCrossEntropy(nn_Loss_BaseLoss):
    """
    Binary Cross-Entropy for predictions in [0,1] and binary targets.

    The Binary Cross-Entropy loss measures the difference between predicted 
    probabilities (in [0,1]) and true binary labels (0 or 1). It is widely used 
    in binary classification tasks and is defined as: 

    Formula: L = -\frac{1}{n} \sum_{i=1}^{n} [y_{true,i} \log(p_i) + (1 - y_{true,i}) \log(1 - p_i)]

    Where:
        - $ n $ is the number of samples
        - $ y_{true,i} $ is the true binary label for sample i (0 or 1)
        - $ p_i $ is the predicted probability for sample i (in [0,1])

    This loss function penalizes incorrect predictions more heavily when the model 
    is confident but wrong. It is differentiable and suitable for optimization via 
    gradient-based methods. However, it requires care to avoid numerical instability 
    (e.g., adding a small epsilon to probabilities near 0 or 1).

    Formula: L = -\frac{1}{n} \sum_{i=1}^{n} [y_{true,i} \log(p_i) + (1 - y_{true,i}) \log(1 - p_i)]

    """

    __attr__ = "MML.nn_Loss_BinaryCrossEntropy"

    def __init__(self,
                 eps: float = 1e-5,
                 *,
                 module_name: str = "nn_Loss_BinaryCrossEntropy",
                 backend: Literal["torch", "numpy"] = "torch",
                 dtype: type | str = None,
                 device: str | None = None,
                 autograd: bool = False,
                 **kwargs):
        """
        A Binary Cross Entropy Loss Function for binary classification.

        Parameters:
            eps: float, The epsilon amount applied to clip() to avoid log(0).
            module_name: str, The name of the module instance. Defaults to "nn_Loss_BinaryCrossEntropy".
            backend: Literal["torch", "numpy"], The computational backend to use. Defaults to "torch".
            dtype: type, The data type for the tensor values. Defaults to None (auto detection). 
                    For PyTorch, this corresponds to torch.dtype; for NumPy, it corresponds to np.dtype.
            device: str | None, The target device (e.g., "cpu", "cuda") where the layer's parameters will be placed. 
                    If None, uses the default device. Defaults to None (auto detection).
            autograd: bool, A flag indicating whether to use PyTorch's autograd for gradient computation. 
                    If True, manual gradient tracking is disabled. Defaults to False.

        Attributes:
            self.eps: float, The epsilon value applied.
            self.dtype: type, The data type of the tensor values.
            self.device: str | None, The device where the parameters are placed.
            self.autograd: bool, Whether PyTorch's autograd is enabled for this layer.
            self._parameters.weight: nn_Parameter, The weight parameter matrix (shape: [in_features, out_features]).
            self._parameters.bias: nn_Parameter | optional, The optional bias vector (shape: [out_features]) if has_bias is True.
        """

        super().__init__(module_name=module_name, backend=backend, dtype=dtype, device=device, autograd=autograd)

        # Record the eps value
        self.__setattr__("eps", eps)

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """
        Compute the Binary Cross Entropy between predictions and targets.

        Args:
            pred (Tensor): Predicted tensor containing model outputs.
            target (Tensor): Target tensor containing ground truth values.

        Returns:
            Tensor: Scalar tensor representing the computed MSE loss.

        Raises:
            ValueError: If `pred` or `target` is not a valid MML.Tensor object, 
                        or if `pred` and `target` do not have the same shape.

        Attributes:
            self.n_classes (scalar): The number of classes to be classified.
            self.numel (scalar): Total number of elements in the input tensors (product of tensor shapes).
            self.pred (Tensor): Saved predicted tensor for backward computation.
            self.target (Tensor): Saved target tensor for backward computation.
            self.loss (Tensor): Save the computed loss for backpropagation uses.
        """

        # Type check, pred and target must be an instance of Tensor
        if isinstance(pred, Tensor) == False or isinstance(target, Tensor) == False:
            raise ValueError(f"In performing forward(), input `pred` or `target` must be in a MML `Tensor` format but you have {type(pred)} and {type(target)}")

        # Shape check, pred and target must have the same shape
        if pred.shape != target.shape:
            raise ValueError(f"In performing forward(), input `pred` and `target` must have the same shape, but you have {pred.shape} and {target.shape}")

        # n_classes check, must only have 1 dimension (DOES NOT SUPPORT ONE-HOT)
        if len(pred.shape) == 2:
            if pred.shape[1] != 1:
                raise ValueError(f"In performing forward(), input `pred` and `target` must be 1 dimensional or 2 dimension with the 2nd one be 1, but you have {pred.shape} and {target.shape}")

        # Compute clipped predictions to avoid log0
        clipped_pred = pred.clip(self.eps, 1-self.eps)

        # Compute binary cross-entropy loss
        loss = - (target * clipped_pred.log() + (1 - target) * (1 - clipped_pred).log())
        loss = loss.mean()

        # Save the pred, input etc for backward
        self.__setattr__("n_classes", 1)
        self.__setattr__("numel", np.array(clipped_pred.shape).prod())
        self.__setattr__("pred", clipped_pred)
        self.__setattr__("target", target)
        self.__setattr__("loss", [loss])

        return loss

    def backward(self, grad_output: Tensor | None = None) -> Tensor | None:
        """
        Backward pass: compute gradients for weight, bias, and input.

        This method performs the gradient computation for a module during backpropagation. 
        It calculates the gradients of the loss with respect to the weights, biases, and input tensor,
        based on the provided `grad_output` (gradient from the next layer). The implementation
        supports both PyTorch autograd and manual gradient calculation modes.

        Args:
            None: Since it is the first in calculating backward.

        Returns:
            Tensor: Gradient with respect to the input tensor, for recursive backward calculations in previous layers.

        """

        # If use autograd, pass; if manual mode, then calculate
        if self.autograd == True:
            self.loss[0].data.backward()
            return None
        
        # If grad_output is None (by default), assign it to 1
        if grad_output is None:
            grad_output = Tensor(1.0, backend=self.backend, dtype=self.dtype, device=self.device)
        
        # Else, it must be a scalar.
        else:
            if isinstance(grad_output, Tensor) == False:
                raise ValueError(f"In performing backward(), input `grad_output` must be in a MML `Tensor` format but you have {type(grad_output)}")
            if len(grad_output.shape) != 0:
                raise ValueError("In performing backward(), input `grad_output` must be in a MML `Tensor` with a scalar stored in")

        # dL/dpred = -(target/pred - (1-target)/(1-pred)) / N
        grad_input = - (self.target / self.pred) + ((1 - self.target) / (1 - self.pred))
        grad_input = grad_input / self.numel
        return grad_input * grad_output

    def __repr__(self):
        return "nn_Loss_BinaryCrossEntropy(Binary Cross Entropy Loss)."


# Alias for nn_Loss_BinaryCrossEntropy
BinaryCrossEntropy = nn_Loss_BinaryCrossEntropy


# Implementation of Multi Cross Entropy Loss
class nn_Loss_MultiCrossEntropy(nn_Loss_BaseLoss):
    """
    Multi Cross-Entropy for predictions in one_hot and probabilities targets.

    The Multi-Class Cross-Entropy loss measures the difference between predicted 
    probability distributions (for multiple classes) and true one-hot encoded labels. 
    It is widely used in multi-class classification tasks and is defined as: 

    Formula: L = -\frac{1}{n} \sum_{i=1}^{n} \sum_{c=1}^{C} y_{true,i,c} \log(p_{i,c})

    Where:
        - $ n $ is the number of samples
        - $ C $ is the number of classes
        - $ y_{true,i,c} $ is the one-hot encoded true label for sample i (1 if class c is correct, 0 otherwise)
        - $ p_{i,c} $ is the predicted probability for sample i belonging to class c

    This loss function penalizes incorrect predictions by measuring the discrepancy between 
    the true distribution (one-hot) and the predicted distribution. It is differentiable and 
    suitable for optimization via gradient-based methods. However, numerical stability 
    must be ensured (e.g., adding a small epsilon to probabilities near 0 or 1) to avoid log(0).

    Formula: L = -\frac{1}{n} \sum_{i=1}^{n} \sum_{c=1}^{C} y_{true,i,c} \log(p_{i,c})

    """

    __attr__ = "MML.nn_Loss_MultiCrossEntropy"

    def __init__(self,
                 eps: float = 1e-5,
                 raw_logits: bool = True,
                 *,
                 module_name: str = "nn_Loss_MultiCrossEntropy",
                 backend: Literal["torch", "numpy"] = "torch",
                 dtype: type | str = None,
                 device: str | None = None,
                 autograd: bool = False,
                 **kwargs):
        """
        A Binary Cross Entropy Loss Function for multi classification.

        Parameters:
            eps: float, The epsilon amount applied to clip() to avoid log(0).
            raw_logits: bool, If True, then not applied softmax, or applied softmax.
            module_name: str, The name of the module instance. Defaults to "nn_Loss_MultiCrossEntropy".
            backend: Literal["torch", "numpy"], The computational backend to use. Defaults to "torch".
            dtype: type, The data type for the tensor values. Defaults to None (auto detection). 
                    For PyTorch, this corresponds to torch.dtype; for NumPy, it corresponds to np.dtype.
            device: str | None, The target device (e.g., "cpu", "cuda") where the layer's parameters will be placed. 
                    If None, uses the default device. Defaults to None (auto detection).
            autograd: bool, A flag indicating whether to use PyTorch's autograd for gradient computation. 
                    If True, manual gradient tracking is disabled. Defaults to False.

        Attributes:
            self.eps: float, The epsilon value applied.
            self.raw_logits: bool, Whether raw logits or not (not applied softmax or not).
            self.dtype: type, The data type of the tensor values.
            self.device: str | None, The device where the parameters are placed.
            self.autograd: bool, Whether PyTorch's autograd is enabled for this layer.
            self._parameters.weight: nn_Parameter, The weight parameter matrix (shape: [in_features, out_features]).
            self._parameters.bias: nn_Parameter | optional, The optional bias vector (shape: [out_features]) if has_bias is True.
        """

        super().__init__(module_name=module_name, backend=backend, dtype=dtype, device=device, autograd=autograd)

        # Record the eps value
        self.__setattr__("eps", eps)

        # Record the status whether it is raw logits
        self.__setattr__("raw_logits", raw_logits)

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """
        Compute the Multi-class Cross Entropy between predictions and targets.

        Args:
            pred (Tensor): Predicted tensor containing model outputs.
            target (Tensor): Target tensor containing ground truth values.

        Returns:
            Tensor: Scalar tensor representing the computed MSE loss.

        Raises:
            ValueError: If `pred` or `target` is not a valid MML.Tensor object, 
                        or if `pred` and `target` do not have the same shape.

        Attributes:
            self.n_classes (scalar): The number of classes to be classified.
            self.n_samples (scalar): Total number of samples in the data.
            self.pred (Tensor): Saved predicted tensor for backward computation.
            self.pred_logprobs (Tensor): Saved log probabilities tensor for backward computation.
            self.target (Tensor): Saved target tensor for backward computation.
            self.target_multiclass (Tensor): Save target but in multiclass form for backward computation.
            self.loss (Tensor): Save the computed loss for backpropagation uses.
        """

        # Type check, pred and target must be an instance of Tensor
        if isinstance(pred, Tensor) == False or isinstance(target, Tensor) == False:
            raise ValueError(f"In performing forward(), input `pred` or `target` must be in a MML `Tensor` format but you have {type(pred)} and {type(target)}")

        # Shape check, pred and target must have the same shape
        if pred.shape != target.shape:
            raise ValueError(f"In performing forward(), input `pred` and `target` must have the same shape, but you have {pred.shape} and {target.shape}")

        # n_classes check, must have 2 dim and greater than 1 2nd dim
        if len(pred.shape) != 2:
            raise ValueError(f"In performing forward(), input `pred` and `target` must be 2 dimensional, but you have {pred.shape} and {target.shape}")
        if pred.shape[1] <= 1:
            raise ValueError(f"In performing forward(), input `pred` and `target` must have greater than 1 outputs in Multi Cross Entropy, but you have {pred.shape} and {target.shape}")

        # Input shape: (N, C) where C = number of classes
        # Target shape: (N, C) with one-hot encoded
        if self.raw_logits == True:
            # Compute log-softmax for numerical stability (log probabilities)
            pred_log_probs = pred.softmax(axis=1).clip(self.eps).log()
        else:
            # Input is probabilities; take log
            pred_log_probs = pred.clip(self.eps).log()

        # Turn the true one_hot result into a multi-class result (N, 1)
        target_multiclass = self._to_labels(target)

        # Gather the log probs along axis 1 by true indices
        gathered_log_probs = pred_log_probs.gather_along(
            pred_log_probs, axis=1, index=target_multiclass)

        # Compute negative log-likelihood loss for each sample and take mean
        losses = -gathered_log_probs
        loss = losses.mean()

        # Save the pred, input etc for backward
        self.__setattr__("n_classes", pred.shape[1])
        self.__setattr__("n_samples", pred.shape[0])
        self.__setattr__("pred", pred)
        self.__setattr__("pred_logprobs", pred_log_probs)
        self.__setattr__("target", target)
        self.__setattr__("target_multiclass", target_multiclass)
        self.__setattr__("loss", [loss])

        return loss

    def backward(self, grad_output: Tensor | None = None) -> Tensor | None:
        """
        Backward pass: compute gradients for weight, bias, and input.

        This method performs the gradient computation for a module during backpropagation. 
        It calculates the gradients of the loss with respect to the weights, biases, and input tensor,
        based on the provided `grad_output` (gradient from the next layer). The implementation
        supports both PyTorch autograd and manual gradient calculation modes.

        Args:
            None: Since it is the first in calculating backward.

        Returns:
            Tensor: Gradient with respect to the input tensor, for recursive backward calculations in previous layers.

        """

        # If use autograd, pass; if manual mode, then calculate
        if self.autograd == True:
            self.loss[0].data.backward()
            return None
        
        # If grad_output is None (by default), assign it to 1
        if grad_output is None:
            grad_output = Tensor(1.0, backend=self.backend, dtype=self.dtype, device=self.device)
        
        # Else, it must be a scalar.
        else:
            if isinstance(grad_output, Tensor) == False:
                raise ValueError(f"In performing backward(), input `grad_output` must be in a MML `Tensor` format but you have {type(grad_output)}")
            if len(grad_output.shape) != 0:
                raise ValueError("In performing backward(), input `grad_output` must be in a MML `Tensor` with a scalar stored in")

        # Initialize grad_input with same shape as predictions
        grad_input = self.pred_logprobs.to_zeros()

        # Calculate the backward gradients
        if self.raw_logits == True:
            # grad = (softmax_prob - one_hot(target)) / N
            grad_input = self.pred_logprobs.exp() - self.target
            grad_input /= self.n_samples
        else:
            # For probabilities: grad = -1/p_target for target class, 0 for others, divided by N
            grad_input = -self.target / (self.pred_logprobs.exp() + self.eps)
            grad_input /= self.n_samples

        return grad_input * grad_output

    def __repr__(self):
        return f"nn_Loss_MultiCrossEntropy(Multi Cross Entropy Loss, with n_classes = {self.n_classes})."


# Alias for nn_Loss_BinaryCrossEntropy
MultiCrossEntropy = nn_Loss_MultiCrossEntropy


# Test case of Loss
if __name__ == "__main__":

    from nn import Tensor
    from nn import Dense
    from nn import Softmax
    from nn import Module, nn_Module
    from nn import MSE, MultiCrossEntropy

    ##############################################
    #
    # Regression Test
    class reg_test(Module):

        def __init__(self, **kwargs):

            super().__init__(module_name="reg_test", **kwargs)
            self.dense = Dense(4, 2, True, **kwargs)
            self.sumover = Dense(2, 1, True, **kwargs)

        def forward(self, inputs):
            out = self.dense.forward(inputs)
            out = self.sumover.forward(out)
            return out

    inputs = Tensor([[1, 2, 3, 4.], [2, 3, 4, 5.]], backend="torch")
    label = Tensor([[0.002312], [0.991215]], backend="torch")

    # Test forward
    x = reg_test()
    x.train()
    out = x.forward(inputs)
    
    # Calculate loss
    crit = MSE()
    loss = crit(out, label)
    lossgrad = crit.backward()

    # Test backward
    x.backward(lossgrad)
    
    from nn import Tensor
    from nn import Dense
    from nn import Softmax
    from nn import Module, nn_Module
    from nn import MSE, MultiCrossEntropy
    
    ##############################################
    #
    # Classification Test
    class cls_test(Module):

        def __init__(self, **kwargs):

            super().__init__(module_name="cls_test", **kwargs)
            self.dense = Dense(4, 16, True, **kwargs)
            self.dense2 = Dense(16, 8, False, **kwargs)
            self.dense3 = Dense(8, 2, False, **kwargs)

        def forward(self, inputs):
            out = self.dense.forward(inputs)
            out = self.dense2.forward(out)
            out = self.dense3.forward(out)
            return out

    inputs = Tensor([[1, 2, 3, 4.], [-1,-7,14,9]], backend="torch").requires_grad_(True)
    label = Tensor([[0, 1], [1, 0]], backend="torch")

    # Test forward
    x = cls_test()
    x.train()
    out = x.forward(inputs)
    
    # Calculate loss
    crit = MultiCrossEntropy()
    loss = crit(out, label)
    lossgrad = crit.backward()

    # Test backward
    x.backward(lossgrad)
    