# nn_layer_flatten.py
#
# A Flatten Functionality Layer Implementation
# From MML Library by Nathmath

import numpy as np
try:
    import torch
except ImportError:
    torch = None

from typing import Any, Literal
    
from .objtyp import Object
from .tensor import Tensor

from .nn_parameter import nn_Parameter
from .nn_module import nn_Module


# Implementation of Flatten Layer (Make Tensor Flatten)
class nn_Layer_Flatten(nn_Module):
    """
    Flatten Layer Implementation
    
    Flatten Layer just turn the input Tensor into a flatten 2D tensor (batch_size, features),
    very suitable for transforming high dimensional data into low dimension ones and then apply
    Dense and Dropout layers. It does not have any learnable parameters.
    """

    __attr__ = "MML.nn_Layer_Flatten"
    
    def __init__(self, 
                 *,
                 module_name: str = "nn_Layer_Flatten", 
                 backend: Literal["torch", "numpy"] = "torch",
                 dtype: type | str = None,
                 device: str | None = None,
                 autograd: bool = False,
                 **kwargs):
        """
        A training active dropout layer.
        
        This class implements a dropout neural network layer, which performs randomly dropout when training and 
        do nothing in evaluation process. Dropping out is controlled by a dropout rate which is typically ranging from
        0 to 1 and common values are [0.1, 0.4].

        Parameters:
            module_name: str, The name of the module instance. Defaults to "nn_Layer_Dense".
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
        
        super().__init__(module_name = module_name, backend = backend, dtype = dtype, device = device, autograd = autograd)
        
        # Record an empty tuple showing the original shape of the input
        self.__setattr__("original_shape", ())

    def forward(self, x: Tensor) -> Tensor:
        """
        Reshape the input tensor into a 2D format (batch_size, features).
        If the input is high dimensional, except the 1st dimension, which is batch_size,
        any other dimensions will be flatten into a flatten tensor.
        The output of this Layer will always be a 2D Tensor.

        Args:
            x (Tensor): Input tensor with arbitrary dimensions. The first dimension represents the batch size.

        Returns:
            Tensor: Output tensor reshaped into 2D format (batch_size, total_features).

        Raises:
            ValueError: If the input `x` is not a valid MML.Tensor object.
        """
        # Type check, x must be an instance of Tensor
        if isinstance(x, Tensor) == False:
            raise ValueError(f"In performing forward(), input `x` must be in a MML `Tensor` format but you have {type(x)}")
        
        # Save the original input shape for backward pass
        self.original_shape = x.shape

        # Flatten all dimensions except the batch dimension (dim 0)
        out = x.reshape([x.shape[0], -1])
        # Reshape to (batch_size, total_features)
        return out

    def backward(self, grad_output: Tensor) -> Tensor:
        """
        Backward pass: compute gradients for weight, bias, and input.

        This method performs the gradient computation for a dense layer during backpropagation. 
        It calculates the gradients of the loss with respect to the weights, biases, and input tensor,
        based on the provided `grad_output` (gradient from the next layer). The implementation
        supports both PyTorch autograd and manual gradient calculation modes.

        Args:
            grad_output (Tensor): Gradient tensor resulting from the output of the layer, used as input for backpropagation.

        Returns:
            Tensor: Gradient with respect to the input tensor, for recursive backward calculations in previous layers.

        Raises:
            ValueError: If `grad_output` is not a valid MML.Tensor object..
        """
        # If use autograd, pass; if manual mode, then calculate
        if self.autograd == True:
            return None
        
        # Type check, grad_output must be an instance of Tensor
        if isinstance(grad_output, Tensor) == False:
            raise ValueError(f"In performing backward(), `grad_output` must be in a MML `Tensor` format but you have {type(grad_output)}")

        # Reshape gradient back to original shape
        return grad_output.reshape(self.original_shape)

    def __repr__(self):
        return f"nn_Layer_Flatten(with original shape {self.original_shape})."


# Alias for nn_Layer_Flatten
Flatten = nn_Layer_Flatten


# Test case of Flatten
if __name__ == "__main__":
    
    x = Flatten()
    
    inputs = Tensor([[[1,2],[3,4.]],[[2,3],[4,5]]], backend="torch")
    
    # Set to train mode
    x.train()
    
    # Test forward
    x.forward(inputs)
    
    # Test backward (use random numbers as grads)
    x.backward(x.forward(inputs))
    