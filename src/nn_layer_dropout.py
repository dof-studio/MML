# nn_layer_dropout.py
#
# A Dropout Masking Layer Implementation
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


# Implementation of Dropout Layer (Masked Layer)
class nn_Layer_Dropout(nn_Module):
    """
    Dropout Layer (Masked Layer) Implementation
    
    Dropout layer that zeros out inputs with probability p during training, and 
    it will not mask anything in non-training mode.
    Dropout layer can be used to improve anti-over-fitting capabilities of your model.
    And it is compatible for any kind of Tensors with any shape (not only a 2D).
    It does not have any learnable parameters.
    """
    
    __attr__ = "MML.nn_Layer_Dropout"  
    
    def __init__(self, 
                 p: float = 0.1,
                 *,
                 module_name: str = "nn_Layer_Dropout", 
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
            p: float, The ratio of dropout when training the neural network. By default, it is 0.1.
            module_name: str, The name of the module instance. Defaults to "nn_Layer_Dense".
            backend: Literal["torch", "numpy"], The computational backend to use. Defaults to "torch".
            dtype: type, The data type for the tensor values. Defaults to None (auto detection). 
                    For PyTorch, this corresponds to torch.dtype; for NumPy, it corresponds to np.dtype.
            device: str | None, The target device (e.g., "cpu", "cuda") where the layer's parameters will be placed. 
                    If None, uses the default device. Defaults to None (auto detection).
            autograd: bool, A flag indicating whether to use PyTorch's autograd for gradient computation. 
                    If True, manual gradient tracking is disabled. Defaults to False.

        Attributes:
            self.dropout_p: float, the dropout rate specified and used in training.
            self.dtype: type, The data type of the tensor values.
            self.device: str | None, The device where the parameters are placed.
            self.autograd: bool, Whether PyTorch's autograd is enabled for this layer.
            self._parameters.weight: nn_Parameter, The weight parameter matrix (shape: [in_features, out_features]).
            self._parameters.bias: nn_Parameter | optional, The optional bias vector (shape: [out_features]) if has_bias is True.
        """
        
        super().__init__(module_name = module_name, backend = backend, dtype = dtype, device = device, autograd = autograd)
        
        # Record the dropout rate as a non-trainable parameter
        self.__setattr__("dropout_p", p)
        
        # Record an empty tuple showing the shape of the mask
        self.__setattr__("shape", ())

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply dropout during forward pass.

        This method implements the forward computation of a dropout layer, which randomly sets elements of the input tensor to zero
        with probability `p` during training. In evaluation mode, it returns the input unchanged. The dropout mask is stored for use
        in backpropagation during training.

        Args:
            x (Tensor): Input tensor to apply dropout to. Shape should match the expected dimensions for the layer.

        Returns:
            Tensor: Output tensor after applying dropout. During training, this tensor has elements randomly zeroed out and scaled.
                   In evaluation mode, it returns the input tensor unchanged.

        Raises:
            ValueError: If the input `x` is not a valid MML.Tensor object.
        """

        # Type check, x must be an instance of Tensor
        if isinstance(x, Tensor) == False:
            raise ValueError(f"In performing forward(), input `x` must be in a MML `Tensor` format but you have {type(x)}")
        
        # Mask it in training mode
        if self.training == True:
            
            # Save the shape of the mask
            self.shape = x.shape
            
            # Create a dropout mask: 1 with probability (1-p), 0 with probability p
            mask = Tensor.rand(x.shape, backend=self.backend, dtype=self.dtype, device=self.device)
            mask.data = mask.data >= self.dropout_p
            mask.astype(self.dtype)
            
            # Scale mask by 1 / (1-p) to keep expectation the same
            mask = mask / (1 - self.dropout_p)
            
            # Save the mask as an attribute (non-parameter attribute)
            self.__setattr__("mask", mask)
            
            return x * mask
        
        # Do nothing in evaluation mode
        else:
            return x

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
        
        # If training, apply the same mask to the gradient
        if self.training:
            return grad_output * self.mask
        
        # Else, return identity
        else:
            return grad_output
        
    def __repr__(self):
        return f"nn_Layer_Dropout(shape: {self.shape} with probability {round(self.dropout_p)})."
    
    
# Alias for nn_Layer_Dropout
Dropout = nn_Layer_Dropout


# Test case of Dropout
if __name__ == "__main__":
    
    x = Dropout()
    
    inputs = Tensor([[1,2,3,4.],[2,3,4,5]], backend="torch")
    
    # Set to train mode
    x.train()
    
    # Test forward
    x.forward(inputs)
    
    # Test backward (use random numbers as grads)
    x.backward(x.forward(inputs))