# nn_layer_dense.py
#
# A Dense Neural Network Layer Implementation
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


# Implementation of Dense Layer (Fully Connected Layer)
class nn_Layer_Dense(nn_Module):
    """
    Dense Layer (Fully-Connected Layer) Implementation
    
    This class serves as the foundation for implementing fully connected (dense)
    neural network layers. It contains weight and bias in nn_Parameter containers
    and ready to perform forward() and backward() pass to perform MLP tasks.    
    """
    
    __attr__ = "MML.nn_Layer_Dense"    
    
    def __init__(self, 
                 in_features: int = 1, 
                 out_features: int = 1, 
                 has_bias: str = True,
                 init_scale: float = 0.1,
                 *,
                 module_name: str = "nn_Layer_Dense", 
                 backend: Literal["torch", "numpy"] = "torch",
                 dtype: type | str = None,
                 device: str | None = None,
                 autograd: bool = False,
                 **kwargs):
        """
        A fully connected layer: y = x @ W + b.
        
        This class implements a dense (fully connected) neural network layer, which performs a linear transformation
        on input data followed by an optional bias addition. It is designed to be compatible with both PyTorch and NumPy backends,
        supporting automatic gradient computation via autograd or manual gradient tracking.

        Parameters:
            in_features: int, The number of input features for this layer. Defaults to 1.
            out_features: int, The number of output features for this layer. Defaults to 1.
            has_bias: str, A flag indicating whether to include a bias term. Valid values are "True" or "False".
                    If set to "True", the layer includes an additive bias term (b). Defaults to "True".
            module_name: str, The name of the module instance. Defaults to "nn_Layer_Dense".
            backend: Literal["torch", "numpy"], The computational backend to use. Defaults to "torch".
            dtype: type, The data type for the tensor values. Defaults to None (auto detection). 
                    For PyTorch, this corresponds to torch.dtype; for NumPy, it corresponds to np.dtype.
            device: str | None, The target device (e.g., "cpu", "cuda") where the layer's parameters will be placed. 
                    If None, uses the default device. Defaults to None (auto detection).
            autograd: bool, A flag indicating whether to use PyTorch's autograd for gradient computation. 
                    If True, manual gradient tracking is disabled. Defaults to False.

        Attributes:
            self.in_features: int, The number of input features for this layer.
            self.out_features: int, The number of output features for this layer.
            self.has_bias: str, A flag indicating whether a bias term is included ("True" or "False").
            self.init_scale: float, A floatting number indicating the maximum value of initial random weights.
            self.backend: Literal["torch", "numpy"], The computational backend used by the layer.
            self.dtype: type, The data type of the tensor values.
            self.device: str | None, The device where the parameters are placed.
            self.autograd: bool, Whether PyTorch's autograd is enabled for this layer.
            self._parameters.weight: nn_Parameter, The weight parameter matrix (shape: [in_features, out_features]).
            self._parameters.bias: nn_Parameter | optional, The optional bias vector (shape: [out_features]) if has_bias is True.
        """
        
        super().__init__(module_name = module_name, backend = backend, dtype = dtype, device = device, autograd = autograd)
        
        # Shape Notation
        # Input: (in_features)
        # Output: (out_features)
        # self._parameters["weight"]: (in_features, out_features)
        # self._parameters["bias"]: (out_features)
        
        # Record shapes etc
        self.__setattr__("in_features", in_features)
        self.__setattr__("out_features", out_features)
        self.__setattr__("has_bias", has_bias)
        self.__setattr__("init_scale", init_scale)
        
        # Initialize weight and bias parameters
        self.__setattr__("weight", nn_Parameter(
            Tensor.rand([in_features, out_features], backend=backend, dtype=dtype, device=device) * init_scale,
            requires_grad = True,
            dtype = None,
            device = None,
            autograd = autograd)
            )
        
        if has_bias == True:
            # If uses bias, then set the bias
            self.__setattr__("bias", nn_Parameter(
                Tensor.zeros([out_features], backend=backend, dtype=dtype, device=device),
                requires_grad = True,
                dtype = None,
                device = None,
                autograd = autograd)
                )
            
        return

    def forward(self, x: Tensor) -> Tensor:
        """
        Compute the layer output: out = x @ W + b and return the out.

        This method performs the forward computation for a dense neural network layer,
        computing the linear transformation `out = x @ W + b`, where `W` is the weight matrix
        and `b` is the bias vector. The input tensor `x` is processed through this operation,
        and the result is returned as the output of the layer.

        Args:
            x (Tensor): Input tensor of shape (batch_size, in_features) to be transformed by the layer.

        Returns:
            Tensor: Output tensor of shape (batch_size, out_features) after applying the dense transformation.

        Raises:
            ValueError: If the input `x` is not a valid MML.Tensor object.

        Attributes:
            self.input (Tensor): The input tensor saved for use in backward propagation.
        """
        
        # Type check, x must be an instance of Tensor
        if isinstance(x, Tensor) == False:
            raise ValueError(f"In performing forward(), input `x` must be in a MML `Tensor` format but you have {type(x)}")
        
        # Save input for backward
        self.__setattr__("input", x)
        
        # Perform forward pass x @ W + b
        out = x @ self._parameters["weight"].data
        if self.has_bias == True:
            out += self._parameters["bias"].data
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
        
        # Gradient wrt. weight: X^T * grad_output
        self._parameters["weight"].grad = self.input.transpose() @ grad_output
        
        # Gradient wrt. bias: sum grad_output over batch dimension
        if self.has_bias == True:
            # Sum over the samples to get the gradients
            self._parameters["bias"].grad = grad_output.sum(axis = 0)
        
        # Gradient wrt. input: grad_output * W^T
        grad_input = grad_output @ self._parameters["weight"].data.transpose()
        
        # Return the gradient with respect to input for recursive backward calculation
        return grad_input
    
    def __repr__(self):
        return f"nn_Layer_Dense(shape: ({self.in_features}, {self.out_features}) with{'out' if self.has_bias == False else ''} bias)."
    
    
# Alias for nn_Layer_Dense
Dense = nn_Layer_Dense


# Test case of Dense
if __name__ == "__main__":
    
    x = Dense(4, 2, True, backend="torch", dtype = torch.float64, device = "cuda")
    
    inputs = Tensor([[1,2,3,4.],[2,3,4,5]], backend="torch", dtype = torch.float64, device = "cuda")
    
    # Set to train mode
    x.train()
    
    # Test forward
    x.forward(inputs)
    
    # Test backward (use random numbers as grads)
    x.backward(x.forward(inputs))
